# PF Pose + LiDAR Occupancy Grid Mapping + CostMap + A* + Frontier Exploration + Metrics

import random
import numpy as np
import matplotlib.pyplot as plt

from ekf_localization import Robot
from particle_filter_localization import Particle, ParticleFilter
from ocuupancy_grid_mapping import OccupancyGridMap
from astar_planner import AStarPlanner
from costmap import CostMap
from frontier_detector import FrontierDetector
from exploration_metrics import ExplorationMetrics
from visited_map import VisitedMap
from uncertainty import ParticleUncertainty
from experiment_logger import ExperimentLogger
from frontier_memory import FrontierMemory


random.seed(42)
np.random.seed(42)


def point_to_segment_distance(px, py, ax, ay, bx, by):
    ap_x = px - ax
    ap_y = py - ay

    ab_x = bx - ax
    ab_y = by - ay

    ab_len_sq = ab_x**2 + ab_y**2

    if ab_len_sq == 0:
        return np.sqrt(ap_x**2 + ap_y**2)

    t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
    t = max(0, min(1, t))

    closest_x = ax + t * ab_x
    closest_y = ay + t * ab_y

    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)


class LidarSensor:
    def __init__(self, max_range, num_beams, field_of_view):
        self.max_range = max_range
        self.num_beams = num_beams
        self.field_of_view = field_of_view

    def scan(self, robot_x, robot_y, robot_theta, obstacles, walls):
        beams = []

        beam_angles = np.linspace(
            -self.field_of_view / 2,
            self.field_of_view / 2,
            self.num_beams
        )

        step_size = 0.1
        wall_threshold = 0.15

        for angle in beam_angles:
            beam_theta = robot_theta + angle
            hit_found = False

            for distance in np.arange(0, self.max_range, step_size):
                ray_x = robot_x + distance * np.cos(beam_theta)
                ray_y = robot_y + distance * np.sin(beam_theta)

                for obstacle in obstacles:
                    ox = obstacle["x"]
                    oy = obstacle["y"]
                    radius = obstacle["radius"]

                    if np.sqrt((ray_x - ox) ** 2 + (ray_y - oy) ** 2) < radius:
                        beams.append({"x": ray_x, "y": ray_y, "hit": True})
                        hit_found = True
                        break

                if hit_found:
                    break

                for wall in walls:
                    (ax, ay), (bx, by) = wall

                    distance_to_wall = point_to_segment_distance(
                        ray_x, ray_y, ax, ay, bx, by
                    )

                    if distance_to_wall < wall_threshold:
                        beams.append({"x": ray_x, "y": ray_y, "hit": True})
                        hit_found = True
                        break

                if hit_found:
                    break

            if not hit_found:
                max_x = robot_x + self.max_range * np.cos(beam_theta)
                max_y = robot_y + self.max_range * np.sin(beam_theta)

                beams.append({"x": max_x, "y": max_y, "hit": False})

        return beams


# -------------------------
# Setup
# -------------------------

landmarks = [(45, 45), (60, 50), (70, 65)]

obstacles = [
    {"x": 65, "y": 65, "radius": 2.0},
    {"x": 45, "y": 70, "radius": 2.5},
    {"x": 75, "y": 45, "radius": 2.0}
]

walls = [
    ((35, 35), (80, 35)),
    ((80, 35), (80, 80)),
    ((80, 80), (35, 80)),
    ((35, 80), (35, 35))
]

grid_map = OccupancyGridMap(
    width=100,
    height=100,
    resolution=0.25
)

visited_map = VisitedMap(
    occupancy_grid=grid_map,
    visit_radius=6
)

frontier_memory = FrontierMemory(
    reject_radius=3.0
)

costmap = CostMap(
    occupancy_grid=grid_map,
    occupancy_threshold=0.65,
    inflation_radius=6,
    cost_radius=8,
    cost_weight=4.0
)

frontier_detector = FrontierDetector(
    occupancy_grid=grid_map,
    visited_map=visited_map,
    strategy="info_gain",
    min_cluster_size=8
)

metrics = ExplorationMetrics(
    occupancy_grid=grid_map
)

logger = ExperimentLogger(
    filename=r"C:\Users\Dr.Subhash Kumar\Documents\Algorithms\info_gain.csv"
)

planner = AStarPlanner(
    occupancy_grid=grid_map,
    costmap=costmap
)

lidar = LidarSensor(
    max_range=25,
    num_beams=120,
    field_of_view=2 * np.pi
)

num_particles = 1000
particles = []

for _ in range(num_particles):
    particles.append(
        Particle(
            x=random.uniform(35, 65),
            y=random.uniform(35, 65),
            theta=random.uniform(-np.pi, np.pi),
            weight=1.0 / num_particles
        )
    )

pf = ParticleFilter(particles)
uncertainty_estimator = ParticleUncertainty(pf)

robot = Robot(50, 50, 0)

v = 0.0
omega = 0.0
dt = 1.0
num_steps = 500

goal_world = None
planned_path = None

true_path_x = []
true_path_y = []

estimate_path_x = []
estimate_path_y = []

current_waypoint_index = 0
waypoint_tolerance = 0.5

mapping_steps = 10
replan_interval = 10
goal_reached_tolerance = 2.0

frontier_cells = []
frontier_clusters = []
scan_in_place_steps = 0
max_scan_failures = 5
scan_failures = 0
exploration_finished = False


def compute_control_to_waypoint(robot, waypoint, max_v=1.0, max_omega=0.3):
    target_x, target_y = waypoint

    dx = target_x - robot.x
    dy = target_y - robot.y

    distance = np.sqrt(dx**2 + dy**2)

    desired_theta = np.arctan2(dy, dx)

    heading_error = desired_theta - robot.theta
    heading_error = np.arctan2(
        np.sin(heading_error),
        np.cos(heading_error)
    )

    alignment = np.exp(-2.0 * abs(heading_error))
    v = min(max_v, distance) * alignment
    omega = max(-max_omega, min(max_omega, heading_error))

    return v, omega, distance


# -------------------------
# Simulation
# -------------------------

plt.figure(figsize=(18, 18))

for step in range(num_steps):
    if exploration_finished:
        break
     
    v = 0.0
    omega = 0.0
    if goal_world is not None:
        distance_to_goal = np.sqrt(
            (robot.x - goal_world[0]) ** 2 +
            (robot.y - goal_world[1]) ** 2
        )
    else:
        distance_to_goal = float("inf")
    
    if scan_in_place_steps >0 :
        v = 0.0
        omega = 0.4
        scan_in_place_steps -= 1
        print(f"Step {step}: Scanning...")


    elif goal_world is not None and distance_to_goal < goal_reached_tolerance:
        v = 0.0
        omega = 0.0
        goal_world = None
        planned_path = None
        current_waypoint_index = 0

    elif planned_path is not None and len(planned_path) > 0:
        if current_waypoint_index >= len(planned_path):
            current_waypoint_index = len(planned_path) - 1

        waypoint = planned_path[current_waypoint_index]

        v, omega, distance_to_waypoint = compute_control_to_waypoint(
            robot,
            waypoint
        )

        if distance_to_waypoint < waypoint_tolerance:
            current_waypoint_index += 1

    else:
        v = 0.0
        omega = 0.0

    robot.move_robot(v, omega, dt)

    measurements = []

    for lx, ly in landmarks:
        dx = lx - robot.x
        dy = ly - robot.y

        true_range = np.sqrt(dx**2 + dy**2)

        true_bearing = np.arctan2(dy, dx) - robot.theta
        true_bearing = (true_bearing + np.pi) % (2 * np.pi) - np.pi

        noisy_range = true_range + random.gauss(0, 0.5)
        noisy_bearing = true_bearing + random.gauss(0, 0.05)

        measurements.append((noisy_range, noisy_bearing))

    pf.predict(v, omega, dt)
    pf.update(measurements, landmarks)

    ess = pf.effective_sample_size()

    if ess < num_particles / 2:
        pf.resample()

    estimated_pose = pf.estimate()

    visited_map.update(
        robot_x=estimated_pose[0],
        robot_y=estimated_pose[1]
    )

    position_uncertainty = uncertainty_estimator.position_uncertainty()

    true_path_x.append(robot.x)
    true_path_y.append(robot.y)

    estimate_path_x.append(estimated_pose[0])
    estimate_path_y.append(estimated_pose[1])

    scan_hits = lidar.scan(
        robot.x,
        robot.y,
        robot.theta,
        obstacles,
        walls
    )

    for beam in scan_hits:
        end_x = beam["x"]
        end_y = beam["y"]

        if beam["hit"]:
            grid_map.update_ray(
                estimated_pose[0],
                estimated_pose[1],
                end_x,
                end_y
            )
        else:
            grid_map.update_free_ray(
                estimated_pose[0],
                estimated_pose[1],
                end_x,
                end_y
            )

    if step >= mapping_steps and step % replan_interval == 0:
        costmap.build()

        frontier_cells = frontier_detector.detect_frontier_cells()
        frontier_clusters = frontier_detector.cluster_frontiers(frontier_cells)

        if goal_world is None:
            goal_world = frontier_detector.choose_frontier_goal(
                robot_world=(estimated_pose[0], estimated_pose[1]),
                uncertainty=position_uncertainty,
                frontier_memory=frontier_memory
            )

            if goal_world is None:
                scan_failures += 1
                if scan_failures >= max_scan_failures:
                    exploration_finished = True
                    print(f"Step {step}: Exploration finished. No Valid frontiers remain.")
                else:
                    scan_in_place_steps = 8
                    print(f"Step {step}: No frontier found, entering scan mode.")

               


            planned_path = None
            current_waypoint_index = 0

        if goal_world is not None:
            scan_failures = 0

            new_path = planner.plan(
                start_world=(estimated_pose[0], estimated_pose[1]),
                goal_world=goal_world
            )

            if new_path is not None:
                scan_failures = 0
                planned_path = new_path
                current_waypoint_index = min(5, len(planned_path) - 1)
            else:
                frontier_memory.add_failed_goal(goal_world)
                goal_world = None
                planned_path = None
                current_waypoint_index = 0

    current_metrics = metrics.update(
        robot_x=robot.x,
        robot_y=robot.y,
        frontier_cells=frontier_cells,
        frontier_clusters=frontier_clusters
    )

    logger.log(
        step=step,
        metrics=current_metrics,
        ess=ess,
        uncertainty=position_uncertainty
    )

    # -------------------------
    # Visualization
    # -------------------------

    plt.clf()

    probabilities = 1 - (1 / (1 + np.exp(grid_map.grid)))

    plt.imshow(
        probabilities,
        cmap="gray_r",
        origin="lower",
        extent=[0, grid_map.width, 0, grid_map.height],
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    plt.plot(true_path_x, true_path_y, label="True Path")
    plt.plot(estimate_path_x, estimate_path_y, label="Estimated Path")

    if planned_path is not None:
        path_x = [p[0] for p in planned_path]
        path_y = [p[1] for p in planned_path]

        plt.plot(
            path_x,
            path_y,
            linewidth=3,
            label="A* Planned Path"
        )

    for beam in scan_hits:
        plt.plot(
            [estimated_pose[0], beam["x"]],
            [estimated_pose[1], beam["y"]],
            linestyle="dashed",
            alpha=0.18
        )

    plt.scatter(robot.x, robot.y, color="red", s=100, label="True Robot")

    plt.scatter(
        estimated_pose[0],
        estimated_pose[1],
        color="green",
        s=100,
        label="PF Estimate"
    )

    if goal_world is not None:
        plt.scatter(
            goal_world[0],
            goal_world[1],
            color="magenta",
            s=150,
            marker="*",
            label="Frontier Goal"
        )

    obstacle_x = [o["x"] for o in obstacles]
    obstacle_y = [o["y"] for o in obstacles]
    obstacle_sizes = [o["radius"] * 500 for o in obstacles]

    plt.scatter(
        obstacle_x,
        obstacle_y,
        color="blue",
        s=obstacle_sizes,
        alpha=0.5,
        label="True Obstacles"
    )

    for i, wall in enumerate(walls):
        (ax, ay), (bx, by) = wall

        plt.plot(
            [ax, bx],
            [ay, by],
            color="black",
            linewidth=2,
            label="Walls" if i == 0 else None
        )

    hit_xs = [b["x"] for b in scan_hits if b["hit"]]
    hit_ys = [b["y"] for b in scan_hits if b["hit"]]

    miss_xs = [b["x"] for b in scan_hits if not b["hit"]]
    miss_ys = [b["y"] for b in scan_hits if not b["hit"]]

    plt.scatter(hit_xs, hit_ys, color="orange", s=35, label="LiDAR Hits")

    plt.scatter(
        miss_xs,
        miss_ys,
        color="purple",
        s=15,
        alpha=0.35,
        label="Max Range Misses"
    )

    plt.xlim(0, grid_map.width)
    plt.ylim(0, grid_map.height)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(
        f"Frontier Exploration | "
        f"Step {step + 1} | "
        f"ESS: {ess:.1f} | "
        f"Uncertainty: {position_uncertainty:.3f} | "
        f"Mapped: {current_metrics['mapped_percent']:.1f}% | "
        f"Frontiers: {current_metrics['frontier_clusters']} | "
        f"Distance: {current_metrics['distance_traveled']:.1f}"
    )

    plt.grid(True, alpha=0.25)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0
    )

    plt.pause(0.1)

plt.tight_layout()
plt.show()