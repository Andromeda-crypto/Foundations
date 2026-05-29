# frontier_detector.py

import numpy as np


class FrontierDetector:
    def __init__(
        self,
        occupancy_grid,
        visited_map=None,
        strategy="uncertainty_aware",
        free_threshold=0.35,
        unknown_low=0.45,
        unknown_high=0.55,
        min_cluster_size=8
    ):
        self.map = occupancy_grid
        self.visited_map = visited_map
        self.strategy = strategy
        self.free_threshold = free_threshold
        self.unknown_low = unknown_low
        self.unknown_high = unknown_high
        self.min_cluster_size = min_cluster_size

    def probability(self, gx, gy):
        log_odds = self.map.grid[gy, gx]
        return 1 - (1 / (1 + np.exp(log_odds)))

    def is_free(self, gx, gy):
        return self.probability(gx, gy) <= self.free_threshold

    def is_unknown(self, gx, gy):
        prob = self.probability(gx, gy)
        return self.unknown_low <= prob <= self.unknown_high

    def has_unknown_neighbor(self, gx, gy):
        directions = [
            (1, 0), (-1, 0),
            (0, 1), (0, -1),
            (1, 1), (1, -1),
            (-1, 1), (-1, -1)
        ]

        for dx, dy in directions:
            nx = gx + dx
            ny = gy + dy

            if not self.map.in_bounds(nx, ny):
                continue

            if self.is_unknown(nx, ny):
                return True

        return False

    def detect_frontier_cells(self):
        frontiers = []

        for gy in range(self.map.grid_height):
            for gx in range(self.map.grid_width):
                if self.is_free(gx, gy) and self.has_unknown_neighbor(gx, gy):
                    frontiers.append((gx, gy))

        return frontiers

    def cluster_frontiers(self, frontier_cells):
        frontier_set = set(frontier_cells)
        visited = set()
        clusters = []

        directions = [
            (1, 0), (-1, 0),
            (0, 1), (0, -1),
            (1, 1), (1, -1),
            (-1, 1), (-1, -1)
        ]

        for cell in frontier_cells:
            if cell in visited:
                continue

            cluster = []
            stack = [cell]
            visited.add(cell)

            while stack:
                current = stack.pop()
                cluster.append(current)

                cx, cy = current

                for dx, dy in directions:
                    neighbor = (cx + dx, cy + dy)

                    if neighbor in frontier_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        return clusters

    def estimate_information_gain(self, gx, gy, radius=20):
        unknown_count = 0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = gx + dx
                ny = gy + dy

                if not self.map.in_bounds(nx, ny):
                    continue

                distance = np.sqrt(dx**2 + dy**2)

                if distance > radius:
                    continue

                if self.is_unknown(nx, ny):
                    unknown_count += 1

        return unknown_count

    def choose_frontier_goal(self, robot_world, uncertainty=0.0, frontier_memory=None):
        frontier_cells = self.detect_frontier_cells()
        clusters = self.cluster_frontiers(frontier_cells)

        if not clusters:
            return None

        robot_gx, robot_gy = self.map.world_to_grid(
            robot_world[0],
            robot_world[1]
        )

        best_cluster = None
        best_score = float("inf")

        for cluster in clusters:
            xs = [cell[0] for cell in cluster]
            ys = [cell[1] for cell in cluster]

            center_gx = int(sum(xs) / len(xs))
            center_gy = int(sum(ys) / len(ys))

            candidate_goal = self.map.grid_to_world(center_gx, center_gy)

            if frontier_memory is not None and frontier_memory.is_failed_goal(candidate_goal):
                continue

            distance = np.sqrt(
                (center_gx - robot_gx) ** 2 +
                (center_gy - robot_gy) ** 2
            )

            information_gain = self.estimate_information_gain(
                center_gx,
                center_gy,
                radius=20
            )

            revisit_penalty = 0.0

            if self.visited_map is not None:
                revisit_penalty = self.visited_map.revisit_penalty(
                    center_gx,
                    center_gy,
                    radius=12
                )

            if self.strategy == "nearest":
                score = distance

            elif self.strategy == "info_gain":
                score = (
                    1.5 * distance
                    - 2.0 * information_gain
                    - 0.3 * len(cluster)
                )

            elif self.strategy == "info_gain_revisit":
                score = (
                    1.5 * distance
                    - 2.0 * information_gain
                    - 0.3 * len(cluster)
                    + 1.2 * revisit_penalty
                )

            elif self.strategy == "uncertainty_aware":
                uncertainty_weight = 25.0 * uncertainty

                score = (
                    (1.5 + uncertainty_weight) * distance
                    - 2.0 * information_gain
                    - 0.3 * len(cluster)
                    + 1.2 * revisit_penalty
                )

            else:
                raise ValueError(f"Unknown frontier strategy: {self.strategy}")

            if score < best_score:
                best_score = score
                best_cluster = cluster
        
        if best_cluster is None:
            return None
        xs = [cell[0] for cell in best_cluster]
        ys = [cell[1] for cell in best_cluster]

        goal_gx = int(sum(xs) / len(xs))
        goal_gy = int(sum(ys) / len(ys))

        return self.map.grid_to_world(goal_gx, goal_gy)