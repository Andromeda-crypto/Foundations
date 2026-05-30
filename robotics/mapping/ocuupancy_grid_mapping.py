# Occupancy Grid Mapping

import numpy as np
import matplotlib.pyplot as plt


class OccupancyGridMap:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution

        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.grid = np.zeros(
            (self.grid_height, self.grid_width)
        )

    def world_to_grid(self,x,y):
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)

        return gx, gy

    def grid_to_world(self,gx,gy):
        x = gx * self.resolution
        y = gy * self.resolution

        return x,y

    def in_bounds(self, gx, gy):
        return (
            0 <= gx < self.grid_width
            and 0 <= gy < self.grid_height
        )
        
    def mark_occupied(self, x, y):
        gx, gy = self.world_to_grid(x, y)

        if self.in_bounds(gx, gy):
            self.grid[gy, gx] += 0.85
    
    def mark_free(self, x, y):
        gx, gy = self.world_to_grid(x, y)

        if self.in_bounds(gx, gy):
            self.grid[gy, gx] -= 0.4

    def clamp_grid(self):
        self.grid = np.clip(self.grid, -4.0, 4.0)

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        error = dx - dy

        while True:
            cells.append((x0,y0))

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x0 += sx

            if e2 < dx:
                error += dx
                y0 += sy
        
        return cells
    
    def update_ray(self, robot_x, robot_y, hit_x, hit_y):
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
        hit_gx, hit_gy = self.world_to_grid(hit_x, hit_y)

        cells = self.bresenham(robot_gx, robot_gy, hit_gx, hit_gy)

        for gx, gy in cells[:-1]:
            if self.in_bounds(gx, gy):
                self.grid[gy, gx] -= 0.4

        if self.in_bounds(hit_gx, hit_gy):
            self.grid[hit_gy, hit_gx] += 0.85

        self.clamp_grid()

    def update_free_ray(self,robot_x, robot_y, end_x, end_y):
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
        end_gx, end_gy = self.world_to_grid(end_x, end_y)

        cells = self.bresenham(robot_gx, robot_gy, end_gx, end_gy)

        for gx, gy in cells:
            if self.in_bounds(gx, gy):
                self.grid[gy, gx] -= 0.4

        self.clamp_grid()

    

    def plot(self):

        probabilities = 1 - (
            1 / (1 + np.exp(self.grid))
        )

        plt.imshow(
            probabilities,
            cmap="gray_r",
            origin="lower",
            extent=[0, self.width, 0, self.height],
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Occupancy Grid Map")
        plt.colorbar(label="Occupancy Probability")

        plt.show()

# test
if __name__ == "__main__":

    grid_map = OccupancyGridMap(
        width=20,
        height=20,
        resolution=0.5
    )

    obstacles = [
        (15, 15),
        (5, 16),
        (16, 5),
        (4, 7),
        (12, 18)
    ]

    robot_x = 2
    robot_y = 2

    plt.figure(figsize=(8, 8))

    for step in range(75):

        robot_x += 0.4
        robot_y += 0.25

        # collision testing
        for ox, oy in obstacles:

            distance_to_obstacle = np.sqrt(
                (robot_x - ox)**2 +
                (robot_y - oy)**2
            )

            if distance_to_obstacle < 0.5:
                print(
                    f"Collision detected at "
                    f"({ox:.2f}, {oy:.2f})"
                )

        # update occupancy map
        for hit_x, hit_y in obstacles:

            grid_map.update_ray(
                robot_x,
                robot_y,
                hit_x,
                hit_y
            )

        plt.clf()

        probabilities = 1 - (
            1 / (1 + np.exp(grid_map.grid))
        )

        plt.imshow(
            probabilities,
            cmap="gray_r",
            origin="lower",
            extent=[0, grid_map.width, 0, grid_map.height],
            vmin=0,
            vmax=1
        )

        # robot
        plt.scatter(
            robot_x,
            robot_y,
            color="red",
            s=100,
            label="Robot"
        )

        # obstacles
        obstacle_x = [o[0] for o in obstacles]
        obstacle_y = [o[1] for o in obstacles]

        plt.scatter(
            obstacle_x,
            obstacle_y,
            color="blue",
            s=80,
            label="Obstacles"
        )

        # sensing rays
        for hit_x, hit_y in obstacles:
            plt.plot(
                [robot_x, hit_x],
                [robot_y, hit_y],
                linestyle="dashed",
                alpha=0.4
            )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Live Occupancy Grid Mapping")
        plt.legend()

        plt.pause(0.1)

    plt.show()