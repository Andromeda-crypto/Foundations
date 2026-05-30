# visited_map.py

import numpy as np


class VisitedMap:
    def __init__(self, occupancy_grid, visit_radius=6):
        self.map = occupancy_grid
        self.visit_radius = visit_radius

        self.visits = np.zeros(
            (self.map.grid_height, self.map.grid_width),
            dtype=float
        )

    def update(self, robot_x, robot_y):
        robot_gx, robot_gy = self.map.world_to_grid(robot_x, robot_y)

        for dx in range(-self.visit_radius, self.visit_radius + 1):
            for dy in range(-self.visit_radius, self.visit_radius + 1):
                nx = robot_gx + dx
                ny = robot_gy + dy

                if not self.map.in_bounds(nx, ny):
                    continue

                distance = np.sqrt(dx**2 + dy**2)

                if distance > self.visit_radius:
                    continue

                self.visits[ny, nx] += 1.0 / (distance + 1.0)

    def revisit_penalty(self, gx, gy, radius=12):
        total = 0.0
        count = 0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = gx + dx
                ny = gy + dy

                if not self.map.in_bounds(nx, ny):
                    continue

                distance = np.sqrt(dx**2 + dy**2)

                if distance > radius:
                    continue

                total += self.visits[ny, nx]
                count += 1

        if count == 0:
            return 0.0

        return total / count