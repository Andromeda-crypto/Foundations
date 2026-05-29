import numpy as np


class CostMap:
    def __init__(
        self,
        occupancy_grid,
        occupancy_threshold=0.65,
        inflation_radius=6,
        cost_radius=8,
        cost_weight=4.0
    ):
        self.occupancy_grid = occupancy_grid
        self.occupancy_threshold = occupancy_threshold
        self.inflation_radius = inflation_radius
        self.cost_radius = cost_radius
        self.cost_weight = cost_weight

        self.blocked = None
        self.costs = None

    def probability(self, gx, gy):
        log_odds = self.occupancy_grid.grid[gy, gx]
        return 1 - (1 / (1 + np.exp(log_odds)))

    def build(self):
        grid_height = self.occupancy_grid.grid_height
        grid_width = self.occupancy_grid.grid_width

        self.blocked = np.zeros((grid_height, grid_width), dtype=bool)
        self.costs = np.zeros((grid_height, grid_width), dtype=float)

        occupied_cells = []

        for gy in range(grid_height):
            for gx in range(grid_width):
                prob = self.probability(gx, gy)

                if prob >= self.occupancy_threshold:
                    occupied_cells.append((gx, gy))

        for ox, oy in occupied_cells:
            for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                    nx = ox + dx
                    ny = oy + dy

                    if not self.occupancy_grid.in_bounds(nx, ny):
                        continue

                    distance = np.sqrt(dx**2 + dy**2)

                    if distance <= self.inflation_radius:
                        self.blocked[ny, nx] = True

            for dx in range(-self.cost_radius, self.cost_radius + 1):
                for dy in range(-self.cost_radius, self.cost_radius + 1):
                    nx = ox + dx
                    ny = oy + dy

                    if not self.occupancy_grid.in_bounds(nx, ny):
                        continue

                    distance = np.sqrt(dx**2 + dy**2)

                    if distance <= self.cost_radius:
                        self.costs[ny, nx] += self.cost_weight / (distance + 1.0)

    def is_blocked(self, gx, gy):
        if self.blocked is None:
            return False

        return self.blocked[gy, gx]

    def cost(self, gx, gy):
        if self.costs is None:
            return 0.0

        return self.costs[gy, gx]