# astar_planner.py

import heapq
import numpy as np


class AStarPlanner:
    def __init__(self, occupancy_grid, costmap):
        self.map = occupancy_grid
        self.costmap = costmap

    def heuristic(self, a, b):
        ax, ay = a
        bx, by = b

        return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def get_neighbors(self, node):
        x, y = node

        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        neighbors = []

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if not self.map.in_bounds(nx, ny):
                continue

            if self.costmap.is_blocked(nx, ny):
                continue

            neighbors.append((nx, ny))

        return neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def plan(self, start_world, goal_world):
        start = self.map.world_to_grid(
            start_world[0],
            start_world[1]
        )

        goal = self.map.world_to_grid(
            goal_world[0],
            goal_world[1]
        )

        if not self.map.in_bounds(start[0], start[1]):
            return None

        if not self.map.in_bounds(goal[0], goal[1]):
            return None

        if self.costmap.is_blocked(start[0], start[1]):
            return None

        if self.costmap.is_blocked(goal[0], goal[1]):
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                grid_path = self.reconstruct_path(came_from, current)

                world_path = [
                    self.map.grid_to_world(gx, gy)
                    for gx, gy in grid_path
                ]

                return world_path

            if current in visited:
                continue

            visited.add(current)

            for neighbor in self.get_neighbors(current):
                nx, ny = neighbor

                movement_cost = self.heuristic(current, neighbor)
                clearance_cost = self.costmap.cost(nx, ny)

                tentative_g_score = (
                    g_score[current]
                    + movement_cost
                    + clearance_cost
                )

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score

                    f_score = (
                        tentative_g_score
                        + self.heuristic(neighbor, goal)
                    )

                    heapq.heappush(open_set, (f_score, neighbor))

        return None