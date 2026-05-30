# exploration_metrics.py

import numpy as np


class ExplorationMetrics:
    def __init__(
        self,
        occupancy_grid,
        free_threshold=0.35,
        occupied_threshold=0.65
    ):
        self.map = occupancy_grid
        self.free_threshold = free_threshold
        self.occupied_threshold = occupied_threshold

        self.mapped_percent_history = []
        self.free_percent_history = []
        self.occupied_percent_history = []
        self.unknown_percent_history = []
        self.distance_traveled_history = []
        self.frontier_cell_count_history = []
        self.frontier_cluster_count_history = []

        self.total_distance_traveled = 0.0
        self.previous_robot_position = None

    def probability_grid(self):
        return 1 - (1 / (1 + np.exp(self.map.grid)))

    def update_distance(self, robot_x, robot_y):
        if self.previous_robot_position is None:
            self.previous_robot_position = (robot_x, robot_y)
            return

        prev_x, prev_y = self.previous_robot_position

        distance = np.sqrt(
            (robot_x - prev_x) ** 2 +
            (robot_y - prev_y) ** 2
        )

        self.total_distance_traveled += distance
        self.previous_robot_position = (robot_x, robot_y)

    def update(self, robot_x, robot_y, frontier_cells=None, frontier_clusters=None):
        probabilities = self.probability_grid()

        free_cells = probabilities <= self.free_threshold
        occupied_cells = probabilities >= self.occupied_threshold
        unknown_cells = (
            (probabilities > self.free_threshold)
            & (probabilities < self.occupied_threshold)
        )

        total_cells = probabilities.size

        free_percent = 100 * np.sum(free_cells) / total_cells
        occupied_percent = 100 * np.sum(occupied_cells) / total_cells
        unknown_percent = 100 * np.sum(unknown_cells) / total_cells
        mapped_percent = free_percent + occupied_percent

        self.update_distance(robot_x, robot_y)

        frontier_cell_count = 0
        frontier_cluster_count = 0

        if frontier_cells is not None:
            frontier_cell_count = len(frontier_cells)

        if frontier_clusters is not None:
            frontier_cluster_count = len(frontier_clusters)

        self.free_percent_history.append(free_percent)
        self.occupied_percent_history.append(occupied_percent)
        self.unknown_percent_history.append(unknown_percent)
        self.mapped_percent_history.append(mapped_percent)
        self.distance_traveled_history.append(self.total_distance_traveled)
        self.frontier_cell_count_history.append(frontier_cell_count)
        self.frontier_cluster_count_history.append(frontier_cluster_count)

        return {
            "mapped_percent": mapped_percent,
            "free_percent": free_percent,
            "occupied_percent": occupied_percent,
            "unknown_percent": unknown_percent,
            "distance_traveled": self.total_distance_traveled,
            "frontier_cells": frontier_cell_count,
            "frontier_clusters": frontier_cluster_count,
        }