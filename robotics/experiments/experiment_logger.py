# experiment logger 

import os
import csv 

class ExperimentLogger:
    def __init__(self, filename='info_gain_revisit.csv'):
        self.filename = filename

        self.headers = ["step", "mapped_percent", "free_percent", "occupied_percent", "unknown_percent", "distance_traveled", "frontier_cells", "frontier_clusters", "ess", "uncertainty"]

        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)   
                writer.writerow(self.headers)
        print(f"Logging to: {os.path.abspath(self.filename)}")

    def log(self,step, metrics, ess, uncertainty):
            row = [
                step,
                metrics['mapped_percent'],
                metrics['free_percent'],
                metrics['occupied_percent'],
                metrics['unknown_percent'],
                metrics['distance_traveled'],
                metrics['frontier_cells'],
                metrics['frontier_clusters'],
                ess,
                uncertainty
            ]

            with open(self.filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
