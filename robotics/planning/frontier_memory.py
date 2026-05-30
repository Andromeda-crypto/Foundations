# frontier_memory.py

import numpy as np

class FrontierMemory:
    def __init__(self,reject_radius=3):
        self.failed_goals = []
        self.reject_radius = reject_radius

    def add_failed_goal(self,goal_world):
        if goal_world is not None:
            self.failed_goals.append(goal_world)

    def is_failed_goal(self, goal_world):
        if goal_world is None:
            return False
        
        gx,gy = goal_world

        for fx,fy in self.failed_goals:
            distance = np.sqrt((gx - fx)**2 + (gy - fy)**2)
            if distance < self.reject_radius:
                return True
        
        return False
    
    