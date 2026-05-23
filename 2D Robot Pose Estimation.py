#2D Robot Pose Estimation

import random
import matplotlib.pyplot as plt
import numpy as np

class Robot:
    def __init__(self,x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def move_Robot(self,v,omega,dt):
        # motion noise

        v_noisy = v + random.gauss(0,0.1)
        omega_noisy = omega + random.gauss(0,0.02)

        self.x += v_noisy + np.cos(self.theta) * dt
        self.y += v_noisy + np.sin(self.theta) * dt
        self.theta += omega_noisy * dt


    
# testing 

robot = Robot(0,0,0)

# 1. storing trajectories

x_positions = []
y_positions = []
landmarks = [
    (5,5), (10,0), (15,0)
]

for _ in range(15):

    robot.move_Robot(v=2.0, omega=0.1, dt=1.0)

    x_positions.append(robot.x)
    y_positions.append(robot.y)

    print(f"{robot.x:.4f}, {robot.y:.4f}, {robot.theta:.4f}")

    # landmark sensing
    for lx, ly in landmarks:

        distance = np.sqrt(
            (robot.x - lx)**2 +
            (robot.y - ly)**2
        )

        noisy_distance = distance + random.gauss(0, 0.5)

        print(
            f"Landmark ({lx}, {ly}) -> "
            f"Noisy Distance: {noisy_distance:.2f}"
        )
        plt.clf()

        # Plot robot trajectory
        plt.plot(x_positions, y_positions, marker='o', label='Robot Path')

        # Plot current robot position
        plt.scatter(robot.x, robot.y, color='red', s=100, label='Robot')

        # Plot landmarks
        landmark_x = [p[0] for p in landmarks]
        landmark_y = [p[1] for p in landmarks]

        plt.scatter(
            landmark_x,
            landmark_y,
            marker='x',
            s=100,
            label='Landmarks'
        )

        # Draw sensing lines
        for lx, ly in landmarks:

            plt.plot(
                [robot.x, lx],
                [robot.y, ly],
                linestyle='dashed'
            )

        plt.xlabel("X")
        plt.ylabel("Y")

        plt.title("Robot Landmark Sensing")

        plt.axis('equal')

        plt.legend()

        plt.pause(0.005)

        print()






        
    
