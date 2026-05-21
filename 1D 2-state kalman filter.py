# Implementation of a 1D 2 state kalman filter for tracking velocity and positon, while velocity itself evolves

import numpy as np
import random
import matplotlib.pyplot as plt

class KalmanFilter1D:
    def __init__(self):
        #state vector
        self.x = np.array([[0.0],
                           [0.0]])
        
        # covariance matrix 
        self.P = np.eye(2)

        # time step 
        self.dt = 1.0

        # state transition matrix 
        self.F = np.array([[1.0, self.dt],
                           [0.0,1.0]])
        
        # measurement matrix 
        self.H = np.array([[1.0, 0.0]]) # only measures position

        # process noise
        self.Q = np.array([[0.1,0.0],
                           [0.0, 0.1]])
        
        # measurement noise
        self.R = np.array([[9.0]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        print("Predicted State:")
        print(self.x)

        print("Predicted Covariance:")
        print(self.P)

    def update(self, measurement):
        z = np.array([[measurement]])
        y = z - (self.H @ self.x)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K@y

        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        return self.x , K
    
# example usage

kf = KalmanFilter1D()

true_position = 0.0
true_velocity = random.gauss(0,0.05)

position_noise_std = 2.0

num_steps = 50

true_positions = []
measurements = []
estimated_positions = []
true_velocities = []
estimated_velocities = []

for step in range(num_steps):
    # true hidden motion
    true_position += true_velocity
    true_positions.append(true_position)
    true_velocities.append(true_velocity)
    # noisy measurement
    measurement = true_position + random.gauss(0, position_noise_std)
    measurements.append(measurement)

    # kalman filter prediction and update
    kf.predict()
    estimate, gain = kf.update(measurement)

    estimated_positions.append(estimate[0,0])
    estimated_velocities.append(estimate[1,0])

    # print results
    print(f"Step {step+1}: ")
    print(f"True Position: {true_position:.2f}")
    print(f"True Velocity: {true_velocity:.2f}")
    print(f"Measurement: {measurement:.2f}")
    print(f"Estimated Position: {estimate[0,0]:.2f}")
    print(f"Estimated Velocity: {estimate[1,0]:.2f}")
    print()

# Plotting results on 3D axes 

from mpl_toolkits.mplot3d import Axes3D

plt.figure(figsize=(12,6))

plt.plot(true_velocities, label="True Velocity")
plt.plot(estimated_velocities, label="Estimated Velocity")

plt.title("True Velocity vs Estimated Velocity")
plt.xlabel("Time Step")
plt.ylabel("Velocity")

plt.legend()
plt.grid(True)

plt.show()


time_steps = list(range(num_steps))
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection = '3d')

ax.plot(
    time_steps, true_positions, zs= 0, zdir = 'z', label = 'True Position', color = 'blue'
)

ax.plot(
    time_steps, measurements, zs = 0, zdir='z', label = 'Measurements', color = 'red',linestyle = 'dashed'
)

ax.plot(
    time_steps, estimated_positions, estimated_velocities, label = 'Kalman Estimates', color = 'magenta', marker = 'o'
)

ax.set_xlabel('Time Step')
ax.set_ylabel('Position')
ax.set_zlabel('Velocity')

ax.set_title("3D Kalman Filter Tracking Visualization")
ax.legend()
plt.show()






