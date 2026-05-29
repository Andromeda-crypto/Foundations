# 2D EKF Robot Localization
# State: [x, y, theta]
# Sensor: range + bearing to known landmarks

import random
import numpy as np
import matplotlib.pyplot as plt


class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def move_robot(self, v, omega, dt):
        v_noisy = v + random.gauss(0, 0.1)
        omega_noisy = omega + random.gauss(0, 0.02)

        self.x += v_noisy * np.cos(self.theta) * dt
        self.y += v_noisy * np.sin(self.theta) * dt
        self.theta += omega_noisy * dt


class EKFLocalizer:
    def __init__(self):
        self.mu = np.array([[0.0], [0.0], [0.0]])
        self.Sigma = np.eye(3) * 0.5

    def predict(self, v, omega, dt):
        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]

        self.mu[0, 0] = x + v * np.cos(theta) * dt
        self.mu[1, 0] = y + v * np.sin(theta) * dt
        self.mu[2, 0] = theta + omega * dt

        F = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0, 1]
        ])

        Q = np.array([
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.01]
        ])

        self.Sigma = F @ self.Sigma @ F.T + Q

    def update(self, measurement, landmark):
        lx, ly = landmark

        mu_x = self.mu[0, 0]
        mu_y = self.mu[1, 0]
        mu_theta = self.mu[2, 0]

        dx = lx - mu_x
        dy = ly - mu_y

        q = dx**2 + dy**2
        predicted_range = np.sqrt(q)

        if predicted_range < 1e-6:
            return

        predicted_bearing = np.arctan2(dy, dx) - mu_theta

        predicted_measurement = np.array([
            [predicted_range],
            [predicted_bearing]
        ])

        y = measurement - predicted_measurement

        # normalize bearing innovation
        y[1, 0] = np.arctan2(np.sin(y[1, 0]), np.cos(y[1, 0]))

        H = np.array([
            [-dx / predicted_range, -dy / predicted_range, 0],
            [dy / q, -dx / q, -1]
        ])

        R = np.array([
            [0.25, 0],
            [0, 0.05]
        ])

        S = H @ self.Sigma @ H.T + R
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ y

        I = np.eye(3)
        self.Sigma = (I - K @ H) @ self.Sigma


if __name__== "__main__":
    # Simulation setup

    robot = Robot(0, 0, 0)
    ekf = EKFLocalizer()

    landmarks = [
        (5, 5),
        (10, 0),
        (15, 0)
    ]

    v = 2.0
    omega = 0.1
    dt = 1.0
    num_steps = 100

    x_positions = []
    y_positions = []

    ekf_x_positions = []
    ekf_y_positions = []

    position_errors = []
    theta_errors = []


    # Simulation loop

    for step in range(num_steps):
        robot.move_robot(v, omega, dt)
        ekf.predict(v, omega, dt)

        x_positions.append(robot.x)
        y_positions.append(robot.y)

        for lx, ly in landmarks:
            dx_true = lx - robot.x
            dy_true = ly - robot.y

            true_range = np.sqrt(dx_true**2 + dy_true**2)
            true_bearing = np.arctan2(dy_true, dx_true) - robot.theta

            noisy_range = true_range + random.gauss(0, 0.5)
            noisy_bearing = true_bearing + random.gauss(0, 0.05)

            measurement = np.array([
                [noisy_range],
                [noisy_bearing]
            ])

            ekf.update(measurement, (lx, ly))

        ekf_x_positions.append(ekf.mu[0, 0])
        ekf_y_positions.append(ekf.mu[1, 0])

        position_error = np.sqrt(
            (robot.x - ekf.mu[0, 0])**2 +
            (robot.y - ekf.mu[1, 0])**2
        )

        theta_error = abs(robot.theta - ekf.mu[2, 0])
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
        theta_error = abs(theta_error)

        position_errors.append(position_error)
        theta_errors.append(theta_error)

        print(f"Step {step + 1}")
        print(f"True Pose: x={robot.x:.3f}, y={robot.y:.3f}, theta={robot.theta:.3f}")
        print(f"EKF Pose:  x={ekf.mu[0,0]:.3f}, y={ekf.mu[1,0]:.3f}, theta={ekf.mu[2,0]:.3f}")
        print(f"Position Error: {position_error:.3f}")
        print(f"Theta Error: {theta_error:.3f}")
        print()


    # Plot 1: trajectory

    plt.figure(figsize=(8, 8))

    plt.plot(
        x_positions,
        y_positions,
        marker="o",
        label="True Robot Path"
    )

    plt.plot(
        ekf_x_positions,
        ekf_y_positions,
        marker="x",
        label="EKF Estimated Path"
    )

    landmark_x = [p[0] for p in landmarks]
    landmark_y = [p[1] for p in landmarks]

    plt.scatter(
        landmark_x,
        landmark_y,
        marker="x",
        s=120,
        label="Landmarks"
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("True Robot Path vs EKF Estimated Path")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()


    # Plot 2: localization errors

    plt.figure(figsize=(10, 5))

    plt.plot(position_errors, label="Position Error")
    plt.plot(theta_errors, label="Theta Error")

    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.title("EKF Localization Error Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()