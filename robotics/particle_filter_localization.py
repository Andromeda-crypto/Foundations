# Particle Filter Localization

import numpy as np
import random
import matplotlib.pyplot as plt

from ekf_localization import Robot

class Particle:
    def __init__(self,x,y,theta,weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        
    
class ParticleFilter:
    def __init__(self,particles):
        self.particles = particles
    
    def predict(self,v,omega,dt):
        # takes control filters 
        for p in self.particles:
            p.theta += omega * dt + random.gauss(0,0.02)
            p.x += v * dt * np.cos(p.theta) + random.gauss(0,0.05)
            p.y += v * dt * np.sin(p.theta) + random.gauss(0,0.05)

            # keeping p.theta bounded between -pi and pi
            p.theta = (p.theta + np.pi) % (2 * np.pi) - np.pi

    def calculate_likelihood(
        self,
        particle,
        z_t,
        landmarks,
        range_noise=0.5,
        bearing_noise=0.05
    ):
        total_likelihood = 1.0

        for i, landmark_pos in enumerate(landmarks):
            measured_range, measured_bearing = z_t[i]

            dx = landmark_pos[0] - particle.x
            dy = landmark_pos[1] - particle.y

            expected_range = np.sqrt(dx**2 + dy**2)

            expected_bearing = np.arctan2(dy, dx) - particle.theta
            expected_bearing = (expected_bearing + np.pi) % (2 * np.pi) - np.pi

            range_error = measured_range - expected_range

            bearing_error = measured_bearing - expected_bearing
            bearing_error = (bearing_error + np.pi) % (2 * np.pi) - np.pi

            range_likelihood = np.exp(
                -0.5 * (range_error / range_noise) ** 2
            )

            bearing_likelihood = np.exp(
                -0.5 * (bearing_error / bearing_noise) ** 2
            )

            total_likelihood *= range_likelihood * bearing_likelihood

        return total_likelihood
        

    def update(self, z_t, map_data):
        eta = 0.0

        for p in self.particles:
            p.weight = self.calculate_likelihood(p, z_t, map_data)
            eta += p.weight

        if eta > 0:
            for p in self.particles:
                p.weight /= eta
        else:
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)


    def resample(self):
        M = len(self.particles)
        weights = [p.weight for p in self.particles]
        cumulative_sum = np.cumsum(weights)

        new_particles = []
        u = random.uniform(0, 1.0/M)
        i = 0

        for j in range(M):
            while u > cumulative_sum[i]:
                i += 1
            parent = self.particles[i]
            new_particles.append(Particle(parent.x, parent.y, parent.theta, 1.0/M))

            u += 1.0 / M

        self.particles = new_particles

    def estimate(self):
        mean_x = 0
        mean_y = 0

        # For angles, we need to  average vector components to prevent 179 deg and -179 deg averaging to 0
        sum_sin = 0
        sum_cos = 0

        for p in self.particles:
            mean_x += p.x * p.weight
            mean_y += p.y * p.weight

            sum_sin += np.sin(p.theta) * p.weight
            sum_cos += np.cos(p.theta) * p.weight
        
        mean_theta = np.arctan2(sum_sin, sum_cos)

        return [mean_x,mean_y,mean_theta]
    def effective_sample_size(self):
        weights = [p.weight for p in self.particles]

        ess = 1.0 / np.sum(np.square(weights))

        return ess


# testing
    
landmarks = [(5, 5), (10, 0), (15, 8)]

num_particles = 1000
particles = []

for _ in range(num_particles):
    particles.append(
        Particle(
            x=random.uniform(-10, 25),
            y=random.uniform(-10, 25),
            theta=random.uniform(-np.pi, np.pi),
            weight=1.0 / num_particles
        )
    )

pf = ParticleFilter(particles)

robot = Robot(0, 0, 0)

v = 1.0
omega = 0.05
dt = 1.0
num_steps = 200

true_x = []
true_y = []

est_x = []
est_y = []

position_errors = []
theta_errors = []
ess_values = []


plt.figure(figsize=(8, 8))

for step in range(num_steps):
    robot.move_robot(v, omega, dt)

    measurements = []

    for lx, ly in landmarks:
        dx = lx - robot.x
        dy = ly - robot.y

        true_range = np.sqrt(dx**2 + dy**2)

        true_bearing = np.arctan2(dy, dx) - robot.theta
        true_bearing = (true_bearing + np.pi) % (2 * np.pi) - np.pi

        noisy_range = true_range + random.gauss(0, 0.5)
        noisy_bearing = true_bearing + random.gauss(0, 0.05)

        measurements.append((noisy_range, noisy_bearing))

    pf.predict(v, omega, dt)
    pf.update(measurements, landmarks)
    ess = pf.effective_sample_size()
    ess_values.append(ess)

    # conditional resampling
    if ess < num_particles / 2:
        pf.resample()
    estimated_pose = pf.estimate()

    position_error = np.sqrt(
        (robot.x - estimated_pose[0])**2 +
        (robot.y - estimated_pose[1])**2
    )

    theta_error = robot.theta - estimated_pose[2]
    theta_error = np.arctan2(
        np.sin(theta_error),
        np.cos(theta_error)
    )
    theta_error = abs(theta_error)

    position_errors.append(position_error)
    theta_errors.append(theta_error)

    true_x.append(robot.x)
    true_y.append(robot.y)

    est_x.append(estimated_pose[0])
    est_y.append(estimated_pose[1])

    plt.clf()

    # true and estimated paths
    plt.plot(true_x, true_y, label="True Path")
    plt.plot(est_x, est_y, label="PF Estimate")

    # current true robot and current estimate
    plt.scatter(robot.x, robot.y, s=80, label="True Robot")
    plt.scatter(
        estimated_pose[0],
        estimated_pose[1],
        s=80,
        label="Estimated Pose"
    )

    # particle cloud
    particle_x = [p.x for p in pf.particles]
    particle_y = [p.y for p in pf.particles]

    plt.scatter(
        particle_x,
        particle_y,
        s=5,
        alpha=0.25,
        label="Particles"
    )

    # landmarks
    landmark_x = [p[0] for p in landmarks]
    landmark_y = [p[1] for p in landmarks]

    plt.scatter(
        landmark_x,
        landmark_y,
        marker="*",
        s=150,
        label="Landmarks"
    )

    # dashed sensing lines from estimated pose to landmarks
    for lx, ly in landmarks:
        plt.plot(
            [estimated_pose[0], lx],
            [estimated_pose[1], ly],
            linestyle="dashed",
            alpha=0.4
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Live Particle Filter Localization")
    plt.grid(True)
    plt.axis("equal")
    plt.legend(loc="upper right")

    plt.pause(0.05)

    print(f"Step {step + 1}")
    print(f"True Pose: x={robot.x:.2f}, y={robot.y:.2f}, theta={robot.theta:.2f}")
    print(f"PF Estimate: x={estimated_pose[0]:.2f}, y={estimated_pose[1]:.2f}, theta={estimated_pose[2]:.2f}")
    print(f"Position Error: {position_error:.3f}")
    print(f"Theta Error: {theta_error:.3f}")
    print(f"ESS: {ess:.2f}")
    if ess < num_particles / 2:
        print("Resampling Triggered")
    print()

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(position_errors, label="Position Error")
plt.plot(theta_errors, label="Theta Error")

plt.xlabel("Time Step")
plt.ylabel("Error")
plt.title("Particle Filter Localization Error")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(ess_values,label= 'ESS')

plt.axhline(
    y = num_particles / 2,
    linestyle= 'dashed',
    label= 'Resample Threshold'
)

plt.xlabel('Time Step')
plt.ylabel('Effective Sample Size')
plt.title('Particle Filter ESS')

plt.grid(True)
plt.legend()
plt.show()