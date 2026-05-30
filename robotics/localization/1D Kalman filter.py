# implementing 1D kalman filter 
class KalmanFilter1D:
    def __init__(
        self,
        process_variance,
        measurement_variance,
        initial_estimate=0.0,
        initial_error_estimate=1.0
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error_estimate = initial_error_estimate

    def update(self, measurement, velocity):
        # prediction step
        self.estimate += velocity
        self.error_estimate += self.process_variance

        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_variance)

        self.estimate += kalman_gain * (measurement - self.estimate)
        
        self.error_estimate *= (1 - kalman_gain)

        return self.estimate, kalman_gain


        
# Analyzing and plotting the results

import random
import matplotlib.pyplot as plt

num_steps = 100
true_position = 0.0
velocity = 1.0

process_std = 0.3
sensor_std = 2.0

kf = KalmanFilter1D(
    process_variance= process_std**2,
    measurement_variance = sensor_std**2,
    initial_estimate= 0.0,
    initial_error_estimate= 1.0
)

true_positions = []
measurements = []
estimates = []
kalman_gains = []

for step in range(num_steps):
    # hidden true position
    true_position += velocity + random.gauss(0, process_std)
    true_positions.append(true_position)

    # noisy measurement
    measurement = true_position + random.gauss(0, sensor_std)
    measurements.append(measurement)

    # update kalman filter
    estimate, kalman_gain = kf.update(measurement, velocity)

    estimates.append(estimate)
    kalman_gains.append(kalman_gain)


# Plotting the results 

plt.figure(figsize=(12,6))
plt.plot(true_positions, label= 'True Position', color = 'blue')
plt.plot(measurements, label = 'Measurements', alpha =0.6)
plt.plot(estimates, label= 'Kalman Estimate', color = 'red')
plt.title('1D kalman Filter tracing a moving object')
plt.xlabel('Time Step')
plt.ylabel('position')
plt.legend()
plt.grid(True)
plt.show()



