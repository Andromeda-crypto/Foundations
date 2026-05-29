# uncertainty.py

import numpy as np

class ParticleUncertainty:
    def __init__(self,particle_filter):
        self.pf = particle_filter

    def covariance_matrix(self):
        particles = self.pf.particles

        xs = np.array([p.x for p in particles])
        ys = np.array([p.y for p in particles])

        weights = np.array([p.weight for p in particles])
        weights /= np.sum(weights)

        mean_x = np.sum(xs * weights)
        mean_y = np.sum(ys * weights)

        var_x = np.sum(weights * (xs - mean_x)**2)
        var_y = np.sum(weights * (ys - mean_y)**2)
        
        cov_xy = np.sum(weights * (xs - mean_x) * (ys - mean_y))

        covariance = np.array([
            [var_x,cov_xy],
            [cov_xy,var_y]
        ])
        
        return covariance
    
    def position_uncertainty(self):
        covariance = self.covariance_matrix()
        
        return np.trace(covariance)