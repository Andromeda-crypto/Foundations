# Robotics

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus)
![Localization](https://img.shields.io/badge/Localization-Implemented-success?style=flat-square)
![Mapping](https://img.shields.io/badge/Mapping-Implemented-success?style=flat-square)
![Planning](https://img.shields.io/badge/Planning-Implemented-success?style=flat-square)
![SLAM](https://img.shields.io/badge/SLAM-Upcoming-orange?style=flat-square)

Robotics algorithms and autonomous navigation systems implemented from first principles.

---

## Overview

This directory contains implementations related to localization, mapping, planning, exploration, and probabilistic robotics.

The objective is to develop a complete understanding of autonomous navigation systems before progressing toward research-level topics such as FastSLAM, GraphSLAM, and modern SLAM architectures.

---

## Technology Stack

### Languages

- Python
- C++

### Libraries

- NumPy
- Matplotlib

### Planned Ecosystem

- ROS2
- Gazebo
- RViz

---

## Implemented Systems

### Localization

| Algorithm | Status |
|------------|---------|
| 1D Kalman Filter | ✓ |
| 2D Pose Estimation | ✓ |
| Extended Kalman Filter (EKF) | ✓ |
| Particle Filter Localization | ✓ |

---

### Mapping

| Component | Status |
|------------|---------|
| Occupancy Grid Mapping | ✓ |
| Log-Odds Updates | ✓ |
| Costmap Generation | ✓ |
| Obstacle Inflation | ✓ |

---

### Planning

| Algorithm | Status |
|------------|---------|
| A* Search | ✓ |
| Cost-Aware Planning | ✓ |
| Frontier Detection | ✓ |
| Frontier Clustering | ✓ |

---

### Exploration

| Feature | Status |
|------------|---------|
| Nearest Frontier Selection | ✓ |
| Information Gain Exploration | ✓ |
| Revisit Penalties | ✓ |
| Frontier Memory | ✓ |
| Uncertainty-Aware Exploration | ✓ |
| Recovery Behaviors | ✓ |

---

## Frontier Exploration Framework

The current flagship project integrates:

```text
Particle Filter Localization
            +
Occupancy Grid Mapping
            +
Costmap Generation
            +
A* Path Planning
            +
Frontier-Based Exploration
```

to produce a complete autonomous exploration pipeline.

### Implemented Features

- Frontier Clustering
- Information Gain Scoring
- Revisit Penalties
- Frontier Memory
- Reachability Filtering
- Costmap-Based Navigation
- Recovery Behaviors
- Exploration Metrics Logging
- Strategy Comparison Framework

---

## Experimental Evaluation

The framework includes tooling for measuring:

- Mapping Coverage
- Distance Traveled
- Localization Uncertainty
- Effective Sample Size (ESS)
- Frontier Statistics
- Exploration Strategy Performance

### Strategies Evaluated

1. Nearest Frontier
2. Information Gain
3. Information Gain + Revisit Penalty
4. Uncertainty-Aware Exploration

---

## Roadmap

### Completed

- Localization
- Mapping
- Planning
- Frontier Exploration

### Next

- FastSLAM 1.0
- FastSLAM 2.0
- GraphSLAM

### Long-Term

- Research Paper Replication
- Autonomous Navigation Systems
- Multi-Robot Exploration
- Advanced Motion Planning

---

## References

- Probabilistic Robotics — Thrun, Burgard, Fox
- Planning Algorithms — Steven M. LaValle
- Modern Robotics — Lynch & Park
- FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem (Montemerlo et al.)

---

## Author

**Om Anand**  
Computer Science & Mathematics  
Pennsylvania State University