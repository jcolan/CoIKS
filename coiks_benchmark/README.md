# CoIKS-Benchmark

Comprehensive benchmarking suite for comparing CoIKS against TracIK and Orocos KDL across 15 commercial robot manipulators.

## Overview

This benchmark evaluates inverse kinematics (IK) performance across different solvers using:
- 10,000 random target poses per robot
- 10,000 random initial configurations
- Full 6D pose constraints (position and orientation)
- Reachable poses generated via forward kinematics
- Standardized test conditions for fair comparison

## Evaluated Robots

The benchmark includes 15 commercial robots representing diverse kinematic configurations:

| Robot Model | Manufacturer | DOF | Type |
|------------|--------------|-----|------|
| HSR | Toyota Motor Corp. | 5 | Service Robot |
| VS050 | DENSO Robotics | 6 | Industrial |
| xArm6 | UFactory | 6 | Collaborative |
| IRB140 | ABB | 6 | Industrial |
| Jaco | Kinova | 6 | Assistive |
| UR-5 | Universal Robots | 6 | Collaborative |
| UR-10 | Universal Robots | 6 | Collaborative |
| Gen3 | Kinova Robotics | 7 | Research |
| LBR iiwa | KUKA | 7 | Collaborative |
| xArm7 | UFactory | 7 | Collaborative |
| Panda | Franka Emika | 7 | Research |
| Sawyer | Rethink Robotics | 7 | Collaborative |
| Yumi | ABB | 7 | Dual-arm |
| Fetch | Fetch Robotics | 8 | Mobile Manipulator |
| Tiago | PAL Robotics | 8 | Service Robot |

## Compared Solvers

- CoIKS (All variants)
- TracIK
- Orocos KDL
- INVJ-RR (standalone version)
