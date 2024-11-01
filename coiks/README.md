# CoIKS: Concurrent Inverse Kinematics Solver

## Overview

CoIKS (Concurrent Inverse Kinematics Solver) is a framework designed for efficient and robust inverse kinematics (IK) solving. By leveraging parallel processing and various IK solving strategies, CoIKS provides multiple solutions for robotic end-effector positioning and selects the most optimal solution based on predefined criteria. This modular, multi-threaded framework is optimized for modern multi-core CPUs, making it adaptable to different robotic applications.

## Key Features

- **Parallel IK Solvers**: Runs multiple IK solvers concurrently to increase the chance of finding a feasible and optimal solution.
- **Standalone Solvers**: Modular design supports different types of IK solvers:
  - **INVJ-RR**: Jacobian-based solver with random restarts to handle joint limits and local minima.
  - **QP Solver**: Quadratic Programming-based solver, designed to handle joint limits as inequality constraints.
  - **NLO Solver**: Nonlinear optimization-based solver, directly optimizing for the target pose.
- **Concurrent Variants**: Combines different solvers (or multiple instances of the same solver) to improve speed and reliability:
  - **CoIKS-QP**: Combines INVJ-RR with QP for enhanced robustness.
  - **CoIKS-NLO**: Combines INVJ-RR with NLO for optimized constraints handling.
  - **CoIKS-nINVJ**: Employs multiple INVJ-RR instances with different random seeds.
- **Solving Modes**: Three modes for selecting the optimal solution:
  - **Speed**: Prioritizes the fastest valid solution.
  - **Distance**: Minimizes the change in joint configuration from the initial position.
  - **Manipulability**: Maximizes the robot's freedom of movement at the solution pose.

## Getting Started

### Prerequisites

- **OS**: Linux Ubuntu 20.04
- **Compiler**: C++14 compatible compiler
- **ROS**: CoIKS integrates with ROS but can operate independently.
- **Dependencies**:
  - [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for kinematic computations.
  - [CASadi](https://web.casadi.org/) as a backend for nonlinear and HQP solvers.
  - [IPOPT](https://coin-or.github.io/Ipopt/) with the HSL/MA57 linear solver for nonlinear optimization.
  - [OSQP](https://osqp.org/) for solving QP problems, with warm start enabled.



  
## License
This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.


