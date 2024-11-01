# CoIKS (Concurrent Inverse Kinematics Solver)

## Overview

CoIKS (Concurrent Inverse Kinematics Solver) is a comprehensive framework designed to solve inverse kinematics (IK) problems efficiently by leveraging concurrent computations with multiple solvers. This repository includes the core CoIKS implementation, benchmarking tools, and a library for TRAC-IK integration, providing a complete setup for performance comparisons and use with various robot models.

## Directory Structure

This repository contains the following main directories:

- **coiks/**: The core implementation of the CoIKS solver, focused on solving inverse kinematics problems through concurrent computations. This directory includes a README specific to CoIKS, covering detailed features, dependancies, usage instructions, and technical information about the solver’s architecture and configuration.
  
- **coiks_benchmark/**: Contains benchmarking scripts and configurations for comparing CoIKS with other IK solvers, specifically TRAC-IK and Orocos KDL. This benchmarking suite is set up with 15 different robot models, providing performance metrics and accuracy comparisons across solvers. This directory is useful for users who want to evaluate CoIKS’s efficiency and performance against established solutions.
  
- **tracik_lib/**: This is a submodule containing the TRAC-IK library, allowing seamless integration with CoIKS for comparative purposes. Users need to initialize this submodule to access TRAC-IK functionality within the benchmark tests.

> **Note**: A preliminary version of this concurrent solver, developed for surgical manipulators under Remote Center of Motion (RCM) constraints, was introduced in the paper *"A Concurrent Framework for Constrained Inverse Kinematics of Minimally Invasive Surgical Robots."* The code for this previous version is available in the [CoIKS-Surg repository](https://github.com/jcolan/CoIKS-Surg). 
