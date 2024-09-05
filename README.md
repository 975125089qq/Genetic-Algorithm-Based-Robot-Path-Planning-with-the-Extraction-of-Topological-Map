# Genetic Algorithm-Based Robot Path Planning with the Extraction of Topological Map

## Overview
This project includes several implementations and methodologies for path planning. It features codes for A*, Theta*, DKGA, DAG+DKGA, and the proposed methods. 

## Methods explanation
A*: We adapted conventional A* to allow it to consider smoothness.

Theta*: A famous improved version of conventional A*.

DKGA: We implemented the algorithm from the paper 'Domain Knowledge-Based Genetic Algorithms for Mobile Robot Path Planning with Single and Multiple Targets' based on our understanding of its content.

DAG: We translated the MATLAB code from the paper 'An Effective Initialization Method for Genetic Algorithm-Based Robot Path Planning Using a Directed Acyclic Graph' into Python.

The proposed methods: For more details, please refer to the paper.

### Directory Structure:
- **A_star_theta_star**: Code for Theta* and A* (smooth versions).
- **GA_paper**: Implementation of DKGA and DAG+DKGA.
- **map**: Maps for single-goal path planning experiments.
- **multi-goal**: Code for multi-goal path planning.
- **proposed_method**: Code for the proposed method.

## Installation Instructions

To run the code, you need the following Python libraries. You can install them using `conda` by running the following commands:

```bash
conda install pandas
conda install matplotlib
conda install networkx
conda install xlrd
conda install scipy
