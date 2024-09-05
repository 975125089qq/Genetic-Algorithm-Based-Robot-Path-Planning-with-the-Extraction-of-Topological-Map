# Genetic Algorithm-Based Robot Path Planning with the Extraction of Topological Map

## Overview
This project includes several implementations and methodologies for path planning. It features codes for A*(smooth), Theta*, DKGA, DAG+DKGA, and the proposed methods. 

In our problem setting, we consider both **path length** and **smoothness**. We deal with **any-angle path planning**.

## Methods in Comparison
**A star(smooth)**: We adapted conventional A* to allow it to consider smoothness.

**Theta star**: A famous improved version of conventional A*.

**DKGA**: We implemented the algorithm from the paper 'Domain Knowledge-Based Genetic Algorithms for Mobile Robot Path Planning with Single and Multiple Targets' based on our understanding of its content.

**DAG**: We translated the MATLAB code from the paper 'An Effective Initialization Method for Genetic Algorithm-Based Robot Path Planning Using a Directed Acyclic Graph' into Python.

**The proposed methods**: For more details, please refer to the paper.

## Directory Structure:
- **A_star_theta_star**: Code for Theta* and A* (smooth versions).
- **GA_paper**: Implementation of DKGA and DAG+DKGA.
- **map**: Maps for single-goal path planning experiments.
- **multi-goal**: Code for multi-goal path planning.
- **proposed_method**: Code for the proposed method.

## Key Functions for Building the Topological Map
```
pf = Pf(MAP, start_point, end_point, threshold_loop, threshold_parent) # initialize the settings
topological_map = pf.search_map() # search map and build the topological map
```

### Descriptions
threshold_loop and threshold_parent: These are two crucial hyperparameters for the proposed initialization method. Increasing their values makes the algorithm more likely to explore paths with a worse estimated fitness value. (The fitness estimation is based on movement in eight directions, which introduces some degree of error (explained further in the paper).

threshold_loop: hyperparameter for the following situation (a longer path and a shorter path meet together)
![image](https://github.com/user-attachments/assets/5d5200da-6d4b-4892-a3cd-38a9b0773551)

threshold_parent: hyperparamter for the following situation (several paths with similar length meet together)
![image](https://github.com/user-attachments/assets/6c3efc7a-ca96-4460-80a3-c27c29b2f10d)



## Installation Instructions

To run the code, you need the following Python libraries. You can install them using `conda` by running the following commands:

```bash
conda install pandas
conda install matplotlib
conda install networkx
conda install xlrd
conda install scipy

