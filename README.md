# Genetic Algorithm-Based Robot Path Planning with the Extraction of Topological Map (source code of the paper)

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
pf = Pf(MAP, start_point, end_point, threshold_loop, threshold_parent)  # Initialize settings
topological_map = pf.search_map()  # Search and build the topological map
```

### Descriptions
**threshold_loop** and**threshold_parent**: These are two crucial hyperparameters for the proposed initialization method. Increasing their values makes the algorithm more likely to explore paths with a worse estimated fitness value. Exploration of paths with worse estimated fitness values is necessary because the fitness estimation, based on movement in eight directions, introduces a certain degree of error (as explained in Fig. 11 of the paper).

**threshold_loop**: A hyperparameter applied when a longer path converges with a shorter path (see Fig. 7(b) in the paper).

**threshold_parent**: A hyperparameter applied when multiple paths of similar lengths converge (see Fig. 7(a) in the paper).



## Installation Instructions

To run the code, you need the following Python libraries. You can install them using `conda` by running the following commands:

```bash
conda install pandas
conda install matplotlib
conda install networkx
conda install xlrd
conda install scipy

```
## Final Note
The first author apologizes for the messy code and inconsistent naming conventions. However, he is more than happy to answer any questions regarding the code or the proposed methods. Bug reports are also welcome.
