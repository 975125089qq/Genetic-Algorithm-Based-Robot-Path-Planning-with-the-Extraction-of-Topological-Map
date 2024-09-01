import pandas as pd

# define the total iterations
ITERATION = 20
# population size of path planning
POPULATION_PP = 20
# elitism
elite_rate_TSP = 0.3
# crossover rate
P_CROSS = 0.8
# crossover rate for TSP
P_CROSS_TSP = 0.9
# mutation possibility
P_MUT = 0.2
# MAP_SIZE
MAP_SIZE = 16
# penalty if there is a obstacle
PENALTY = 100
# possibility of deletion
P_D = 0.4
# possibility of insertion
P_IN = 0.3
# starting position
p_start = 0
# ending position
p_end = [115, 156, MAP_SIZE * MAP_SIZE - 1]
# symbol for start point in TSP
p_start_TSP = -1
# the radius of circle(for plot)
radius = 0.3
# the width of line when drawing the number of segments
LINE_WIDTH = 12
# penalty of path that cannot be passed
PENALTY_TSP = 1000
# the possibility to choose a branch road
POS_BRANCH = 0