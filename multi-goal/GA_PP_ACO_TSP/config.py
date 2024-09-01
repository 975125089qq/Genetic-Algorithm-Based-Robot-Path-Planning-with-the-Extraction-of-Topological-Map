import pandas as pd

# define the total iterations
ITERATION = 20
# population size of path planning
POPULATION_PP = 20
# crossover rate
P_CROSS = 0.8
# crossover rate for TSP
P_CROSS_TSP = 0.9
# mutation possibility
P_MUT = 0.2
# mutation rate for TSP
P_MUT_TSP = 0.05
# MAP_SIZE
MAP_SIZE = 16
# penalty if there is a obstacle
PENALTY = 100_000
# possibility of deletion
P_D = 0.4
# possibility of insertion
P_IN = 0.3
# number of intermediate nodes initialized
NUM_NODE = 30
# starting position
p_start = None
# ending position
p_end = None
# target will be indexed as 0,1,2...
target_encoding = {}
# the radius of circle(for plot)
radius = 0.3
# repeat several times to avoid error
TEST_NUM = 1
# the width of line when drawing the number of segments
LINE_WIDTH = 12
# penalty of path that cannot be passed
PENALTY_TSP = 1000
