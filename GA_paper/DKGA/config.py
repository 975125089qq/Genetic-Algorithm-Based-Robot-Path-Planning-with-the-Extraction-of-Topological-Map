import pandas as pd

# define the total iterations
ITERATION = 15
# define the population size
POPULATION = 15
# number of nodes
NUM_NODE = 50
# crossover rate
P_CROSS = 0.8
# mutation possibility
P_MUT = 0.1
# insertion possibility
P_IN = 0.3
# deletion possibility
P_D = 0.3
# MAP_SIZE
MAP_SIZE = 16
# penalty if there is a obstacle
PENALTY = 1_000_000
# starting position
p_start = None
# ending position
p_end = None
# the radius of circle(for plot)
radius = 0.3
# scale factor (weight of distance in the fitness function)
SCALE_FACTOR = 1
# magnitude_smoothness/magnitude_length
weight = 1