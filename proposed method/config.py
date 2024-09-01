import pandas as pd

# define the total iterations
ITERATION = 100
# define the population size
POPULATION = 100
# crossover rate
P_CROSS = 0.8
# mutation possibility
P_MUT = 0.2
# MAP_SIZE
MAP_SIZE = 16
# penalty if there is a obstacle
PENALTY = 100_000
# possibility of deletion
P_D = 0.4
# possibility of insertion
P_IN = 0.3
# starting position
p_start = None
# ending position
p_end = None
# how much should the MAP be enlarged
TIMES = 1
# weight of smoothness
SCALE_FACTOR = 1
# smoothness penalty
SMOOTH_PENALTY = [0.7853981634, 1.5707963268, 2.3561944902] # 45, 90, 135 degrees
# record line_of_sight
line_of_sight = {} # {(pos1, pos2):True(no obstacle)/False}
# magnitude_smoothness/magnitude_length
weight = 1


