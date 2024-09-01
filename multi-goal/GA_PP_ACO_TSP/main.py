# genetic algorithm search of the one max optimization problem
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ACO_complete import ACO_solver
import config
from functions import *
from genetic_algorithm_single_target import genetic_algorithm
from itertools import combinations
import pickle


# test MAP in paper
MAP = pd.read_excel("../map_multi.xls",header=None,sheet_name="middle_b").values
config.MAP_SIZE = MAP.shape[0]
calculate_start_goal(MAP)


score_history = []
target_counts_history = []
best_all, score_all = {}, {}

valid_nodes_list = valid_nodes_generation(MAP)
for i in combinations([config.p_start] + config.p_end, 2):
    two_targets = i if i[0] < i[1] else (i[1], i[0])
    pop = initialize_segment_best(two_targets[0], two_targets[1],config.POPULATION_PP, valid_nodes_list,  MAP)
    best, score, pop = genetic_algorithm(MAP, pop)
    best_all[two_targets] = best
    score_all[two_targets] = score
print("initialization OK")

# TSP
plot_paths_same_map(best_all, MAP,False)
fitness = ACO_solver(score_all, best_all, MAP=MAP, input=True, mode="plot")
