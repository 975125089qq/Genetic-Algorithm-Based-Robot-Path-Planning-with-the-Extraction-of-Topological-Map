# genetic algorithm search of the one max optimization problem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy
import config
from functions import calculate_start_goal, calculate_coordinate, plot_potential_map, initialize_potential_map, \
    detect_division_point, initialize_path_single, compatibility_new_expression, \
    compatibility_old_GA, plot_path_single, plot_paths_same_map
from collections import deque
from genetic_algorithm import genetic_algorithm
import pickle
from ACO_complete import ACO_solver

# test MAP in paper
MAP = pd.read_excel("../map_multi.xls",header=None,sheet_name="middle_b").values
config.MAP_SIZE = MAP.shape[0]
calculate_start_goal(MAP)

# initialize the potential MAP
map_potential_list, junctions_list = [], []
for target_num in range(-1, len(config.p_end)):
    map_potential, junctions = initialize_potential_map(MAP, target_num)
    map_potential_list.append(map_potential)
    junctions_list.append(junctions)

# initialize the path
best_all, score_all = {}, {}
for target_num in range(-1, len(config.p_end)):
    for target2 in range(target_num+1, len(config.p_end)):
        map_potential_two_targets = detect_division_point(copy.deepcopy(map_potential_list[target_num+1]), map_potential_list[target2+1], junctions_list[target_num+1])
        # plot_potential_map(map_potential_two_targets, junctions_list[target_num+1], MAP, False)
        pop = [initialize_path_single(MAP, map_potential_two_targets, target_num, target2) for i in range(config.POPULATION_PP)]
        pop = [compatibility_old_GA(path) for path in pop]
        best, score, pop = genetic_algorithm(pop, MAP)
        best_all[(target_num, target2)] = compatibility_new_expression(best)
        score_all[(target_num, target2)] = score
plot_paths_same_map(best_all, MAP, False)

ACO_solver(score_all, best_all, MAP=MAP, input=True, mode="plot")
