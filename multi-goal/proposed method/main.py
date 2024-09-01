""" multi-targets path planning using potential field"""
import pandas as pd
import config
from functions import *
from genetic_algorithm import genetic_algorithm
from ACO_incomplete import ACO_solver
from potential_field import Pf
import matplotlib as mpl
# from ACO_incomplete import ACO_solver
import pickle
mpl.use('TkAgg')  # interactive mode works with this, pick one


# load the MAP
""" To do: when generating the path between a pair of targets, the nodes are selected randomly
    can sort the sections by value_all, and generate the path"""
MAP = pd.read_excel("../map_multi.xls",header=None,sheet_name="middle_b").values
config.MAP_SIZE = MAP.shape[0]
calculate_start_goal(MAP)

pf = Pf(MAP, config.p_start, config.p_end)
pmap, key_sect, conn = pf.search_map()
pop = pf.initialize_path_all()

best_all, score_all, score_history_all = {}, {}, {}
for two_targets in pop.keys():
    best_two_targets, score_two_targets, _,  score_history_two_targets = genetic_algorithm(pop[two_targets], pf.pmap, key_sect, MAP)
    # best_two_targets = compatibility_new_expression(best_two_targets)
    best_all[two_targets] = best_two_targets
    score_all[two_targets] = score_two_targets
    score_history_all[two_targets] = score_history_two_targets
plot_paths_same_map(best_all, MAP, False)
targets_connection_table = targets_connect_table_generation(pop)

# TSP
ACO_solver(score_all, best_all, targets_connection_table, pf.pmap, MAP, True, mode="plot")