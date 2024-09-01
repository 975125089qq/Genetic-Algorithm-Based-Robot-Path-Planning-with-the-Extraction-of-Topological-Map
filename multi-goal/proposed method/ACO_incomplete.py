import math
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from TSP.ACO_test.ant_library.aco import ACO, Graph
from genetic_algorithm import genetic_algorithm
import pandas as pd
import config
from functions import plot_path_single, calculate_start_goal, pop_order_to_path, duplicate_targets_deletion
import pickle
from potential_field import Pf

def ACO_solver(score_all=None, best_all=None, targets_connection_table=None, map_potential=None, MAP=None, input=False, mode="average"):
    # if there is no input
    if not input:
        map_name = "big_a"
        MAP = pd.read_excel("C:/Users/97512/OneDrive/デスクトップ/Matlab/MAP/map_domain_knowledge/table.xls",
                            header=None, sheet_name=map_name).values
        config.MAP_SIZE = MAP.shape[0]
        calculate_start_goal(MAP)

        # load data of the path
        with open("tem_data/" +map_name +  "/score_all.pkl", "rb") as tf:
            score_all = pickle.load(tf)
        with open("tem_data/"+ map_name + "/best_all.pkl", "rb") as tf:
            best_all = pickle.load(tf)
        with open("tem_data/"+ map_name + "/targets_connection_table.pkl", "rb") as tf:
            targets_connection_table = pickle.load(tf)
        with open("tem_data/"+ map_name + "/map_potential.pkl", "rb") as tf:
            map_potential = pickle.load(tf)

    rank = len(config.p_end) + 1
    # print(rank)
    config.PENALTY_TSP = max(score_all.values()) * int(math.sqrt(rank))

    cost_matrix = []
    for target1 in range(-1, rank - 1):
        row = []
        for target2 in range(-1, rank - 1):
            if (target1, target2) in score_all:
                row.append(score_all[(target1, target2)])
            elif (target2, target1) in score_all:
                row.append(score_all[(target2, target1)])
            else:
                row.append(config.PENALTY_TSP)
        cost_matrix.append(row)
    aco = ACO(int(rank*30), 100, 1.0, 10.0, 0.5, 10, 0, targets_connection_table)
    graph = Graph(cost_matrix, rank)
    target_order, fitness = aco.solve(graph)
    target_order = [target-1 for target in target_order[1:]]
    if mode != "average": # visualization
        print('fitness: {}, path: {}'.format(fitness, target_order))
        print("number of duplicate targets",len(target_order)-rank+1)

        path_TSP = pop_order_to_path(target_order, best_all, )
        plot_path_single(path_TSP, MAP, False)
        aco.plot_history(False)

    target_order, TSP_result, best_all, score_all = duplicate_targets_deletion(fitness, best_all, score_all, cost_matrix, target_order, generate_new_path, map_potential, MAP)
    path_TSP = pop_order_to_path(target_order, best_all)
    if mode != "average": # visualization
        print()
        print("fitness_new:", TSP_result)
        print("number of duplicate targets",len(target_order)-rank+1)
        print(target_order)
        plot_path_single(path_TSP, MAP, True)
    return TSP_result

def generate_new_path(target_before, target_middle, target_after, pmap, MAP):
    if target_before < target_after:
        target1, target2 = target_before, target_after
    else:
        target1, target2 = target_after, target_before
    targets_connected = {target1, target_middle, target2}
    pf = Pf(MAP, None, None)
    pmap_new, key_sect, _ = pf.search_map_single(target2, targets_connected, pmap)
    pop = [pf.initialize_path(target1, target2, ACO_flag=True) for _ in range(config.POPULATION)]
    best_two_targets, score_two_targets, _,  score_history_two_targets = genetic_algorithm(pop, pmap_new, key_sect, MAP)
    return best_two_targets, score_two_targets

if __name__ == '__main__':
    ACO_solver()
