import math
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from TSP.ACO_test.ant_library.aco_original import ACO, Graph
import pandas as pd
import config
from functions import plot_path_single, calculate_start_goal, pop_order_to_path
import pickle

def ACO_solver(score_all=None, best_all=None, MAP=None, input=False, mode="plot"):
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

    rank = len(config.p_end)+1
    config.PENALTY_TSP = max(score_all.values()) * int(math.sqrt(rank))
    cost_matrix = []
    for ind1 in range(-1, rank - 1):
        row = []
        target1 = config.target_encoding[ind1]
        for ind2 in range(-1, rank - 1):
            target2 = config.target_encoding[ind2]
            if (target1, target2) in score_all:
                row.append(score_all[(target1, target2)])
            elif (target2, target1) in score_all:
                row.append(score_all[(target2, target1)])
            else:
                row.append(config.PENALTY_TSP)
        cost_matrix.append(row)
    aco = ACO(int(rank*20), 200, 1.0, 10.0, 0.5, 10, 0)
    graph = Graph(cost_matrix, rank)
    target_order, fitness = aco.solve(graph)
    target_order = [target-1 for target in target_order[1:]]
    path_TSP = pop_order_to_path(target_order, best_all, )

    if mode == "plot":
        print('fitness: {}, path: {}'.format(fitness, target_order))
        print("number of duplicate targets",len(target_order)-rank+1)
        plot_path_single(path_TSP, MAP, False)
        aco.plot_history()
    return fitness


if __name__ == '__main__':
    ACO_solver()
