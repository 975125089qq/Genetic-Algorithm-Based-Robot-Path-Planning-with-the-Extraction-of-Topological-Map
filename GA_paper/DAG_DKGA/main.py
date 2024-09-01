from draw_dag import draw_dag
from draw_path import draw_path
from sgmp import sgmp
from domain_knowledge.functions import calculate_start_goal, plot_path_single, calculate_coordinate, load_map
from domain_knowledge.evaluation import evaluation
import pandas as pd
from domain_knowledge.genetic_algorithm import genetic_algorithm
import config
import scipy.io
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# load map
sp, dp, MAP = load_map("6")
config.MAP_SIZE = len(MAP)

numPt = 15
pathSet,execTime,dag = sgmp(MAP,sp,dp,config.POP_NUM,numPt)
# for path in pathSet:
#     print(path)
#     print(evaluation(path, MAP))
#     draw_path(MAP, path, True)


path = pathSet[0]
# draw_path(MAP,path, False)
draw_dag(MAP,dag, False)
best, score, pop, score_history= genetic_algorithm(pathSet, MAP)
print("time", execTime)
# print(evaluation(best, MAP, "length"))
print("fitness", evaluation(best, MAP, "fitness"))
print([calculate_coordinate(i) for i in best])
fig = plt.figure()
plt.plot(score_history)
draw_path(MAP, best)