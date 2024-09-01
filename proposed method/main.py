# genetic algorithm search of the one max optimization problem
from matplotlib import pyplot as plt
import config
from functions import load_map
from evaluation import evaluation
from genetic_algorithm import genetic_algorithm
import math
import time
from potential_field import Pf
import matplotlib as mpl
mpl.use('TkAgg')  # interactive mode

# load map
MAP = load_map("6")
config.MAP_SIZE = MAP.shape[0]

# search map
time_start = time.time()
pf = Pf(MAP, config.p_start, config.p_end, 1, 1.35)
pmap = pf.search_map()  # key_sect: division points of the map
print("done")

# initialize path
pop = [pf.initialize_path() for i in range(config.POPULATION)]

# genetic algorithm
best, score, pop, score_history_one_time = genetic_algorithm(pop, pmap, pf, MAP)

# result
print(score)
print(best)
print("time", time.time() - time_start)
pf.plot_path_single_light(best, False, "pf")
fig = plt.figure()
plt.plot(score_history_one_time)
plt.xlabel("iteration")
plt.ylabel("fitness value")
plt.title("self_work")
plt.show()

# # # debug
# pf.draw_structure(False, "all")
# pf.plot_map(False, "value", )  # slow on large maps
# pf.plot_map(False, "value_all")  # slow on large maps
# pf.plot_map_key_sect(True, "common")