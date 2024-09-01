# genetic algorithm search of the one max optimization problem
from evaluation import evaluation
from functions import generate_map, plot_path_single, initialize_pop_v3, write_result, initialize_pop_v2, \
    calculate_start_goal, transform_number, plot_path_single_light, calculate_coordinate, load_map
from genetic_algorithm import genetic_algorithm
import matplotlib.pyplot as plt
import config
import time

config.weight = 1
MAP = load_map("6")
time_start = time.time()
score_history = []
for i in range(1):
    score_history_one_time = []
    pop = initialize_pop_v3(MAP)
    best, score, pop, score_history_one_time = genetic_algorithm(MAP, pop, score_history_one_time)
    score_history.append(score_history_one_time)
    print(time.time()-time_start)
print(score)
plt.plot(score_history_one_time)
best = [calculate_coordinate(i) for i in best]
print(best)
plot_path_single_light(best, MAP, calculate_coordinate(config.p_start), calculate_coordinate(config.p_end))
