# genetic algorithm search of the one max optimization problem
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from random import randint, random,shuffle
from domain_knowledge.evaluation import evaluation
from domain_knowledge.operators import detect_cycle,circuit_removal, insertion_deletion, refinement
from domain_knowledge.functions import calculate_coordinate, calculate_number, plot_paths, plot_path_single
import config
from draw_path import draw_path

# Roulette Wheel Selection
def selection(pop, scores):
    # create a roulette wheel
    scores = [1 / i for i in scores]
    p_score = scores / sum(scores)
    pop_num = len(pop)
    p_score_cumsum = [0] * pop_num
    for i in range(pop_num):
        p_score_cumsum[i] = p_score_cumsum[i - 1] + p_score[i]

    # select
    new_pop = []
    index = [random() for i in range(pop_num)]
    index.sort()
    pointer = 0
    pos_wheel = 0
    while (pointer < pop_num):
        if (index[pointer]) < p_score_cumsum[pos_wheel]:
            new_pop.append(pop[pos_wheel])
            pointer += 1
        else:
            pos_wheel += 1
    shuffle(new_pop)
    return new_pop


# crossover two parents to create two children
def crossover(p1, p2):
    if random() < config.P_CROSS:
        # create childrens
        length = min(len(p1),len(p2))
        cross_place = randint(1,length-1) # pay attention???

        c1 = p1[:cross_place] + p2[cross_place:]
        c2 = p2[:cross_place] + p1[cross_place:]
        return c1, c2
    else:
        return p1, p2


# mutation operator
def mutation(path, MAP):
    for mut_point in range(1,len(path[1:-1])):
        if random() < config.P_MUT:
            x, y = calculate_coordinate(path[mut_point])
            direction = [-1,0,1]
            evaluation_result = float("Inf")
            mutation_result = path[mut_point]
            for i in direction:
                x_mut = x + i
                if 0 <= x_mut < config.MAP_SIZE:
                    for j in direction:
                        y_mut = y + j
                        if 0 <= y_mut < config.MAP_SIZE and MAP[x_mut][y_mut] != 1:
                            number_mutated = calculate_number(x_mut, y_mut)
                            path[mut_point] = number_mutated
                            tem = evaluation(path[mut_point - 1:mut_point + 2], MAP)
                            if evaluation_result > tem:
                                evaluation_result = tem
                                mutation_result = number_mutated
            path[mut_point] = mutation_result
    return path

# genetic algorithm
def genetic_algorithm(pop, MAP):
    # keep track of best solution
    # print("OK")
    best, best_fitness = 0, float("Inf")
    best_fitness_gen_before = None # elitist strategy
    score_history = []
    # iterations
    for gen in range(config.ITERATION):
        score_history.append(best_fitness)
        # evaluate all candidates in the population
        fitness = [evaluation(path, MAP) for path in pop]
        # find the best and worst solution in this generation
        worst_fitness_current_gen = 0
        best_fitness_current_gen = float("Inf")
        for i in range(len(pop)):
            if fitness[i] > worst_fitness_current_gen:
                worst_fitness_current_gen = fitness[i]
                index_worst = i
            if fitness[i] < best_fitness_current_gen:
                best_fitness_current_gen = fitness[i]
                index_best = i
        if best_fitness_current_gen < best_fitness:
            best, best_fitness = pop[index_best], fitness[index_best]
            # print(f"gen: {gen}\n length: {shortest_length} ")

        # elitist strategy
        if best_fitness_gen_before and worst_fitness_current_gen > best_fitness_gen_before:
            pop[index_worst] = best_path_gen_before
        best_fitness_gen_before = best_fitness_current_gen
        best_path_gen_before = pop[index_best]

        # selection
        selected = selection(pop, fitness)

        # create the next generation
        children = []

        # crossover
        parity = len(selected) % 2        # to see if new_pop is plural or singular
        for i in range(0, len(selected) - parity, 2):
            selected[i], selected[i+1] = crossover(selected[i], selected[i + 1])

        for path in selected:
            # circuit removal
            path_new = circuit_removal(path, MAP)
            # mutation
            path_new = mutation(path_new, MAP)
            # insertion and deletion
            path_new = insertion_deletion(path_new, MAP)
            # refinement
            path_new = refinement(path_new, MAP)
            # pass the mutation to the children
            children.append(path_new)

        # replace population
        pop = children
    return [best, best_fitness, pop, score_history]





