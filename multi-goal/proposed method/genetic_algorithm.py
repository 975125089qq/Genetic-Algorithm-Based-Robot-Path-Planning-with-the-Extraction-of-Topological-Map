# genetic algorithm search of the one max optimization problem
from random import shuffle
import random
from evaluation import evaluation
from operators import insertion_deletion
from functions import no_obstacle
import config


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
    index = [random.random() for _ in range(pop_num)]
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
def crossover(selected, pmap, key_sect, MAP):
    parity = len(selected) % 2  # to see if new_pop is plural or singular
    for i in range(0, len(selected) - parity, 2):
        if random.random() > config.P_CROSS:
            continue
        # find out common key nodes
        p1, p2 = selected[i], selected[i+1]
        key_node1 = {pmap[j[0]][j[1]].tree_pos: ind for ind, j in enumerate(p1[1:-2]) if pmap[j[0]][j[1]].tree_pos in key_sect}
        key_node2 = {pmap[j[0]][j[1]].tree_pos: ind for ind, j in enumerate(p2[1:-2]) if pmap[j[0]][j[1]].tree_pos in key_sect}
        common = list(key_node1.keys() & key_node2.keys())
        if len(common) == 0:
            continue
        num_common = random.choice(common)
        index1, index2 = key_node1[num_common]+1, key_node2[num_common]+1  # start point should be counted

        # create children
        if not no_obstacle(p1[index1-1], p2[index2], MAP) or not no_obstacle(p2[index2-1],p1[index1], MAP): # infeasible
            continue
        selected[i], selected[i+1] = p1[:index1] + p2[index2:], p2[:index2] + p1[index1:]
        #  debug
        # if evaluation(selected[i], MAP) > config.PENALTY or evaluation(selected[i+1], MAP) > config.PENALTY:
        #     print("wrong")
        #     plot_path_single(p1, MAP, False)
        #     plot_path_single(p2, MAP, False)
        #     plot_path_single(selected[i], MAP, False)
        #     plot_path_single(selected[i+1], MAP, False)
        #     a = 1
        #     a += 1

# mutation operator
# need improvement.......
def mutation(path, MAP):
    for mut_point in range(1, len(path[1:-1])):
        if random.random() < config.P_MUT:
            x, y = path[mut_point]
            direction = [-1, 0, 1]
            evaluation_result = float("Inf")
            mutation_result = path[mut_point]
            for i in direction:
                x_mut = x + i
                if 0 <= x_mut < config.MAP_SIZE:
                    for j in direction:
                        y_mut = y + j
                        if 0 <= y_mut < config.MAP_SIZE and MAP[x_mut][y_mut] != 1:
                            number_mutated = (x_mut, y_mut)
                            path[mut_point] = number_mutated
                            tem = evaluation(path[mut_point - 1:mut_point + 2], MAP)
                            if evaluation_result > tem:
                                evaluation_result = tem
                                mutation_result = number_mutated
            path[mut_point] = mutation_result
    return path


# genetic algorithm
def genetic_algorithm(pop, pmap ,key_sect, MAP):
    best_fitness = float("Inf")
    best_fitness_gen_before = None
    score_history = []  # elitist strategy
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
            best, best_fitness = pop[index_best], fitness[index_best],
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
        crossover(selected, pmap, key_sect, MAP)

        for path in selected:
            # # circuit removal
            # path_new = circuit_removal(path)

            # mutation
            path_new = mutation(path, MAP)

            # insertion and deletion
            ## this one needs improvement
            path_new = insertion_deletion(path_new, MAP)

            # pass the mutation to the children
            children.append(path_new)

        # replace population
        pop = children
    return [best, best_fitness, pop, score_history]
