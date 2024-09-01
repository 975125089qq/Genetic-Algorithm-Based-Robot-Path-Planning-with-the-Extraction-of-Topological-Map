from functions import no_obstacle
import random
from functions import generate_middle_nodes, plot_path_single
import config

def insertion_deletion(path, MAP):
    """insert and delete"""
    if random.random() < 0.5:
        # insertion
        index = 0
        while index < len(path) - 1:
            if random.random() < config.P_IN:
                nodes_middle = generate_middle_nodes(path[index], path[index + 1])
                if nodes_middle != []:
                    node_insert = random.choice(nodes_middle)
                    if not no_obstacle(path[index], node_insert, MAP) or not no_obstacle(node_insert, path[index+1], MAP):
                        continue
                    path = path[:index+1] + [node_insert] + path[index+1:]
                    index += 1
            index += 1
    else:
        # deletion
        index = 0
        while index < len(path) - 2:
            if random.random() < config.P_D:
                if no_obstacle(path[index], path[index+2], MAP):
                    path.pop(index + 1)
            index += 1
    return path

# to be finished and tested
def insertion_deletion_new(path, MAP):
    """insert and delete, a faster version?"""
    path_new = []
    if random.random() < 0.5:
        # insertion
        for index in range(len(path)-1):
            path_new.append(path[index])
            if random.random() < config.P_IN:
                nodes_middle = generate_middle_nodes(path[index], path[index + 1])
                if nodes_middle != []:
                    node_insert = random.choice(nodes_middle)
                    if not no_obstacle(path[index], node_insert, MAP) or not no_obstacle(node_insert, path[index+1], MAP):
                        continue
                    path_new.append(node_insert)
    else:
        # deletion
        index = 0
        while index < len(path) - 2:
            if random.random() < config.P_D:
                if no_obstacle(path[index], path[index+2], MAP):
                    path.pop(index + 1)
            index += 1
    return path_new