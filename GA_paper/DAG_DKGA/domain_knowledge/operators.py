from .evaluation import map_dist, evaluation
import random
from .functions import calculate_coordinate, calculate_number, generate_all_middle_nodes, plot_paths, plot_path_single, generate_middle_nodes
import matplotlib.pyplot as plt
import config

def detect_cycle(path, gen):
    dic = {}
    for index, number in enumerate(path):
        if number not in dic:
            dic[number] = index
        else:
            length = index - dic[number]
            print(f"detected, generation:{gen}, length:{length}")


# to be improved
def circuit_removal(path, MAP):
    sett = set()  # all the middle nodes in path2
    path2 = []  # path after removal
    path2_with_all_middle_nodes = []  # all the middle nodes added in path2
    path_with_all_middle_nodes = generate_all_middle_nodes(path)
    for num in (path_with_all_middle_nodes):
        if num not in sett:
            if num in path:  # key node in original path
                path2.append(num)
            sett.add(num)
            path2_with_all_middle_nodes.append(num)
        else:
            while path2_with_all_middle_nodes[-1] != num:
                out = path2_with_all_middle_nodes.pop()
                if out == path2[-1]:  # key node discarded
                    path2.pop()
                sett.discard(out)
            if num != path2[-1]:
                path2.append(num)
    if evaluation(path2, MAP, type="distance") < config.PENALTY:
        return path2
    else:
        return path

# slow version of circuit removal
# for index, node in enumerate(path_with_all_middle_nodes):
#     index_node_list = []
#     key_node_list = []
#     for i in range(index+1, len(path_with_all_middle_nodes)):
#         node_examined = path_with_all_middle_nodes[i]
#         if  node_examined == node:
#             index_node_list.append(i)
#         if node_examined in path:
#             key_node_list.append(i)
#
#     if len(index_node_list) > 0:
#         del path_with_all_middle_nodes[]


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
                    path = path[:index+1] + [node_insert] + path[index+1:]
                    index += 1
            index += 1
    else:
        # deletion
        index = 0
        while index < len(path) - 2:
            if random.random() < config.P_D and map_dist(path[index], path[index + 2], MAP) < config.PENALTY:
                path.remove(path[index + 1])
            index += 1
    return path


def refinement(path, MAP):
    """if there is a right angle
       try to find a shorter path"""
    index = 0
    while index < len(path) - 2:
        if path[index] == path[index+1] or path[index+1] == path[index+2]: # the same node
            del path[index+1]
            continue

        x1, y1 = calculate_coordinate(path[index])
        x2, y2 = calculate_coordinate(path[index + 1])
        x3, y3 = calculate_coordinate(path[index + 2])
        if (x1 == x2 and y2 == y3) or (y1 == y2 and x2 == x3):  # if there is a right angle

            if map_dist(path[index], path[index + 2], MAP) < config.PENALTY:  # two nodes can be connected directly
                path.remove(path[index + 1])
                index -= 1
            else:
                if x1 == x2 and y2 == y3:  # possible shape of right angle
                    # generate intermediate nodes
                    if y1 < y2:
                        nodes_middle1 = [calculate_number(x1, y1 + i) for i in range(y2 - y1)]
                    else:
                        nodes_middle1 = [calculate_number(x1, y1 - i) for i in range(y1 - y2)]
                    if x2 > x3:
                        nodes_middle2 = [calculate_number(x3 + i, y2) for i in range(x2 - x3)]
                    else:
                        nodes_middle2 = [calculate_number(x3 - i, y2) for i in range(x3 - x2)]
                else:  # the other possible shape of right angle (y1 == y2 and x2 == x3)
                    if x1 < x2:
                        nodes_middle1 = [calculate_number(x1 + i, y1) for i in range(x2 - x1)]
                    else:
                        nodes_middle1 = [calculate_number(x1 - i, y1) for i in range(x1 - x2)]
                    if y2 > y3:
                        nodes_middle2 = [calculate_number(x2, y3 + i) for i in range(y2 - y3)]
                    else:
                        nodes_middle2 = [calculate_number(x2, y3 - i) for i in range(y3 - y2)]

                # find a shorter path
                index2 = 1
                flag = 0 # no refinement
                while index2 < len(nodes_middle1) and index2 < len(nodes_middle2):
                    if map_dist(nodes_middle1[index2], nodes_middle2[index2], MAP) < config.PENALTY:
                        path[index+1] = nodes_middle1[index2]
                        path.insert(index+2, nodes_middle2[index2])
                        flag = 1
                        break
                    index2 += 1
                if not flag: # no refinement
                    path[index+1] = nodes_middle1[-1]
                    path.insert(index+2, nodes_middle2[-1])
        index += 1
    return path
