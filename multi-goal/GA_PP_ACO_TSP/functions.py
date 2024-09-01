import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import config
import pandas as pd
from random import randint
import random
import collections

def generate_map(per):
    map = np.random.random((config.MAP_SIZE, config.MAP_SIZE))
    for i in range(map.shape[0]):
        for j in range(map.shape[0]):
            if map[i, j] < per:
                map[i, j] = 1
            else:
                map[i, j] = 0
    return map


def initialize_segment_old(point1, point2, MAP):
    """initialize paths that encounter no obstacles"""
    while 1:
        path_session = [point1]
        while len(path_session) < config.NUM_NODE + 1:
            # can connect to the goal directly
            if detect_obstacle(path_session[-1], point2, MAP):
                path_session.append(point2)
                return path_session
            else:
                # add a new node
                path_session.append(randint(0, config.MAP_SIZE * config.MAP_SIZE - 1))
                # there is an obstacle, discard the path
                if not detect_obstacle(path_session[-2], path_session[-1], MAP):
                    break

def valid_nodes_generation(MAP):
    valid_nodes_list = []
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] != 1:
                valid_nodes_list.append(calculate_number(i, j))
    return valid_nodes_list

def initialize_segment_best(point1, point2, num_path, valid_nodes_list, MAP):
    """initialize paths that encounter no obstacles"""
    def initialize_segment_single(point1, point2, MAP):
        while 1:
            path_single = [point1]
            path_single_set = (point1, point2) # nodes that cannot been chosen for a new node
            while len(path_single) < config.NUM_NODE + 1:
                # can connect to the goal directly
                if detect_obstacle(path_single[-1], point2, MAP):
                    path_single.append(point2)
                    return path_single
                else:
                    # add a new node (randomly chosen from valid_nodes_list)
                    while 1:
                        tem = random.choice(valid_nodes_list)
                        while tem in path_single_set:
                            tem = random.choice(valid_nodes_list)
                        if detect_obstacle(path_single[-1], tem, MAP):
                            path_single.append(tem)
                            break
    pop = [initialize_segment_single(point1, point2, MAP) for i in range(num_path)]
    return pop


def plot_path_single(path, MAP, block=True):
    """plot one path
        input: MAP, path, block(false to disable blocking of figure)"""
    LENGTH = MAP.shape[0]
    fig, ax = plt.subplots()
    # plot the MAP
    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i, j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                if MAP[i, j] == "S":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))
                elif MAP[i, j] == "G":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))

    row, line = [], []
    for i in path:
        row.append(i % LENGTH + 0.5)
        line.append(i // LENGTH + 0.5)
    plt.plot(row, line, color='red')
    ax.autoscale()
    plt.show(block=block)
    return fig


# plot the MAP for all the paths
def plot_paths(map, paths, block=True):
    """plot 15 paths at the same time
    input: MAP, paths, block(false to disable blocking of figure)"""
    LENGTH = map.shape[0]
    fig, ax = plt.subplots(3, 5, figsize=(16, 8))
    for k in range(len(paths)):
        if k > 14:
            break
        solution = paths[k]
        for i in range(LENGTH):
            for j in range(LENGTH):
                if map[i, j] == 1:
                    ax[k // 5, k % 5].add_patch(
                        patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
                else:
                    ax[k // 5, k % 5].add_patch(
                        patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))

        row, line = [], []
        for i in solution:
            row.append(i % LENGTH + 0.5)
            line.append(i // LENGTH + 0.5)
        ax[k // 5, k % 5].plot(row, line, color='red')
        ax[k // 5, k % 5].autoscale()
    plt.show(block=block)
    return fig


def calculate_coordinate(number):
    y, x = divmod(number, config.MAP_SIZE)
    return x, y


def calculate_number(x, y):
    return y * config.MAP_SIZE + x


# find out if two nodes can be connected directly
def detect_obstacle(node1, node2, MAP):
    """True: no obstacle
    False: obstacle in the segment"""
    row1, line1 = calculate_coordinate(node1)
    row2, line2 = calculate_coordinate(node2)
    EPS = 1e-6
    if row1 != row2:
        k = (line2 - line1) / (row2 - row1)
        b = line1 - k * (row1 + 0.5) + 0.5
        row_min, row_max = min(row1, row2), max(row1, row2)
        rows = [row_min + 0.5] + [i for i in range(row_min + 1, row_max + 1)] + [row_max + 0.5]
        lines = [i * k + b for i in rows]

        if k > 0:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] + EPS)
                line_2_int = int(lines[i + 1] - EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int + 1):
                    if MAP[row_current][j] == 1:
                        return False
        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    if MAP[row_current][j] == 1:
                        return False
    else:
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1, line2 + line_direction, line_direction):
            if MAP[row1][i] == 1:
                return False
    return True


### be cautious ###
### to be tested
def detect_obstacle_plus(node1, node2, MAP):
    """True: no obstacle
    False: obstacle in the segment
    intermediate nodes will also be returned"""

    row1, line1 = calculate_coordinate(node1)
    row2, line2 = calculate_coordinate(node2)
    nodes_middle = []
    EPS = 1e-6
    if row1 != row2:
        k = (line2 - line1) / (row2 - row1)
        b = line1 - k * (row1 + 0.5) + 0.5
        row_min, row_max = min(row1, row2), max(row1, row2)
        rows = [row_min + 0.5] + [i for i in range(row_min + 1, row_max + 1)] + [row_max + 0.5]
        lines = [i * k + b for i in rows]

        if k > 0:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] + EPS)
                line_2_int = int(lines[i + 1] - EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int + 1):
                    nodes_middle.append(calculate_number(row_current, j))
                    if MAP[row_current][j]:
                        return False, None
        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    nodes_middle.append(calculate_number(row_current, j))
                    if MAP[row_current][j]:
                        return False, None
        nodes_middle.pop(-1)
        nodes_middle.pop(0)
        return True, nodes_middle

    elif line1 != line2:
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1+line_direction, line2, line_direction):
            nodes_middle.append(calculate_number(row1, i))
            if MAP[row1][i]:
                return False, None
        return True, nodes_middle
    else:
        return True, []


def generate_middle_nodes(node1, node2):
    """intermediate nodes will be returned
       input: two nodes
       tem: middle nodes between those two nodes"""
    row1, line1 = calculate_coordinate(node1)
    row2, line2 = calculate_coordinate(node2)
    nodes_middle = []
    EPS = 1e-6
    if row1 != row2:  # not in the same row
        k = (line2 - line1) / (row2 - row1)
        b = line1 - k * (row1 + 0.5) + 0.5
        row_min, row_max = min(row1, row2), max(row1, row2)
        rows = [row_min + 0.5] + [i for i in range(row_min + 1, row_max + 1)] + [row_max + 0.5]
        lines = [i * k + b for i in rows]

        if k > 0:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] + EPS)
                line_2_int = int(lines[i + 1] - EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int + 1):
                    nodes_middle.append(calculate_number(row_current, j))
        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    nodes_middle.append(calculate_number(row_current, j))
        # adjust the order
        if node1 == nodes_middle[-1]:
            nodes_middle.reverse()
        nodes_middle.pop(-1)
        nodes_middle.pop(0)
        return nodes_middle

    elif line1 != line2:  # in the same row, but not the same node
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1+line_direction, line2, line_direction):
            nodes_middle.append(calculate_number(row1, i))
        return nodes_middle

    else:  # the same node, no middle nodes
        return []


def generate_all_middle_nodes(path):
    """generate all the middle nodes for a path
       input: key nodes
       tem: all the middle nodes"""
    path2 = [path[0]]
    for i in range(len(path) - 1):
        path2 = path2 + generate_middle_nodes(path[i], path[i + 1]) + [path[i+1]]
    return path2
# MAP = pd.read_csv("C:/Users//97512//Matlab/GA_history/GA2/map3").values.astype("int")
# MAP = MAP.T
# map_size = MAP.shape[0]
# for i in range(2):
#     best = list(MAP(int, input().split()))
#     print(best)
#     plt_map(MAP, best)


def write_result(result):
    """ouput the average of result to an Excel file
        input: array: [[]]
        ouput: a file"""
    df = pd.DataFrame(result)
    df.to_excel("result_single_paper.xlsx")


def calculate_start_goal(MAP):
    """generate the start point and goals
        input: MAP
        tem: None(changes are made to config)"""
    goals = []
    ind = 0
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] == "S":
                config.p_start = calculate_number(i, j)
                config.target_encoding[-1] = config.p_start
            elif MAP[i][j] == "G":
                goals.append(calculate_number(i, j))
                config.target_encoding[ind] = goals[-1]
                ind += 1
    config.p_end = goals

def plot_paths_same_map(paths, MAP, block=True):
    """plot multiple paths on the same MAP"""
    LENGTH = MAP.shape[0]
    fig, ax = plt.subplots()

    # plot the MAP
    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i, j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                if MAP[i, j] == "S":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))
                elif MAP[i, j] == "G":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))

    # plot paths
    for path in paths.values():
        row, line = [], []
        for i in path:
            row.append(i % LENGTH + 0.5)
            line.append(i // LENGTH + 0.5)
        plt.plot(row, line, color='red')
        ax.autoscale()
    plt.show(block=block)

def pop_order_to_path(pop_order, best_all):
    """generate the best path in a given order
        input: pop_order, best_all(the best path between two targets)
        ouput: path"""
    pop_order = [config.target_encoding[i] for i in pop_order]
    if config.p_start < pop_order[0]:
        two_targets = (config.p_start, pop_order[0])
        path = best_all[two_targets]
    else:
        two_targets = (pop_order[0], config.p_start)
        path = list(reversed(best_all[two_targets]))
    # path = []
    for index in range(len(pop_order) - 1):
        if pop_order[index] < pop_order[index + 1]:
            two_targets = (pop_order[index], pop_order[index + 1])
            path = path + best_all[two_targets]
        else:
            two_targets = (pop_order[index + 1], pop_order[index])
            path = path + list(reversed(best_all[two_targets]))
    return path
