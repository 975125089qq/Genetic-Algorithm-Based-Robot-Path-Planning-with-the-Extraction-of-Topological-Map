import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import config
import pandas as pd
from random import randint
import random
import os
import scipy
import scipy.io

def generate_map(per):
    map = np.random.random((config.MAP_SIZE, config.MAP_SIZE))
    for i in range(map.shape[0]):
        for j in range(map.shape[0]):
            if map[i, j] < per:
                map[i, j] = 1
            else:
                map[i, j] = 0
    return map

def initialize_pop_v2(MAP):
    """initialize paths that encounter no obstacles"""
    paths = []
    while len(paths) < config.POPULATION:
        while 1:
            flag = 0
            path_session = [config.p_start]
            while len(path_session) < config.NUM_NODE + 1:
                # can connect to the goal directly
                if detect_obstacle(path_session[-1], config.p_end, MAP):
                    path_session.append(config.p_end)
                    flag = 1
                    break
                else:
                    # add a new node
                    path_session.append(randint(0, config.MAP_SIZE * config.MAP_SIZE - 1))
                    # there is an obstacle, discard the path
                    if not detect_obstacle(path_session[-2], path_session[-1], MAP):
                        break
            if flag:
                paths.append(path_session)
                break
    return paths

def initialize_pop_v3(MAP):
    """initialize paths that encounter no obstacles"""
    paths = []
    valid_nodes_list = valid_nodes_generation(MAP)
    while len(paths) < config.POPULATION:
        # print(len(paths))
        while 1:
            path_flag = 0
            path_single = [config.p_start]
            path_single_set = (config.p_start, config.p_end) # nodes that cannot been chosen for a new node
            while len(path_single) < config.NUM_NODE + 1:
                # can connect to the goal directly
                if detect_obstacle(path_single[-1], config.p_end, MAP):
                    path_single.append(config.p_end)
                    path_flag = 1
                    break
                else:
                    # add a new node
                    # randomly select a node until there is a path

                    while 1:
                        tem = random.choice(valid_nodes_list)
                        while tem in path_single_set:
                            tem = random.choice(valid_nodes_list)
                        if detect_obstacle(path_single[-1], tem, MAP):
                            path_single.append(tem)
                            break
            if path_flag:
                paths.append(path_single)
                break
    return paths

def valid_nodes_generation(MAP):
    valid_nodes_list = []
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] != 1:
                valid_nodes_list.append(calculate_number(i,j))
    return valid_nodes_list

def plot_path_single(path, MAP, block=True):
    """plot one path
        input: MAP, path, block(false to disable blocking of figure)"""
    LENGTH = MAP.shape[0]
    fig, ax = plt.subplots()
    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i, j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))

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

def transform_number(number):
    x, y = divmod(number, config.MAP_SIZE)
    return calculate_number(x, y)

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
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] == "S":
                config.p_start = calculate_number(i, j)
                MAP[i][j] = 0
            elif MAP[i][j] == "G":
                goals.append(calculate_number(i, j))
                MAP[i][j] = 0
    config.p_end = goals

    # to be deleted
    if len(config.p_end) == 1:
        config.p_end = config.p_end[0]

def plot_path_single_light(path, MAP, start, end, block=True):
    plt.figure()
    MAP[start[1]][start[0]] = 0
    MAP[end[1]][end[0]] = 0
    MAP = MAP.astype("int")
    # Invert the color mapping: 1 to black and 0 to white
    cmap = plt.get_cmap("gray_r")
    plt.imshow(MAP, cmap=cmap)
    plt.plot([i[1] for i in path], [i[0] for i in path], "r-", linewidth=3)
    plt.show(block=block)

def load_map(map_id):
    """

    Args:
        map_id (str): the name for the map

    Returns:
        list: map
    """
    # test MAP in paper
    if map_id in "123456":
        MAP = pd.read_excel("../../map/map.xls",header=None, sheet_name=map_id).values
        config.MAP_SIZE = len(MAP)
        calculate_start_goal(MAP)

    else:
        if map_id == "9":
            path_file = os.path.join("../map/big_map", "tokyo2000.mat")
        elif map_id == "7":
            path_file = os.path.join("../map/big_map", "vatican2000.mat")
        elif map_id == "8":
            path_file = os.path.join("../map/big_map", "triumph2000.mat")
        mat_data = scipy.io.loadmat(path_file)

        # # Access the matrix in Python
        # MAP = pd.read_excel("C:/Users/97512/OneDrive/デスクトップ/programs/MAP/big_map.xlsx",
        #     header=None, sheet_name=map_id).values
        MAP = mat_data["map"]
        sp = mat_data["sp"][0][0]-1
        dp = mat_data["dp"][0][0]-1
        config.MAP_SIZE = MAP.shape[0]
        config.p_start = transform_number(sp)
        config.p_end = transform_number(dp)

    config.SCALE_FACTOR = return_weight(map_id)
    return MAP

def return_weight(map_id):
    weight = {
        "1": 1 /
             0.15019244547464575,
        "2": 1 /
             0.1148424870526288,
        "3": 1 /
             0.03465306739049245,
        "4": 1 /
             0.050224756446432504,
        "5": 1 /
             0.08414185971497533,
        "6": 1 /
             0.05452525913413291,
        "7": 1 /
             0.003956612169651631,
        "8": 1 /
              0.001450886754676564,
        "9": 1 /
               0.00237294651054368,
    }
    for key, item in weight.items():
        weight[key] *= config.weight
    print("weight", config.weight)
    return weight[map_id]