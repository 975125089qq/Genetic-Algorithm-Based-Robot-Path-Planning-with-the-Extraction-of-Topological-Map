import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import config
import pandas as pd
import os
import scipy
import scipy.io

from random import randint
import collections

def no_obstacle(node1, node2, MAP):
    """True: no obstacle
    False: obstacle in the segment"""
    row1, line1 = node1
    row2, line2 = node2
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
        row.append(i // LENGTH + 0.5)
        line.append(i % LENGTH + 0.5)
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
    x, y = divmod(number, config.MAP_SIZE)
    return x, y


def calculate_number(x, y):
    return x * config.MAP_SIZE + y


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
        for i in range(line1 + line_direction, line2, line_direction):
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
        path2 = path2 + generate_middle_nodes(path[i], path[i + 1]) + [path[i + 1]]
    return path2


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
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] == "S":
                sp = i * len(MAP) + j
                MAP[i][j] = 0
            elif MAP[i][j] == "G":
                dp = i * len(MAP) + j
                MAP[i][j] = 0
    return sp, dp

def load_map(map_id):
    # small maps
    if map_id in "123456":
        MAP = pd.read_excel("../../map/map.xls",header=None, sheet_name=map_id).values
        sp, dp = calculate_start_goal(MAP)
        MAP = MAP.astype("int")
    else:
        if map_id == "9":
            path_file = os.path.join("../map/big_map", "tokyo2000.mat")
        elif map_id == "7":
            path_file = os.path.join("../map/big_map", "vatican2000.mat")
        elif map_id == "8":
            path_file = os.path.join("../map/big_map", "triumph2000.mat")
        mat_data = scipy.io.loadmat(path_file)

        # Access the matrix in Python
        MAP = mat_data["map"]
        sp = mat_data["sp"][0][0] - 1
        dp = mat_data["dp"][0][0] - 1
    config.SCALE_FACTOR = return_weight(map_id)
    return sp, dp , MAP

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
