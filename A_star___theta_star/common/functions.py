import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from common import config
import scipy
import scipy.io
import os
import random
import pandas as pd
from PIL import Image

def plot_points_searched(close_list, MAP, map_node, block=True, mode=None):
    """plot one path"""
    fig, ax = plt.subplots()
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                if mode == "value":
                    if isinstance(map_node[j][i].fs, list):
                        ax.text(i, j + 0.7, str(round(min([k - map_node[j][i].hs for k in map_node[j][i].fs]), 1)))
                    else:
                        ax.text(i, j + 0.7, str(round(map_node[j][i].fs - map_node[j][i].hs, 1)))

    if type(close_list) is set:
        nodes_searched = [i for i in close_list]
    else:
        nodes_searched = [i.pos for i in close_list]
    for node in nodes_searched:
        if type(node[0]) is tuple:  # for a star eight directions
            node = node[0]
        ax.add_patch(patches.Rectangle(xy=(node[1], node[0]), width=1, height=1, edgecolor="yellow", fill=False))

    ax.autoscale()
    plt.subplots_adjust(left=0.04, right=1, bottom=0.04, top=1)
    plt.show(block=block)


def plot_path_single(path, MAP, block=True):
    """plot one path"""
    LENGTH = MAP.shape[0]
    fig, ax = plt.subplots()

    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i, j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
            else:
                if LENGTH <= 30:
                    ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                if MAP[i, j] == "S":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))
                elif MAP[i, j] == "G":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))

    row, line = [], []
    for i in path:
        row.append(i[0] + 0.5)
        line.append(i[1] + 0.5)
    plt.plot(row, line, color='red')
    ax.autoscale()
    plt.show(block=block)


def plot_path_single_light(path, MAP, start, end, block=True, title:str = None):
    plt.figure()
    MAP[start[1]][start[0]] = 0
    MAP[end[1]][end[0]] = 0
    MAP = MAP.astype("int")
    cmap = plt.get_cmap("gray_r")
    plt.imshow(MAP, cmap=cmap)
    plt.plot([i[1] for i in path], [i[0] for i in path], "r-", linewidth=3)
    plt.title(title)
    plt.show(block=block)

def calculate_coordinate(number):
    y, x = divmod(number, config.MAP_SIZE)
    return x, y


def calculate_number(x, y):
    return y * config.MAP_SIZE + x


# find out if two nodes can be connected directly
def detect_obstacle(node1, node2, MAP):
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


def calculate_start_goal(Node, map_data):
    """generate the start point and goals
        input: MAP
        tem: None(changes are made to config)"""
    for i in range(len(map_data)):
        for j in range(len(map_data)):
            if map_data[i][j] == "S":
                Node.start = (j, i)
            elif map_data[i][j] == "G":
                Node.goal = (j, i)
    return Node


def write_result(result):
    """ouput the average of result to an Excel file
        input: array: [[]]
        ouput: a file"""
    df = pd.DataFrame(result)
    df.to_excel("result_self_work_selection_change_v2.xlsx")


def reduce_reduent_nodes_initialization(path, MAP):
    """ reduce reduent nodes in the initialized path
        input: path
        ouput: path with reduced nodes"""
    index = 1
    path2 = [path[0]]
    while index < len(path):
        if detect_obstacle((path2[-1][0], path2[-1][1]), (path[index][0], path[index][1]), MAP):
            index += 1
        else:
            path2.append((path[index - 1]))
    path2.append(path[-1])
    return path2


def enlarge_map(MAP):
    """make the MAP much bigger by adjusting the size of each node"""
    size = len(MAP) * config.TIMES
    map2 = np.array([[0 for i in range(size)] for j in range(size)], dtype=object)
    for i in range(size):
        for j in range(size):
            if MAP[i // config.TIMES][j // config.TIMES] != "S" and MAP[i // config.TIMES][j // config.TIMES] != "G":
                map2[i][j] = MAP[i // config.TIMES][j // config.TIMES]
            elif i % config.TIMES == 0 and j % config.TIMES == 0:
                map2[i][j] = MAP[i // config.TIMES][j // config.TIMES]
            else:
                map2[i][j] = 0
    return map2


def load_map(map_id, Node):
    """

    Args:
        map_id (str): the name for the map
        choose one from "123456789"

    Returns:
        list: map
    """
    # small maps
    if map_id in "123456":
        MAP = pd.read_excel("../map/map.xls",header=None, sheet_name=map_id).values
        config.TIMES = 1
        MAP = enlarge_map(MAP)
        calculate_start_goal(Node, MAP)

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
        Node.start = (sp % len(MAP), sp // len(MAP))
        Node.goal = (dp % len(MAP), dp // len(MAP))

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
