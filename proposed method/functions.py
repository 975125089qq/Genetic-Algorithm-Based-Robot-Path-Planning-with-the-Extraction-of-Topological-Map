import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import config
import pandas as pd
import os
import scipy
import scipy.io
from PIL import Image

def plot_path_single(path, MAP, block=True):
    """plot one path"""
    LENGTH = len(MAP)
    fig, ax = plt.subplots()

    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i][j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
                # ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                # ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, fill=True, color="white"))
                if MAP[i][j] == "S":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=1, edgecolor="black", fill=True))
                elif MAP[i][j] == "G":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=1, edgecolor="black", fill=True))

    row, line = [], []
    for i in path:
        row.append(i[0] + 0.5)
        line.append(i[1] + 0.5)
    plt.plot(row, line, color='red')
    ax.autoscale()
    plt.show(block=block)


def plot_path_single_with_potential(path, MAP, map_potential, block=True):
    """plot one path"""
    LENGTH = MAP.shape[0]
    fig, ax = plt.subplots()

    # potential MAP
    for i in range(config.MAP_SIZE):
        for j in range(config.MAP_SIZE):
            if map_potential[i][j] == 1000:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                ax.text(i, j, str(map_potential[i][j][0]))

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
        row.append(i[0] + 0.5)
        line.append(i[1] + 0.5)
    plt.plot(row, line, color='red')
    ax.autoscale()
    plt.show(block=block)


# plot the MAP for all the paths
def plot_paths(map, paths, block=True):
    """plot 15 paths at the same time"""
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


# plot the MAP for all the paths
def plot_paths_with_potential(paths, map, map_potential, block=True):
    """plot 15 paths at the same time"""
    LENGTH = map.shape[0]
    fig, ax = plt.subplots(3, 5, figsize=(16, 8))
    for k in range(len(paths)):
        if k > 14:
            break
        solution = paths[k]

        # potential MAP
        for i in range(config.MAP_SIZE):
            for j in range(config.MAP_SIZE):
                if map_potential[i][j] == 1000:
                    ax[k // 5, k % 5].add_patch(
                        patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
                else:
                    ax[k // 5, k % 5].add_patch(
                        patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                    ax[k // 5, k % 5].text(i, j, str(map_potential[i][j]))

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
            row.append(i[0] + 0.5)
            line.append(i[1] + 0.5)
        ax[k // 5, k % 5].plot(row, line, color='red')
        ax[k // 5, k % 5].autoscale()
    plt.show(block=block)


# find out if two nodes can be connected directly
def no_obstacle(node1, node2, MAP):
    """True: no obstacle
    False: obstacle in the segment"""
    if (min(node1, node2), max(node1, node2)) in config.line_of_sight:
        return config.line_of_sight[(min(node1, node2), max(node1, node2))]
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
                        config.line_of_sight[(min(node1, node2), max(node1, node2))] = False
                        return False
        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    if MAP[row_current][j] == 1:
                        config.line_of_sight[(min(node1, node2), max(node1, node2))] = False
                        return False
    else:
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1, line2 + line_direction, line_direction):
            if MAP[row1][i] == 1:
                config.line_of_sight[(min(node1, node2), max(node1, node2))] = False
                return False
    config.line_of_sight[(min(node1, node2), max(node1, node2))] = True
    return True


def generate_middle_nodes(node1, node2):
    """intermediate nodes will be returned
       input: two nodes
       tem: middle nodes between those two nodes"""
    row1, line1 = node1
    row2, line2 = node2
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
                    nodes_middle.append((row_current, j))
        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    nodes_middle.append((row_current, j))
        # adjust the order
        if node1 == nodes_middle[-1]:
            nodes_middle.reverse()
        nodes_middle.pop(-1)
        nodes_middle.pop(0)
        return nodes_middle

    elif line1 != line2:  # in the same raw, but not the same node
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1, line2, line_direction):
            nodes_middle.append((row1, i))
        nodes_middle.pop(0)  # needs improvement for speed
        return nodes_middle

    else:  # the same node, no middle nodes
        return []


def calculate_start_goal(MAP):
    """generate the start point and goals
        input: MAP
        tem: None(changes are made to config)"""
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] == "S":
                config.p_start = (i, j)
            elif MAP[i][j] == "G":
                goals = (i, j)
    config.p_end = goals


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


def smoothness(pos_re1, pos_re2):
    smoothness = 0  # 0 degrees
    tem = np.dot(pos_re1, pos_re2)
    if tem > 0:
        if not (pos_re1[0] == pos_re2[0] and pos_re1[1] == pos_re2[1]):
            smoothness = config.SMOOTH_PENALTY[0]  # 45 degrees
    elif tem == 0:
        smoothness = config.SMOOTH_PENALTY[1]  # 90 degrees
    else:
        smoothness = config.SMOOTH_PENALTY[2]  # 135 degrees
    return smoothness


def calculate_smooth(cur_value, node, node_parent):
    if cur_value == 0:
        node.value_all = 1
        return
    node_grandparent = node_parent.mother_node
    vector1 = (node.pos[0] - node_parent.pos[0], node.pos[1] - node_parent.pos[1])
    vector2 = (node_parent.pos[0] - node_grandparent.pos[0], node_parent.pos[1] - node_grandparent.pos[1])
    value_all = node_parent.value_all + smoothness(vector1, vector2) + 1
    if value_all < node.value_all:
        node.value_all = value_all
        node.mother_node = node_parent


def load_image(file_path):
    """ load map from png file

    Args:
        file_path str:

    Returns:
        image_array ndarray: map
    """
    # Open the JPG file
    image = Image.open(file_path)
    image = image.convert("L")
    new_size = (1000, 1000)
    image = image.resize(new_size)
    image_array = np.array(image)
    return image_array

def load_map(map_id):
    """

    Args:
        map_id (str): the name for the map

    Returns:
        list: map
    """
    # small maps
    if map_id in "123456":
        MAP = pd.read_excel("../map/map.xls",header=None, sheet_name=map_id).values
        MAP = enlarge_map(MAP)
        calculate_start_goal(MAP)

    # google map from paper
    else:
        if map_id == "9":
            path_file = os.path.join("../map/big_map", "tokyo2000.mat")
        elif map_id == "7":
            path_file = os.path.join("../map/big_map", "vatican2000.mat")
        elif map_id == "8":
            path_file = os.path.join("../map/big_map", "triumph2000.mat")
        mat_data = scipy.io.loadmat(path_file)

        MAP = mat_data["map"]
        sp = mat_data["sp"][0][0] - 1
        dp = mat_data["dp"][0][0] - 1
        config.p_start = (sp // len(MAP), sp % len(MAP))
        config.p_end = (dp // len(MAP), dp % len(MAP))

    config.SCALE_FACTOR = return_weight(map_id)
    config.SMOOTH_PENALTY = [i * config.SCALE_FACTOR for i in [0.7853981634, 1.5707963268, 2.3561944902]]
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
