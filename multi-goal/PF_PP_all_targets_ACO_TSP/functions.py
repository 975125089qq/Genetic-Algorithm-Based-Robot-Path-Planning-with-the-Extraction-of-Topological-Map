import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import config
from random import randint
import collections
import random
import pandas as pd
from collections import deque
from random import choice
from collections import deque

def generate_map(per):
    map = np.random.random((config.MAP_SIZE, config.MAP_SIZE))
    for i in range(map.shape[0]):
        for j in range(map.shape[0]):
            if map[i, j] < per:
                map[i, j] = 1
            else:
                map[i, j] = 0
    return map


def plot_potential_map(map_potential, junctions, MAP, block=True):
    """plot one path"""
    fig, ax = plt.subplots()
    for i in range(config.MAP_SIZE):
        for j in range(config.MAP_SIZE):
            if MAP[i, j] == "S":
                ax.add_patch(
                    patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True, facecolor="r"))
            elif MAP[i, j] == "G":
                ax.text(i + 0.25, j + 0.25, config.p_end.index(calculate_number(i, j)), color="r", size=15)

            else:
                if map_potential[i][j][0] == 1000:
                    ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
                else:
                    ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                    ax.text(i, j+0.7, str(map_potential[i][j][0]))
                    junction_num = (map_potential[i][j][1])
                    pos_text = 0 # determine the position of the text
                    for key in junction_num:
                        ax.text(i, j + 0.5 - 0.2 * pos_text, str(key) + ":" + junction_num[key], size=9)
                        pos_text += 1

    for junction in junctions:
        ax.add_patch(patches.Rectangle(xy=junction, width=1, height=1, edgecolor="yellow", fill=False))
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
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                if MAP[i, j] == "S":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, color="red", fill=True))
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
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, color="red", fill=True))
                elif MAP[i, j] == "G":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))
                    # ax.text(i, j, config.p_end.index(calculate_number(i,j))+1)

    # plot paths
    for path in paths.values():
        row, line = [], []
        for i in path:
            row.append(i[0] + 0.5)
            line.append(i[1] + 0.5)
        plt.plot(row, line, color='red')
        ax.autoscale()
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
        for i in range(line1 + line_direction, line2, line_direction):
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

    elif line1 != line2:  # in the same raw, but not the same node
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1, line2, line_direction):
            nodes_middle.append(calculate_number(row1, i))
        nodes_middle.pop(0)  # needs improvement for speed
        return nodes_middle

    else:  # the same node, no middle nodes
        return []


def generate_all_middle_nodes(path):
    """generate all the middle nodes for a path
       input: key nodes
       tem: all the middle nodes"""
    path2 = []
    for i in range(len(path) - 1):
        path2 = path2 + [path[i]] + generate_middle_nodes(path[i], path[i + 1]) + [path[i + 1]]
    return path2


def target_segment_generation(path):
    """return a dictionary where targets are the key and intermediate nodes are values"""
    target_segement_dic = {}

    segment = []
    target_before = config.p_start
    for node in path[1:]:
        if node in config.p_end:
            target_segement_dic[(target_before, node)] = segment
            target_before = node
            segment = []
        else:
            segment.append(node)
    return target_segement_dic


def target_segment_restore(target_segment_dic):
    """return a complete path with targets and intermediate nodes
        input: a dictionary of targets and intermediate nodes
        tem: a complete path"""
    path_new = []
    path_new.append(config.p_start)
    for i in target_segment_dic.keys():
        path_new = path_new + target_segment_dic[i]
        path_new.append(i[1])
    return path_new


## multi targets_self work
def count_targets(pop_order):
    """count how many times a certain targets appear in the pop_order
        input: pop_order: [[a certain order of targets]]
        ouput: targets_count: {(targets):count}"""
    targets_count = {}
    for path_single in pop_order:
        for j in range(len(path_single) - 1):
            two_targets = (path_single[j], path_single[j + 1]) if path_single[j] < path_single[j + 1] else (
                path_single[j + 1], path_single[j])
            targets_count[two_targets] = targets_count[two_targets] + 1 if two_targets in targets_count else 1
    return targets_count


def initialize_order():
    """initialize the order of targets"""
    order = []
    for i in range(config.POPULATION_TSP):
        tem = random.sample(config.p_end, len(config.p_end))
        tem = [config.p_start] + tem
        order.append(tem)
    return order


def initialize_segments(pop_order, MAP):
    """"input: arrays of targets
        tem: dictionary: {(targets): [middle nodes]}"""
    target_segments = {}  # contain all the segments
    for path_single in pop_order:
        for j in range(len(path_single) - 1):
            two_targets = (path_single[j], path_single[j + 1]) if path_single[j] < path_single[j + 1] else (
                path_single[j + 1], path_single[j])
            if two_targets in target_segments:
                target_segments[two_targets].append(initialize_segment_single_random_selsection(two_targets[0],
                                                                                                two_targets[1], MAP))
            else:
                target_segments[two_targets] = [initialize_segment_single_random_selsection(two_targets[0],
                                                                                            two_targets[1], MAP)]
    return target_segments


def initialize_segment_single_random_walk(point1, point2, MAP):
    """initialize paths that encounter no obstacles"""
    while 1:
        path_session = [point1]
        while len(path_session) < config.NUM_NODE + 1:  # num_node is the number of middle nodes, +1 means the start
            if detect_obstacle(path_session[-1], point2, MAP):
                return path_session
            else:
                random_node = randint(0, config.MAP_SIZE * config.MAP_SIZE - 3)
                print(random_node)
                (plot_path_single([point1] + [random_node], MAP))
                if detect_obstacle((path_session[-1]), random_node, MAP):
                    path_session.append(random_node)
            print(path_session)


def initialize_segment_single_random_selsection(point1, point2, MAP):
    """initialize paths that encounter no obstacles"""
    while 1:
        path_session = [point1] + [randint(0, config.MAP_SIZE * config.MAP_SIZE - 1) for _ in
                                   range(config.NUM_NODE)] + [point2]
        flag = 1
        for i in range(len(path_session) - 1):
            if not detect_obstacle(path_session[i], path_session[i + 1], MAP):
                flag = 0
                break
        if flag:  # no obstacle
            return path_session


def initialize_segment_single_random_selsection_v2(point1, point2, MAP):
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


def count_list(a):
    """count components of a list
        input: list [[]]
        ouput: number of components in the list"""
    a_tuple = [tuple(i) for i in a]
    a_count = collections.Counter(a_tuple)
    return a_count


def target_segments_restore(path_single, fitness_segments, target_segments):
    """"return the best path with targets in a certain order
        input: path_single:[targets], fitness_segments:{[targets]:[fitness]}, target_segments:{[targets]:[[segment]]}
        tem: one single path:[]"""
    path_with_middle_nodes = []
    for j in range(len(path_single) - 1):
        if path_single[j] < path_single[j + 1]:
            two_targets = (path_single[j], path_single[j + 1])
            index = fitness_segments[two_targets].index(min(fitness_segments[two_targets]))
            path_with_middle_nodes = path_with_middle_nodes + target_segments[two_targets][index]
        else:
            two_targets = (path_single[j + 1], path_single[j])
            index = fitness_segments[two_targets].index(min(fitness_segments[two_targets]))
            middle_nodes = list(reversed(target_segments[two_targets][index]))
            path_with_middle_nodes = path_with_middle_nodes + middle_nodes
    return path_with_middle_nodes


def calculate_start_goal(MAP):
    """generate the start point and goals
        input: MAP
        tem: None(changes are made to config)"""
    goals = []
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] == "S":
                config.p_start = calculate_number(i, j)
            elif MAP[i][j] == "G":
                goals.append(calculate_number(i, j))
    config.p_end = goals

    # # to be deleted
    # if len(config.p_end) == 1:
    #     config.p_end = config.p_end[0]


def write_result(result):
    """ouput the average of result to an Excel file
        input: array: [[]]
        ouput: a file"""
    df = pd.DataFrame(result)
    df.to_excel("result_self_work_selection_change_v2.xlsx")


def plot_segments_num_all(target_counts_history, MAP):
    """"plot how many segments there are at each iteration during the search
        input: target_counts_history_average: [[{targets:num} * generation}] * test]
        ouput: a figure"""
    target_counts_history_average = segments_num_average(target_counts_history)
    FIG, AX = plt.subplots()
    for gen in range(config.ITERATION):
        plot_sements_num(target_counts_history_average[gen], MAP, AX, gen)


def segments_num_average(target_counts_history):
    """calculate how many segements exist on average at each iteration
        input: target_counts_history: [[target_counts_history_one_test]]
        tem: target_counts_history_average: [{targets:num} * generation}]"""
    target_counts_history_average = []
    for gen in range(config.ITERATION):  # each generation
        target_counts = {}
        for target_counts_history_one_test in target_counts_history:  # each test
            for two_targets in target_counts_history_one_test[gen].keys():  # sum the num
                if two_targets in target_counts:
                    target_counts[two_targets] += target_counts_history_one_test[gen][two_targets]
                else:
                    target_counts[two_targets] = target_counts_history_one_test[gen][two_targets]
        for two_targets in target_counts.keys():  # calculate the average
            target_counts[two_targets] /= config.TEST_NUM
        target_counts_history_average.append(target_counts)
    return target_counts_history_average


def plot_sements_num(target_counts, MAP, AX, gen):
    """plot how many segments are being found at this time
        input: count_list_dic {targets:number}
        tem: a figure"""
    # plot the MAP
    LENGTH = MAP.shape[0]
    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i, j] == 1:
                AX.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
            else:
                AX.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                if MAP[i, j] == "S":
                    AX.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))
                elif MAP[i, j] == "G":
                    AX.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))
    lines = []
    for i in target_counts.keys():
        x1, y1 = calculate_coordinate(i[0])
        x2, y2 = calculate_coordinate(i[1])
        line = plt.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], "-o",
                        lw=target_counts[i] / config.POPULATION_TSP * config.LINE_WIDTH, color="black")
        lines.append(line)
    plt.show(block=False)
    plt.title(f"iteration: {gen}")
    plt.pause(1)
    for line in lines:
        tem = line.pop(0)
        tem.remove()


# to be finished
def target_segments_excel_output(target_segments, map_name):
    """save initialized segments to an excel file for future use
        input: target segments {[targets]:[[segment]]}
        tem: an excel file. Each line is "targets: node node node" """
    tem = []
    for i in target_segments.keys():
        tem.append([i, target_segments[i]])

    df = pd.DataFrame(tem, columns=[0, 1])
    name = "initialized_segments_" + map_name + ".xlsx"
    df.to_excel(name)

    df = pd.read_excel(name)  ## the data type is str!!!!
    print(df)


def generate_potential_field(node_to_search, node_searched, map_potential, node_relation_map, junctions, MAP):
    """ generate potential field
        input: node_to_search, node_searched, map_potential, MAP
        reuturn: map_potential"""
    while len(node_to_search) > 0:
        pos_x, pos_y = node_to_search.popleft()
        for (i, j) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            # calculate the potential value for a new node
            pos_x_next, pos_y_next = pos_x + i, pos_y + j
            if 0 <= pos_x_next < config.MAP_SIZE and 0 <= pos_y_next < config.MAP_SIZE:
                if (pos_x_next, pos_y_next) not in node_searched and (pos_x_next,pos_y_next) not in node_to_search:
                    if MAP[pos_x_next, pos_y_next] == 1:
                        # obstacle
                        map_potential[pos_x_next][pos_y_next][0] = 1000
                    else:
                        node_relation_map[pos_x_next][pos_y_next] = (pos_x, pos_y)
                        map_potential[pos_x_next][pos_y_next][0] = map_potential[pos_x][pos_y][0] + 1
                        node_to_search.append((pos_x_next, pos_y_next))

                # finding a junction
                elif (pos_x_next, pos_y_next) in node_to_search and map_potential[pos_x_next][pos_y_next][0] >= map_potential[pos_x][pos_y][0]:
                    parent1 = node_relation_map[pos_x_next][pos_y_next]
                    parent2 = node_relation_map[pos_x][pos_y]
                    grandparent1 = node_relation_map[parent1[0]][parent1[1]]
                    grandparent2 = node_relation_map[parent2[0]][parent2[1]]

                    direction1 = (pos_x_next - parent1[0], pos_y_next - parent1[1])
                    direction2 = (pos_x - parent2[0], pos_y - parent2[1])
                    # an obstacle exists
                    if direction1[0] * direction2[0] + direction1[1] * direction2[1] < 0 or not detect_obstacle(
                            calculate_number(grandparent1[0],grandparent1[1]), calculate_number(grandparent2[0],grandparent2[1]), MAP):
                        junctions.add((pos_x_next, pos_y_next))
                        if (pos_x, pos_y) not in junctions:
                            junctions.add((pos_x, pos_y))
        node_searched.add((pos_x, pos_y))


def initialize_potential_map(MAP, target_num):
    """initialize the potential MAP.
        input: MAP, beginning point to decide the 0 potential value
        ouput: potential MAP"""
    # the potential value of the start point is 0
    if target_num == -1:
        coordinate_start = calculate_coordinate(config.p_start)
    # the potential value of the end point is 0
    else:
        coordinate_start = calculate_coordinate(config.p_end[target_num])
    map_potential = [[[1000, {}] for i in range(config.MAP_SIZE)] for j in range(config.MAP_SIZE)] # potential value and {junction_num: middle_node or junction}
    map_potential[coordinate_start[0]][coordinate_start[1]][0] = 0

    node_to_search = deque()
    node_to_search.append(tuple(coordinate_start))
    node_searched = set()
    node_relation_map = [[-100 for i in range(config.MAP_SIZE)] for j in range(config.MAP_SIZE)]
    node_relation_map[coordinate_start[0]][coordinate_start[1]] = coordinate_start

    junctions = set()
    generate_potential_field(node_to_search, node_searched, map_potential, node_relation_map, junctions, MAP)
    return map_potential, junctions

def reduce_reduent_nodes_initialization(path, MAP):
    """ reduce reduent nodes in the initialized path
        input: path
        ouput: path with reduced nodes"""
    index = 1
    path2 = [path[0]]
    while index < len(path):
        if detect_obstacle(calculate_number(path2[-1][0], path2[-1][1]), calculate_number(path[index][0], path[index][1]), MAP):
            index += 1
        else:
            path2.append((path[index-1]))
    path2.append(path[-1])
    return path2

def compatibility_new_expression(path):
    """change the expression of path node from number to (row, line)
        input: path:[number]
        tem: path: [(row, line)]"""
    path_new = []
    for node in path:
        path_new.append(calculate_coordinate(node))
    return path_new

def compatibility_old_GA(path):
    """change the expression of path node from (row, line) to number
        input: path: [(row, line)]
        tem: path:[number]"""
    path_new = []
    for node in path:
        path_new.append(calculate_number(node[0], node[1]))
    return path_new

def initialize_path_single(MAP, map_potential, target_num, target2):
    pos_x, pos_y = calculate_coordinate(config.p_end[target2])
    path = [(pos_x, pos_y)]
    search_around_list = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if target_num == -1:
        coordinate_p_start = tuple(calculate_coordinate(config.p_start))
    else:
        coordinate_p_start = tuple(calculate_coordinate(config.p_end[target_num]))
    flag_middle_node = False # flag to show whether this is a middle node
    flag_junction = False # flat to show whether this is a flag_junction
    junction_encountered = set() # junctions that have been encountered
    while((pos_x, pos_y) != coordinate_p_start):

        # if this is a normal node
        if not flag_middle_node and not flag_junction:
            potential_around = []
            middle_nodes_around = {}  # {junction_num: [node]}
            junction_encountered_tem = []
            # search around
            for (i, j) in search_around_list:
                pos_x_next, pos_y_next = pos_x + i, pos_y + j
                # whether in the MAP
                if 0 <= pos_x_next < config.MAP_SIZE and 0 <= pos_y_next < config.MAP_SIZE:
                    potential_around.append(map_potential[pos_x_next][pos_y_next][0])
                    # if there is a junction
                    if len(map_potential[pos_x_next][pos_y_next][1]) > 0:
                        for junction_num in map_potential[pos_x_next][pos_y_next][1]:
                            if junction_num in junction_encountered:
                                continue
                            junction_encountered_tem.append(junction_num)
                            if map_potential[pos_x_next][pos_y_next][1][junction_num] == "middle_node":
                                # add the middle node into dict junction
                                if junction_num not in middle_nodes_around:
                                    middle_nodes_around[junction_num] = []
                                middle_nodes_around[junction_num].append((pos_x_next, pos_y_next))
                else:
                    potential_around.append(float("Inf"))
                # add encountered junctions into junction_encountered
                for junction_num in junction_encountered_tem:
                    junction_encountered.add(junction_num)

            # determine the next node to go
            # if there are middle nodes
            if len(middle_nodes_around) > 0 and random.random() < config.POS_BRANCH:
                junction_num = random.choice(list(middle_nodes_around.keys()))
                pos_x, pos_y = random.choice(middle_nodes_around[junction_num])
                flag_middle_node, flag_junction = True, False
                path.append((pos_x, pos_y))

            # there are no middle nodes
            else:
                next_direction = search_around_list[choice([index for index, num in enumerate(potential_around) if num == min(potential_around)])]
                pos_x += next_direction[0]
                pos_y += next_direction[1]
                path.append((pos_x, pos_y))

                # sometimes may pass a junction

        # this is a middle node
        elif flag_middle_node:
            middle_node_around = []
            junction_around = []
            for (i, j) in search_around_list:
                pos_x_next, pos_y_next = pos_x + i, pos_y + j
                if 0 <= pos_x_next < config.MAP_SIZE and 0 <= pos_y_next < config.MAP_SIZE:
                    # find another middle node or junction node
                    if junction_num in map_potential[pos_x_next][pos_y_next][1] and map_potential[pos_x_next][pos_y_next][0] > map_potential[pos_x][pos_y][0]:
                        # the next node is a middle node
                        if map_potential[pos_x_next][pos_y_next][1][junction_num] == "middle_node":
                            middle_node_around.append((pos_x_next, pos_y_next))
                        # the next node is a junction
                        else:
                            junction_around.append((pos_x_next, pos_y_next))
            if len(junction_around) == 0:
                pos_x, pos_y = random.choice(middle_node_around)
            else:
                pos_x, pos_y = random.choice(junction_around)
                flag_middle_node, flag_junction = False, True
            path.append((pos_x, pos_y))

        # this is a junction
        elif flag_junction:
            """ middle nodes appeared after the junction
                a junction may be passed two times"""
            # plot_path_single(path, MAP, True)
            potential_around = []
            direction_around = [] # to record the direction of nearby nodes, the way to find the next node is different from usual
            junction_list = [] # to record the position of nearby junctions
            a = 1
            middle_nodes_around = {}  # {junction_num: [node]}
            # search around
            for (i, j) in search_around_list:
                pos_x_next, pos_y_next = pos_x + i, pos_y + j
                # in the MAP
                if 0 <= pos_x_next < config.MAP_SIZE and 0 <= pos_y_next < config.MAP_SIZE:
                    # a node that is not a middle node with a smaller potential value exists
                    if (junction_num not in map_potential[pos_x_next][pos_y_next][1] or map_potential[pos_x_next][pos_y_next][1][junction_num] != "middle_node") and map_potential[pos_x_next][pos_y_next][0] < map_potential[pos_x][pos_y][0]:
                        potential_around.append(map_potential[pos_x_next][pos_y_next][0])
                        direction_around.append((i,j))
                    # a junction node that has the same value
                    elif junction_num in map_potential[pos_x_next][pos_y_next][1] and map_potential[pos_x_next][pos_y_next][1][junction_num] == "junction"  and map_potential[pos_x_next][pos_y_next][0] == map_potential[pos_x][pos_y][0]:
                        junction_list.append((pos_x_next, pos_y_next))

            # if a node that is not a middle node with a smaller potential value exists
            if len(potential_around) > 0:
                next_direction = direction_around[choice([index for index, num in enumerate(potential_around) if num == min(potential_around)])]
                pos_x += next_direction[0]
                pos_y += next_direction[1]

            else:
                # search for a junction node that has a smaller potential value nearby
                for junction_node in random.sample(junction_list, len(junction_list)):
                    pos_x, pos_y = junction_node
                    junction_node_next = None
                    for (i,j) in search_around_list:
                        pos_x_next, pos_y_next = pos_x + i, pos_y + j
                        if 0 <= pos_x_next < config.MAP_SIZE and 0 <= pos_y_next < config.MAP_SIZE and (junction_num not in map_potential[pos_x_next][pos_y_next][1] or
                                map_potential[pos_x_next][pos_y_next][1][junction_num] != "middle_node") and map_potential[pos_x_next][pos_y_next][0] < map_potential[pos_x][pos_y][0]:
                                    junction_node_next = junction_node
                                    break
                    if junction_node_next is not None:
                        break
                pos_x, pos_y = junction_node_next
            path.append((pos_x, pos_y))
            flag_junction, flag_middle_node = False, False

        # # whether a junction has been passed
        # if len(map_potential[pos_x][pos_y][1]) > 0:
        #     for junction_num in map_potential[pos_x][pos_y][1].keys():
        #         if junction_num not in junction_passed:
        #             junction_passed.add(junction_num)

    path = reduce_reduent_nodes_initialization(path, MAP)
    return path

def junctions_group(junctions, map_potential1):
    """divide the junctions into sub groups, used in detect_division_point
        input: junctions:set()
        ouput:junctions_gouped: {junction_num: [junction,...]}"""
    direction_list = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    junctions_grouped = {} # {junction_num: [junction,...]}
    junction_num = 0 # the num of the group
    # there are nodes that have not been found
    while len(junctions) > 0:
        # get one node to search
        node_new = next(iter(junctions))
        node_to_search = deque([node_new])
        junction_num += 1
        junctions_grouped[junction_num] = [node_new]
        while len(node_to_search) > 0:
            node = node_to_search.popleft()
            pos_x, pos_y = node
            for direction in direction_list:
                pos_x_next, pos_y_next = pos_x + direction[0], pos_y + direction[1]
                if 0 <= pos_x_next < config.MAP_SIZE and 0 <= pos_y_next < config.MAP_SIZE:
                    # node of the junction
                    if (pos_x_next, pos_y_next) in junctions and (pos_x_next, pos_y_next) not in node_to_search:
                        node_to_search.append((pos_x_next, pos_y_next))
                        junctions_grouped[junction_num].append((pos_x_next, pos_y_next))
                        # tell the potential MAP that this is a junction
                        if junction_num not in map_potential1[pos_x_next][pos_y_next][1]:
                            map_potential1[pos_x_next][pos_y_next][1][junction_num] = "junction"
            # this node has been found
            junctions.remove((pos_x, pos_y))
    return junctions_grouped

def detect_division_point(map_potential1, map_potential2, junctions1):
    """ find the division point in the path from the end point to the start point
        input: map_potential1, map_potential2, junctions1, junctions2
        tem: map_potential1"""
    junctions_grouped1 = junctions_group(junctions1.copy(), map_potential1)
    direction_list = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for junction_num in junctions_grouped1.keys():
        node_to_search = deque(list(junctions_grouped1[junction_num]))
        # define the root of junction
        for node in node_to_search:
            pos_x, pos_y = node
            map_potential1[pos_x][pos_y][1][junction_num] = "junction"
        while len(node_to_search) > 0:
            node = node_to_search.popleft()
            pos_x, pos_y = node
            for direction in direction_list:
                pos_x_next, pos_y_next = pos_x + direction[0], pos_y + direction[1]
                # whether the found node is in the MAP
                if 0 <= pos_x_next < config.MAP_SIZE and 0 <= pos_y_next < config.MAP_SIZE:
                    # potential value of map1 and map2 decreases
                    """find the division_point"""
                    if map_potential1[pos_x_next][pos_y_next][0] < map_potential1[pos_x][pos_y][0] and map_potential2[pos_x_next][pos_y_next][0] < map_potential2[pos_x][pos_y][0] and junction_num not in map_potential1[pos_x_next][pos_y_next][1]:
                        map_potential1[pos_x_next][pos_y_next][1][junction_num] = "middle_node"
                        node_to_search.append((pos_x_next, pos_y_next))
    return map_potential1

def pop_order_to_path(pop_order, best_all):
    """generate the best path in a given order
        input: pop_order, best_all(the best path between two targets)
        ouput: path"""
    if config.p_start_TSP < pop_order[0]:
        two_targets = (config.p_start_TSP, pop_order[0])
        path = list(reversed(best_all[two_targets]))
    else:
        two_targets = (pop_order[0], config.p_start_TSP)
        path = best_all[two_targets]
    # path = []
    for index in range(len(pop_order) - 1):
        if pop_order[index] < pop_order[index + 1]:
            two_targets = (pop_order[index], pop_order[index + 1])
            path = path + list(reversed(best_all[two_targets]))
        else:
            two_targets = (pop_order[index + 1], pop_order[index])
            path = path + best_all[two_targets]
    return path


