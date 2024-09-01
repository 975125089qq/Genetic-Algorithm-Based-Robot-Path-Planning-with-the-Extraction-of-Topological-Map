import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import config


def plot_potential_map(map_potential, junctions, block=True):
    """plot one path"""
    fig, ax = plt.subplots()
    for i in range(config.MAP_SIZE):
        for j in range(config.MAP_SIZE):
            if map_potential[i][j].potential_value == 1000:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, color="black", fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, color="black", fill=False))
                ax.text(i, j + 0.7, str(map_potential[i][j].potential_value))
                junction_num = (map_potential[i][j].junction)
                pos_text = 0  # determine the position of the text
                for key in junction_num:
                    if key == "tem":
                        continue
                    # ax.text(i, j + 0.5 - 0.2 * pos_text, str(key) + ":" + junction_num[key], size=9)
                    ax.text(i, j + 0.5 - 0.3 * pos_text, str(key) + ":" + str(map_potential[i][j].junction_type(key)),
                            size=9)
                    pos_text += 1

    for junction in junctions:
        ax.add_patch(patches.Rectangle(xy=junction, width=1, height=1, edgecolor="yellow", fill=False))
    ax.autoscale()
    plt.subplots_adjust(left=0.04, right=1, bottom=0.04, top=1)
    plt.show(block=block)


def plot_path_single(path, MAP, block=True):
    """plot one path"""
    LENGTH = len(MAP)
    fig, ax = plt.subplots()

    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i][j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, color="black", fill=True))
                # ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, color="black", fill=False))
                # ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, fill=True, color="white"))
                if MAP[i][j] == "S":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, edgecolor="black", fill=True))
                elif MAP[i][j] == "G":
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
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, color="black", fill=True))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, color="black", fill=False))
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


### be cautious ###
### to be tested
def detect_obstacle_plus(node1, node2, MAP):
    """True: no obstacle
    False: obstacle in the segment
    intermediate nodes will also be returned"""

    row1, line1 = node1
    row2, line2 = node2
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
                    nodes_middle.append((row_current, j))
                    if MAP[row_current][j]:
                        return False, None
        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    nodes_middle.append((row_current, j))
                    if MAP[row_current][j]:
                        return False, None
        nodes_middle.pop(-1)
        nodes_middle.pop(0)
        return True, nodes_middle

    elif line1 != line2:
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1 + line_direction, line2, line_direction):
            nodes_middle.append((row1, i))
            if MAP[row1][i]:
                return False, None
        return True, nodes_middle
    else:
        return True, []


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


def generate_all_middle_nodes(path):
    """generate all the middle nodes for a path
       input: key nodes
       tem: all the middle nodes"""
    path2 = []
    for i in range(len(path) - 1):
        path2 = path2 + [path[i]] + generate_middle_nodes(path[i], path[i + 1]) + [path[i + 1]]
    return path2


def calculate_start_goal(MAP):
    """generate the start point and goals
        input: MAP """
    goals = []
    for i in range(len(MAP)):
        for j in range(len(MAP)):
            if MAP[i][j] == "S":
                config.p_start = (i, j)
            elif MAP[i][j] == "G":
                goals.append((i, j))
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
    smoothness = 0 # 0 degrees
    tem = np.dot(pos_re1, pos_re2)
    if tem > 0:
        if not(pos_re1[0] == pos_re2[0] and pos_re1[1] == pos_re2[1]):
            smoothness = 3  # 45 degrees
    elif tem == 0:
        smoothness = 9  # 90 degrees
    else:
        smoothness = 12 # 135 degrees
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

def plot_paths_same_map(paths, MAP, block=True):
    """plot multiple paths on the same MAP"""
    LENGTH = MAP.shape[0]
    fig, ax = plt.subplots()

    # plot the MAP
    for i in range(LENGTH):
        for j in range(LENGTH):
            if MAP[i, j] == 1:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True, facecolor="black"))
            else:
                ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False, facecolor="black"))
                if MAP[i, j] == "S":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, fill=True, edgecolor="black"))
                elif MAP[i, j] == "G":
                    ax.add_patch(
                        patches.Circle(xy=(i + 0.5, j + 0.5), radius=config.radius, fill=True, edgecolor="black"))
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

def plot_paths_same_map_light(paths, MAP, block=True):
    """plot multiple paths on the same MAP"""
    plt.figure()
    # turn targets into passable nodes
    for i in range(len(MAP)):
        for j in range(len(MAP[i])):
            if MAP[i][j] != 0 and MAP[i][j] != 1:
                MAP[i][j] = 0
    MAP = MAP.astype("int")
    # Invert the color mapping: 1 to black and 0 to white
    cmap = plt.get_cmap("gray_r")
    plt.imshow(MAP, cmap=cmap)

    # plot paths
    for path in paths.values():
        row, line = [], []
        for i in path:
            row.append(i[1])
            line.append(i[0])
        plt.plot(row, line, color='red')
    plt.show(block=block)

def pop_order_to_path(pop_order, best_all):
    """generate the best path in a given order
        input: pop_order, best_all(the best path between two targets)
        ouput: path"""
    two_targets = (config.p_start_id, pop_order[0])
    if two_targets in best_all:
        path = best_all[two_targets]
    else:
        path = [config.p_end[two_targets[0]], config.p_end[two_targets[1]]]

    # path = []
    for index in range(len(pop_order) - 1):
        if pop_order[index] < pop_order[index + 1]:
            two_targets = (pop_order[index], pop_order[index + 1])
            if two_targets in best_all:
                path = path + best_all[two_targets]
            else:
                path = path + [config.p_end[two_targets[0]], config.p_end[two_targets[1]]]
        else:
            two_targets = (pop_order[index + 1], pop_order[index])
            if two_targets in best_all:
                path = path + list(reversed(best_all[two_targets]))
            else:
                path = path + [config.p_end[two_targets[0]], config.p_end[two_targets[1]]]
    return path

def duplicate_targets_deletion(fitness, best_all, score_all, cost_matrix, target_order, path_planning, map_potential, MAP):
    """find duplicate targets and delete them from the path
       output: target_order_new"""
    target_appeared = {}
    duplicate_target = {}
    for index, target_num in enumerate(target_order):
        if target_num not in target_appeared:
            target_appeared[target_num] = index
        elif target_num not in duplicate_target:
            duplicate_target[target_num] = [target_appeared[target_num]]
            duplicate_target[target_num].append(index)
        else:
            duplicate_target[target_num].append(index)

    index_deleted_list = []
    for target_num in duplicate_target:
        index_list = duplicate_target[target_num]
        # calculate the benefits of deleting one target
        cost_list = []
        for index in index_list:
            if index != 0:
                target_before = target_order[index-1]
                target_after = target_order[index+1]
            else:
                target_before = -1
                target_after = target_order[index+1]
            two_targets = (min(target_before, target_after), max(target_before, target_after))
            path_deleted_benefit = cost_matrix[target_before+1][target_num+1] + cost_matrix[target_num+1][target_after+1]
            if (target_before, target_after) not in best_all:
                path_added, path_added_cost = path_planning(target_before, target_num, target_after, map_potential, MAP)
                best_all[two_targets] = path_added
                score_all[two_targets] = path_added_cost
            else:
                path_added_cost = score_all[two_targets]

            cost = path_added_cost - path_deleted_benefit
            cost_list.append(cost)

        cost_max = max(cost_list)
        reserve_index = cost_list.index(cost_max)
        index_deleted_list = index_deleted_list + index_list[:reserve_index] + index_list[reserve_index+1:]
        for index, cost in enumerate(cost_list):
            if index == reserve_index:
                continue
            else:
                fitness += cost


    index_deleted_list.sort(reverse=True)

    for index in index_deleted_list:
        target_order.pop(index)

    return target_order, fitness, best_all, score_all

def targets_connect_table_generation(connect_dict)->list:
    """record the information of generation into a list"""
    targets_connect_table = [[] for _ in range(len(config.p_end) + 1)]
    for key in connect_dict:
        node1, node2 = key
        targets_connect_table[node1+1].append(node2)
        targets_connect_table[node2+1].append(node1)
    return targets_connect_table