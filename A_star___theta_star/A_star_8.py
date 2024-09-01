import pandas as pd
import math
from common.functions import plot_path_single, calculate_start_goal, plot_points_searched, \
    reduce_reduent_nodes_initialization, enlarge_map, plot_path_single_light, load_map
from common.evaluation import evaluation, cal_smo
import matplotlib.pyplot as plt
import heapq
import numpy as np
from common import config
import os
import scipy
import time
from common.class_node import Node_A

# load map
MAP = load_map("6", Node_A)

time_start = time.time()
map_width = max([len(x) for x in MAP])
map_height = len(MAP)
direction = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))

# OpenリストとCloseリストを設定
map_node = [[Node_A(j, i) for i in range(len(MAP))] for j in range(len(MAP))]
start_node = map_node[Node_A.start[0]][Node_A.start[1]]
start_node.parent_node = [start_node for _ in range(8)]
start_node.fs = [start_node.hs] * 8

close_list = set()
open_list = [(start_node.fs[i], start_node.pos, i) for i in range(8)]
heapq.heapify(open_list)
index2 = 0
while True:
    # Openリストが空になったら解なし
    if open_list == []:
        print("There is no route until reaching a goal.")
        exit(1);
    index2 += 1
    # Openリストからf*が最少のノードnと方向を取得
    _, n_pos, dir_index = heapq.heappop(open_list)  # fitness, node and the direction
    if (n_pos, dir_index) in close_list:
        continue
    n = map_node[n_pos[0]][n_pos[1]]
    par_pos = n.parent_node[dir_index].pos
    close_list.add((n.pos, dir_index))

    # 最小ノードがゴールだったら終了
    if n.isGoal():
        end_node = n
        # print("index", index2)
        # print(len(open_list))
        break

    # f*() = g*() + h*() -> g*() = f*() - h*()
    n_gs = n.fs[dir_index] - n.hs

    # move towards the next node (the direction is decided)
    x = n.pos[0] + direction[dir_index][0]
    y = n.pos[1] + direction[dir_index][1]

    # マップが範囲外または壁(O)の場合はcontinue
    if not (0 <= y < map_height and
            0 <= x < map_width and
            MAP[y][x] != 1):
        continue

    dist = math.sqrt((n.pos[0] - x) ** 2 + (n.pos[1] - y) ** 2) * config.SCALE_FACTOR
    m = map_node[x][y]
    # search for all the possible directions
    for index, v in enumerate(direction):
        if ((x, y), index) not in close_list:
            cost_direction = cal_smo(v, direction[dir_index])
            if m.fs[index] > n_gs + m.hs + dist + cost_direction:
                m.fs[index] = n_gs + m.hs + dist + cost_direction
                m.parent_node[index] = n
                heapq.heappush(open_list, (m.fs[index], m.pos, index))

# endノードから親を辿っていくと、最短ルートを示す
path = [(n.pos[1], n.pos[0])]
while True:
    path.append((n.pos[1], n.pos[0]))
    index = n.fs.index(min(n.fs))
    n = n.parent_node[index]
    if n == start_node:
        path.append((n.pos[1], n.pos[0]))
        break

# plot_path_single_light(path, MAP, Node_A.start, Node_A.goal, title="A*8_before_smoothing", block=False)
path = reduce_reduent_nodes_initialization(path, MAP)
print("time:", time.time() - time_start)
print(evaluation(path, "fitness"))
# plot_points_searched(close_list, MAP, map_node, True, "value")
print(path)
plot_path_single_light(path, MAP, Node_A.start, Node_A.goal, title="A*8")
