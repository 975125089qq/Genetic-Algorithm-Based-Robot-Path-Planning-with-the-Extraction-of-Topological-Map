"""use fitness rather than distance to decide which node to expand
   However, this program cannot find the optimal path, and is not true Theta*(smooth)."""
import pandas as pd
import math
from common.functions import *
from common.evaluation import *
from common.class_node import Node_theta
from common import config
from matplotlib import pyplot as plt
import heapq
import time

# load map
MAP = load_map("6", Node_theta)

time_start = time.time()
map_width = max([len(x) for x in MAP])
map_height = len(MAP)

# OpenリストとCloseリストを設定
map_node = [[Node_theta(j, i) for i in range(len(MAP))] for j in range(len(MAP))]
close_list = set()
start_node = map_node[Node_theta.start[0]][Node_theta.start[1]]
start_node.parent_node = start_node
start_node.fs = start_node.hs
open_list = [(0, start_node.pos)]
heapq.heapify(open_list)
open_list_set = {start_node.pos}

while True:
    # Openリストが空になったら解なし
    if open_list == []:
        print("There is no route until reaching a goal.")
        exit(1);

    # Openリストからf*が最少のノードnを取得
    d, n_pos = heapq.heappop(open_list)  # distance and node
    if n_pos in close_list:
        continue
    n = map_node[n_pos[0]][n_pos[1]]
    open_list_set.remove(n.pos)
    close_list.add(n.pos)

    # f*() = g*() + h*() -> g*() = f*() - h*()
    n_gs = n.fs - n.hs
    par = n.parent_node
    par_gs = par.fs - par.hs

    # ノードnの移動可能方向のノードを調べる
    for v in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
        x = n.pos[0] + v[0]
        y = n.pos[1] + v[1]

        # マップが範囲外または壁(O)の場合はcontinue
        if not (0 <= y < map_height and
                0 <= x < map_width and
                MAP[y][x] != 1):  # the MAP transposed
            continue

        if (x, y) not in close_list:
            m = map_node[x][y]
            if detect_obstacle((par.pos[1], par.pos[0]), (m.pos[1], m.pos[0]), MAP):
                # path2
                dist = math.sqrt((par.pos[0] - x) ** 2 + (par.pos[1] - y) ** 2)
                vec1 = [(x - par.pos[0]), (y - par.pos[1])]
                vec2 = [(par.pos[0] - par.parent_node.pos[0]), (par.pos[1] - par.parent_node.pos[1])]
                smooth = cal_smo(vec1, vec2)
                if m.fs > par_gs + m.hs + dist + smooth:
                    m.fs = par_gs + m.hs + dist + smooth
                    m.parent_node = par
                    heapq.heappush(open_list, (m.fs, m.pos))
                    open_list_set.add((x, y))
            else:
                # path1
                dist = math.sqrt((n.pos[0] - x) ** 2 + (n.pos[1] - y) ** 2)
                vec1 = [(x - n.pos[0]), (y - n.pos[1])]
                vec2 = [(n.pos[0] - n.parent_node.pos[0]), (n.pos[1] - n.parent_node.pos[1])]
                smooth = cal_smo(vec1, vec2)
                if m.fs > n_gs + m.hs + dist + smooth:
                    m.fs = n_gs + m.hs + dist + smooth
                    m.parent_node = n
                    heapq.heappush(open_list, (m.fs, m.pos))
                    open_list_set.add((x, y))

    # 最小ノードがゴールだったら終了
    if n.isGoal():
        end_node = n
        break

# endノードから親を辿っていくと、最短ルートを示す
path = [(n.pos[1], n.pos[0])]
while True:
    if n.parent_node == start_node:
        path.append((n.pos[1], n.pos[0]))
        path.append((Node_theta.start[1], Node_theta.start[0]))
        break
    path.append((n.pos[1], n.pos[0]))
    n = n.parent_node
print("time", time.time() - time_start)
print(evaluation(path, "fitness"))
# plot_points_searched(close_list, MAP, map_node, mode="value")
print(path)
plot_path_single_light(path, MAP, Node_theta.start, Node_theta.goal, title="theta*")
