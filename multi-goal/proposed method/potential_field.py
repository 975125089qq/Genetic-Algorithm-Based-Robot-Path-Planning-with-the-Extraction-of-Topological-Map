from basic_class import Node, UnionFind, Pftree, Conn
import random
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from functions import calculate_smooth
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import deque
import config
from collections import defaultdict

class Pf():
    """potential field
       Use tree structure and UnionFind to search the MAP and detect the loops"""

    def __init__(self, mapp, start=None, end_list=None):
        self.map = mapp
        self.size = len(mapp)
        self.pmap = [[Node(i, j) for j in range(self.size)] for i in range(self.size)]  # potential MAP
        self.direction = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        # initialize start (because of the generation of new paths in ACO, start can be None)
        if start is not None:
            self.start = self.pmap[start[0]][start[1]]
            self.start.value = 0
            self.start.value_all = 0
            self.start.start_id = -1
        if end_list is not None:
            self.end = [self.pmap[end[0]][end[1]] for end in end_list]
            for ind, node in enumerate(self.end): # the node is found by which starting point
                node.start_id = ind
                node.index = ind + 1
                node.value, node.value_all = 0, 0
        self.pftree = Pftree(self.size, self.map, self.pmap)
        self.loop_path_set = set() # the paths from loop to branches that have been generated
        self.conn = Conn()

        # the radius of circle(for plot)
        self.radius = 0.3

    def search_map(self):
        """search all the MAP and generate all paths"""
        node_list = [self.start] + self.end
        # assign the potential value to each node
        while len(node_list) > 0:
            node_list = self.search_one_step(node_list)
        return self.pmap, self.pftree.key_sect, self.conn

    def search_one_step(self, node_list):
        """search the MAP for one step and divide the current node_list into several neighbouring sections
           input: node_list[node]"""
        node_next_list = []
        union = UnionFind(len(node_list))
        cur_value = node_list[0].value  # current potential value
        index_count = 0  # give an index to each node for UnionFind
        # print(self.pmap[17][5].value_all)
        # self.plot_map(False, "value_all")
        for node in node_list:
            for dir in self.direction:
                pos_next = (node.x + dir[0], node.y + dir[1])
                # not in map
                if not self.in_map(pos_next):
                    continue
                node_next = self.pmap[pos_next[0]][pos_next[1]]
                # not searched
                if node_next.value == float("Inf"):
                    node_next.mother_node = node
                    node_next.value = cur_value + 1
                    node_next.index = index_count
                    index_count += 1
                    node_next_list.append(node_next)
                    calculate_smooth(cur_value, node_next, node)
                    node_next.start_id = node.start_id

                # searched by the same starting point
                elif node_next.start_id == node.start_id:
                    # node in node_list (node with same value)
                    if node_next.value == cur_value:
                        union.unite(node.index, node_next.index)
                        # loop (not from the same parent node group)
                        if not self.pftree.loop_detection(node.mother_node, node_next.mother_node, cur_value - 1):
                            node.loop_node.append(node_next)
                            node_next.loop_node.append(node)

                    # already been searched and is a loop
                    elif node_next.value == cur_value - 1:
                        if not self.pftree.loop_detection(node.mother_node, node_next, cur_value - 1):
                            node.loop_node2.append(node_next)

                # node searched by other starting point
                else:
                    if node_next.value == cur_value:
                        node.node_other_start.append(node_next)
                        node_next.node_other_start.append(node)
                    elif node_next.value == cur_value - 1:
                        node.node_other_start.append(node_next)


        self.pftree.add_section(union, node_list, self.conn)
        # print("value:", cur_value, "  num:", self.pftree.union_list[-1].group_size(), "len:", len(node_list), "group:",
        #       self.pftree.sections[-1].keys())
        return node_next_list

    def initialize_path_all(self):
        """initialize path for all pairs of targets"""
        path = defaultdict(list)
        for i in range(config.POPULATION):
            for two_targets in self.conn.conn_all.keys():
                node1, node2 = random.choice(self.conn.conn_all[two_targets])
                path1 = self.initialize_path(node1, self.end[node1.start_id] if node1.start_id!=-1 else self.start)
                path2 = self.initialize_path(node2, self.end[node2.start_id]if node2.start_id!=-1 else self.start)
                path_tem = self.reduce_reduent_nodes_initialization(path1[::-1] + path2)
                path[two_targets].append(path_tem)
                # self.plot_path_single(path[two_targets][0], False)
        return path

    def initialize_path(self, end, start, ACO_flag=False):
        if ACO_flag: # it is a special version used in ACO, when end is a tuple
            end_pos = config.p_end[end] if end != -1 else config.p_start
            start_pos = config.p_end[start] if start != -1 else config.p_start
            end, start = self.pmap[end_pos[0]][end_pos[1]], self.pmap[start_pos[0]][start_pos[1]]
        cur_section = self.section(end.value, end.index_section)
        node = end
        path = [node.pos]
        while node != start:
            # encounter a branching section
            if cur_section.loop_section is not None:
                next_section, node = self.path_branching(cur_section, cur_section.loop_section, node, path)

            # with a certain chance of trying a loop section with bigger value_all
            elif len(cur_section.loop_candi) > 0 and random.random() > 1/(len(cur_section.loop_candi)+1):
                loop_section = random.choice(cur_section.loop_candi)
                next_section, node = self.path_branching(cur_section, loop_section, node, path)

            # # encounter a loop section
            elif len(cur_section.parent) > 1:
                # print(cur_section.pos, cur_section.parent_section.pos, cur_section.value_all, len(cur_section) // 2, node.value_all)
                section_candi = [sect for sect, sec_value in cur_section.value_all_dict.items() if sec_value+len(cur_section)//2 <= cur_section.value_all*1.3]
                if len(section_candi) > 0:
                    sec_next = random.choice(section_candi)
                    next_section, node = self.path_from_loop(self.section(sec_next[0], sec_next[1]), node, path)
                else:
                    node = self.path_one_step(node, path)
                    next_section = self.section(*node.tree_pos)

            # just move forward
            else:
                for _ in range(cur_section.depth - cur_section.parent[0].depth):
                    node = self.path_one_step(node, path)
                next_section = cur_section.parent[0]

            # record section that has been visited
            cur_section = next_section

        # self.plot_path_single(path, False)
        return path

    def path_branching(self, cur_section, loop_section, node, path):
        """initialize path from a branching section to another branching section via a loop section"""
        # generate path from the loop section to cur_section
        if (cur_section.pos, loop_section.pos) not in self.loop_path_set:
            self.loop_path_set.add((cur_section.pos, loop_section.pos))
            self.path_loop_branching(cur_section, loop_section)

        # if the current node does not lead to the loop section
        # dfs, find out the node in this section that leads to the loop section
        node_searched = set()
        _, node= self.dfs_path_branching(node, loop_section, path, node_searched)

        # move to the loop_section
        for _ in range(loop_section.depth - cur_section.depth):
            around = []
            x, y = node.pos
            for i in self.direction:
                pos = (x + i[0], y + i[1])
                if self.in_map(pos) and self.pmap[pos[0]][pos[1]].value == node.value + 1 and loop_section.pos in \
                        self.pmap[pos[0]][pos[1]].middle_path and self.same_start(pos, node):
                    around.append(pos)
            node_pos = random.choice(around)
            node = self.pmap[node_pos[0]][node_pos[1]]
            path.append(node_pos)
        cur_section = loop_section

        # move to the branching section
        branching_section = cur_section.parent_section
        cur_section, node = self.path_from_loop(branching_section, node, path)
        return cur_section, node

    def path_from_loop(self, branching_section, node, path):
        """generate the path from a loop section"""
        # choose the branching section and move towards it by one step
        # dfs, find out the node that leads to the branching section
        node_searched = set()
        path.pop() # the node will be added again in the DFS
        return self.dfs_path_from_loop(node, path, branching_section, node_searched)

    def dfs_path_from_loop(self, node, path, branching_section, node_searched):
        """move among the section, use DFS to find out the node connected to designated section"""
        node_searched.add(node.pos)
        path.append(node.pos)
        pos1 = (node.mother_node.value, node.mother_node.index_section)  # the position of mother node
        section_next = self.section(pos1[0], pos1[1])
        if branching_section == section_next or (
                len(section_next.parent) == 1 and branching_section == section_next.parent[0]):
            path.append(node.mother_node.pos)
            node = node.mother_node
            cur_section = section_next
            return cur_section, node
        # adjacent nodes with smaller value but are not mother nodes
        elif len(node.loop_node2) > 0:
            for i in node.loop_node2:
                section_next = self.section(i.value, i.index_section)
                if branching_section == section_next or (
                        len(section_next.parent) == 1 and branching_section == section_next.parent[0]):
                    path.append(i.pos)
                    node = i
                    cur_section = section_next
                    return cur_section, node
        for i in self.direction:
            x_next, y_next = node.x + i[0], node.y + i[1]
            if not self.in_map((x_next, y_next)) or self.pmap[x_next][y_next].value != node.value or (x_next, y_next) in node_searched or not self.same_start((x_next, y_next), node): # already searched
                continue
            node_next = self.pmap[x_next][y_next]
            result = self.dfs_path_from_loop(node_next, path, branching_section, node_searched)
            if result != False: # has been found
                return result
        path.pop()
        return False

    def dfs_path_branching(self, node, loop_section, path, node_searched):
        """the dfs part used in the path branching function to move in one section"""
        node_searched.add(node.pos)
        if loop_section.pos in node.middle_path:
            return True, node
        else:
            for i in self.direction:
                x_next, y_next = node.x + i[0], node.y + i[1]
                if (x_next, y_next) in node_searched or not (self.in_map((x_next, y_next))) or not self.same_start((x_next, y_next), node):
                    continue
                node_next = self.pmap[x_next][y_next]
                path.append(node_next.pos)
                res = self.dfs_path_branching(node_next, loop_section, path, node_searched)
                if res[0]:  # found
                    return res
                path.pop()
            return [False]

    def path_loop_branching(self, cur_section, loop_section):
        """generate path from loop section to branching section (current section)
           used when no path exist in self.path_loop_set"""
        # loop_section = cur_section.loop_section

        # generate the path from the loop to the branching loop_section
        for node in loop_section:
            # repeat finding the parent node until reaching the branching loop_section
            flag = False # whether there is a need to repeatedly find out the mother node
            section_next = self.section(*node.mother_node.tree_pos)
            if section_next.pos == cur_section.pos or section_next.parent[0].pos == cur_section.pos:
                node.middle_path.add(loop_section.pos)
                # sometimes a mother node can be visited several times
                if loop_section.pos not in node.mother_node.middle_path:
                    flag = True
                    node = node.mother_node
            else:
                for node_next in node.loop_node2:
                    section_next = self.section(*node_next.tree_pos)
                    if section_next.pos == cur_section.pos or section_next.parent[0].pos == cur_section.pos:
                        node.middle_path.add(loop_section.pos)
                        if loop_section.pos not in node_next.middle_path:
                            flag = True
                            node = node_next
                        break

            # this node is connected to the branching loop_section
            if flag:
                while node.tree_pos != cur_section.pos and loop_section.pos not in node.middle_path:
                    node.middle_path.add(loop_section.pos)
                    node = node.mother_node
                node.middle_path.add(loop_section.pos)


    def path_one_step(self, node, path):
        """just move one section forward"""
        # move one depth
        x, y = node.pos
        cur_value = node.value
        around = []
        for i in self.direction:
            pos = (x + i[0], y + i[1])
            if self.in_map(pos) and self.pmap[pos[0]][pos[1]].value < cur_value and self.same_start(pos, node):
                around.append(pos)
        node_pos = random.choice(around)
        node = self.pmap[node_pos[0]][node_pos[1]]
        path.append(node_pos)
        return node

    def reduce_reduent_nodes_initialization(self, path):
        """ reduce reduent nodes in the initialized path
            input: path
            ouput: path with reduced nodes"""
        index = 1
        path2 = [path[0]]
        while index < len(path):
            if self.detect_obstacle(path2[-1],path[index]):
                index += 1
            else:
                path2.append((path[index - 1]))
        path2.append(path[-1])
        return path2

    def plot_map(self, block=True, type="value"):
        """plot one path"""
        fig, ax = plt.subplots()
        for i in range(self.size):
            for j in range(self.size):
                if self.pmap[i][j].value == float("Inf"):
                    # ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=True))
                    ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, fill=True, color="blue"))
                else:
                    # ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black", fill=False))
                    # ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, fill=True, color="white"))
                    if type == "value":
                        ax.text(i, j + 0.7, str(self.pmap[i][j].tree_pos)+str(self.pmap[i][j].start_id))
                    else:
                        ax.text(i, j + 0.7, str(self.pmap[i][j].value_all))
                    if len(self.pmap[i][j].middle_path) > 0:
                        ax.text(i, j + 0.5, str(self.pmap[i][j].middle_path))
                    if self.map[i, j] == "S":
                        ax.add_patch(
                            patches.Circle(xy=(i + 0.5, j + 0.5), radius=self.radius, edgecolor="black", fill=True))
                    elif self.map[i, j] == "G":
                        ax.add_patch(
                            patches.Circle(xy=(i + 0.5, j + 0.5), radius=self.radius, edgecolor="black", fill=True))
        ax.autoscale()
        plt.subplots_adjust(left=0.04, right=1, bottom=0.04, top=1)
        plt.show(block=block)

    def plot_path_single(self, path, block=True):
        """plot one path"""
        fig, ax = plt.subplots()

        for i in range(self.size):
            for j in range(self.size):
                if self.pmap[i][j].value == float("Inf"):
                    ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1,color ="blue", fill=True))
                else:
                    ax.add_patch(patches.Rectangle(xy=(i, j), width=1, height=1, edgecolor="black",  fill=False))
                    if self.map[i, j] == "S":
                        ax.add_patch(
                            patches.Circle(xy=(i + 0.5, j + 0.5), radius=self.radius, edgecolor="black", fill=True))
                    elif self.map[i, j] == "G":
                        ax.add_patch(
                            patches.Circle(xy=(i + 0.5, j + 0.5), radius=self.radius, edgecolor="black", fill=True))

        row, line = [], []
        for i in path:
            row.append(i[0] + 0.5)
            line.append(i[1] + 0.5)
        plt.plot(row, line, color='red')
        ax.autoscale()
        plt.show(block=block)

    def plot_path_single_light(self, path, block=True):
        plt.figure()
        self.map[config.p_start[0]][config.p_start[1]] = 0
        self.map[config.p_end[0]][config.p_end[1]] = 0
        self.map = self.map.astype("int")
        # Invert the color mapping: 1 to black and 0 to white
        cmap = plt.get_cmap("gray_r")
        plt.imshow(self.map, cmap=cmap)
        plt.plot([i[1] for i in path], [i[0] for i in path], "r-", linewidth=3)
        plt.show(block=block)

    def print_structure(self):
        """print out the structure of the pftree"""
        for i in self.pftree.sections:
            for section in i.values():
                print((section.depth, section.index), section.pass_flag, section.branching_section,
                      section.loop_section, end=" ")
                if len(section.parent) > 0:
                    print([(section.parent[i].depth, section.parent[i].index) for i in range(len(section.parent))],
                          end="/    ")

            print()

    def draw_structure(self, block=True):
        """draw the structure of the pftree"""
        G = nx.MultiDiGraph()
        section_searched = set()
        end_section_pos = {i.tree_pos for i in self.end}
        # bfs, search for sections
        section_search = deque([self.section(*self.start.tree_pos)]+[self.section(*i) for i in end_section_pos])
        G.add_node(self.start.tree_pos, )
        rn_colmap = {}
        node_labels = {}
        while len(section_search) > 0:
            section = section_search.popleft()
            if section.pos in section_searched:
                continue

            # different colors and labels for starting session and end session
            if section.pos == self.start.tree_pos or section.pos in end_section_pos:
                rn_colmap[section.pos] = "darkorange"
            elif len(section.parent) > 1:
                # print(section.pos, [section.parent[i].pos for i in range(len(section.parent))])
                rn_colmap[section.pos] = "red"
            else:
                rn_colmap[section.pos] = "lightblue"
            node_labels[section.pos] = [section.pos]

            for child in section.child:
                child = self.section(*child)
                G.add_edge(section.pos, child.pos)
                section_search.append(child)
            section_searched.add(section.pos)
        pos = graphviz_layout(G, prog="dot")  # 描画レイアウトの指定 Graphvizの"dot"というレイアウトを使用
        # for node, (x,y) in pos.items():
        #     pos[node] = (x, -200 * node[0])
        nx.draw_networkx(G, pos, with_labels=True, labels=node_labels, node_color=[rn_colmap[i] for i in pos], node_size=200, font_size=12)
        plt.show(block=block)


    def in_map(self, pos):
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size and self.map[pos[0]][pos[1]] != 1

    def same_start(self, pos1, node2):
        """True: same start, False: different start"""
        return self.pmap[pos1[0]][pos1[1]].start_id == node2.start_id

    def section(self, depth, section_index):
        """return the section"""
        return self.pftree.sections[depth][section_index]

    # find out if two nodes can be connected directly
    def detect_obstacle(self, node1, node2):
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
                        if self.map[row_current][j] == 1:
                            return False
            else:
                for i in range(len(rows) - 1):
                    line_1_int = int(lines[i] - EPS)
                    line_2_int = int(lines[i + 1] + EPS)
                    row_current = int(rows[i] + EPS)
                    for j in range(line_1_int, line_2_int - 1, -1):
                        if self.map[row_current][j] == 1:
                            return False
        else:
            line_direction = 1 if line1 < line2 else -1
            for i in range(line1, line2 + line_direction, line_direction):
                if self.map[row1][i] == 1:
                    return False
        return True

    def search_map_single(self,target, target_connected, pmap):
        """search the map for one pair of targets, used in the ACO part"""
        node_pos = config.p_end[target] if target != -1 else config.p_start
        node = self.pmap[node_pos[0]][node_pos[1]]
        node.index, node.value, node.value_all = 0, 0, 0
        node_list = [node]
        # assign the potential value to each node
        while len(node_list) > 0:
            node_list = self.search_one_step_single(node_list, target_connected, pmap)
        return self.pmap, self.pftree.key_sect, self.conn

    def search_one_step_single(self, node_list, target_connected, pmap):
        """search the MAP for one step and divide the current node_list into several neighbouring sections
           used in the ACO part
           input: node_list[node]"""
        node_next_list = []
        union = UnionFind(len(node_list))
        cur_value = node_list[0].value  # current potential value
        index_count = 0  # give an index to each node for UnionFind
        for node in node_list:
            for dir in self.direction:
                pos_next = (node.x + dir[0], node.y + dir[1])
                # not in map or the area found by target_connected
                if not self.in_map(pos_next):
                    continue
                start_id_check = pmap[pos_next[0]][pos_next[1]].start_id
                if start_id_check not in target_connected:
                    continue

                node_next = self.pmap[pos_next[0]][pos_next[1]]
                # not searched
                if node_next.value == float("Inf"):
                    node_next.mother_node = node
                    node_next.value = cur_value + 1
                    node_next.index = index_count
                    index_count += 1
                    node_next_list.append(node_next)
                    calculate_smooth(cur_value, node_next, node)
                    node_next.start_id = node.start_id

                # searched by the same starting point
                elif node_next.start_id == node.start_id:
                    # node in node_list (node with same value)
                    if node_next.value == cur_value:
                        union.unite(node.index, node_next.index)
                        # loop (not from the same parent node group)
                        if not self.pftree.loop_detection(node.mother_node, node_next.mother_node, cur_value - 1):
                            node.loop_node.append(node_next)
                            node_next.loop_node.append(node)

                    # already been searched and is a loop
                    elif node_next.value == cur_value - 1:
                        if not self.pftree.loop_detection(node.mother_node, node_next, cur_value - 1):
                            node.loop_node2.append(node_next)


        self.pftree.add_section(union, node_list, self.conn)
        # print("value:", cur_value, "  num:", self.pftree.union_list[-1].group_size(), "len:", len(node_list), "group:",
        #       self.pftree.sections[-1].keys())
        return node_next_list
