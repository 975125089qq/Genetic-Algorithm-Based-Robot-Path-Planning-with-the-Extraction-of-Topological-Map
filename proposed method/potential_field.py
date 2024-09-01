from basic_class import Node, UnionFind, Pftree
import random
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from functions import calculate_smooth
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import deque
import config
from collections import defaultdict
import numpy as np


class Pf():
    """potential field
       Use tree structure and UnionFind to search the MAP and detect the loops"""

    def __init__(self, mapp, start, end, thresh_loop=1, thresh_parent=1.35):
        self.map = mapp
        self.size = len(mapp)
        self.pmap = [[Node(i, j) for j in range(self.size)] for i in range(self.size)]  # potential MAP
        self.direction = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        self.start = self.pmap[start[0]][start[1]]
        self.start.value = 0
        self.start.value_all = 0
        self.end = self.pmap[end[0]][end[1]]
        self.pftree = Pftree(self.end, self.size, self.map, self.pmap)
        self.loop_path_set = set()  # the paths from loop to branches that have been generated
        self.sect_chosen = defaultdict(
            set)  # {(sect.pos):set(sect2.pos)} record the choices at sections to diversify the initialized population
        self.child_sect_check_set = set()  # sections that have calculated loop sections
        self.child_sect_check_set2 = set() # sections that have checked whether the child section will go back
        self.parent_sect_check_set = set()  # sections that have calculated the candidates for parent sections
        self.sect_in_path = set()  # sections that have been visited
        self.radius = 0.3  # the radius of circle(for plot)
        self.thresh_loop = thresh_loop  # for choosing loop sections. value(loop)/value(current) <= thresh_loop
        self.thresh_parent = thresh_parent  # for choosing parent sections, similar to thresh_loop

    def search_map(self):
        """search all the MAP and generate all paths"""
        node_list = [self.start]
        # assign the potential value to each node
        while len(node_list) > 0:
            node_list = self.search_one_step(node_list)
        return self.pmap

    def search_one_step(self, node_list):
        """search the MAP for one step and divide the current node_list into several neighbouring sections
           input: node_list[node]"""
        node_next_list = []
        union = UnionFind(len(node_list))
        cur_value = node_list[0].value  # current potential value
        index_count = 0  # give an index to each node for UnionFind
        for node in node_list:
            for dir in self.direction:
                pos_next = (node.x + dir[0], node.y + dir[1])
                # not in map
                if not self.in_map(pos_next):
                    continue
                node_next = self.pmap[pos_next[0]][pos_next[1]]
                # node_list
                if node_next.value == cur_value:
                    union.unite(node.index, node_next.index)
                    # loop
                    if not self.pftree.loop_detection(node.mother_node, node_next.mother_node, cur_value - 1):
                        node.loop_node.append(node_next)
                        node_next.loop_node.append(node)

                # not searched
                elif node_next.value == float("Inf"):
                    node_next.mother_node = node
                    node_next.value = cur_value + 1
                    node_next.index = index_count
                    index_count += 1
                    node_next_list.append(node_next)
                    calculate_smooth(cur_value, node_next, node)


                # already been searched and is a loop
                elif node_next.value == cur_value - 1:
                    if not self.pftree.loop_detection(node.mother_node, node_next, cur_value - 1):
                        node.loop_node2.append(node_next)

                # already searched and calculate the angle
                elif node_next.value == cur_value + 1:
                    # smoothness estimation
                    calculate_smooth(cur_value, node_next, node)

        self.pftree.add_section(union, node_list)

        return node_next_list

    def initialize_path(self):
        cur_section = self.section(self.end.value, self.end.index_section)
        node = self.end
        # node = self.pmap[718][812]
        # cur_section = self.section(*node.tree_pos)
        self.sect_in_path = {cur_section.pos}
        path = [node.pos]
        while node != self.start:
            # encounter a branching section
            self.calculate_loop_candidate(cur_section)  # the loop_section may have already been visited
            self.judge_loop_candidate(cur_section) # see whether the loop section will go back to the cur_section
            loop_candi_not_in_path = [sect for sect in cur_section.loop_candi if sect not in self.sect_in_path]
            if cur_section.loop_section is not None and random.random() < 0.5 and cur_section.loop_section.pos not in self.sect_in_path:
                next_section, node = self.path_branching(cur_section, cur_section.loop_section, node, path)

            # with a certain chance of trying a loop section with bigger value_all
            elif len(loop_candi_not_in_path) > 0 and random.random() > 1 / (len(loop_candi_not_in_path) + 1):
                loop_section = self.check_sect_history(cur_section.pos, cur_section.loop_candi)
                next_section, node = self.path_branching(cur_section, loop_section, node, path)

            # # encounter a loop section
            elif len(cur_section.parent) > 1:
                next_section, node = self.move_from_section_with_several_parents(cur_section, node, path)

            # just move forward
            else:
                for _ in range(cur_section.depth - cur_section.parent[0].depth):
                    node = self.path_one_step(node, path)
                next_section = cur_section.parent[0]

            # record section that has been visited
            cur_section = next_section
            self.sect_in_path.add(cur_section.pos)

        # self.plot_path_single_light(path, False) # debug
        path = self.reduce_reduent_nodes_initialization(path)
        return path

    def move_from_section_with_several_parents(self, cur_section, node, path):
        self.calculate_parent_sect(cur_section)
        if len(cur_section.parent_candi) > 0:
            sec_next = self.check_sect_history(cur_section.pos, cur_section.parent_candi)
            next_section, node = self.path_from_loop(sec_next, node, path)
        else:  # randomly choose the section of the mother_node or the section with the smallest value_all
            _, section_candi = min(
                [[sec_value, [sect]] for sect, sec_value in cur_section.v_all_dict.items()])
            if node.mother_node.tree_pos != section_candi[-1]:
                section_candi.append(node.mother_node.tree_pos)
            sec_next = self.check_sect_history(cur_section.pos, section_candi)
            next_section, node = self.path_from_loop(sec_next, node, path)
        return next_section, node

    def calculate_loop_candidate(self, cur_section):
        """check whether cur_section has a loop_section and create loop_candi"""

        def parent_sect_min(child_sect):
            """find the parent section with the smallest estimated fitness value, used to prevent a loop"""
            return self.section(*min(child_sect.v_all_dict, key=lambda k: child_sect.v_all_dict[k]))

        if cur_section.pos in self.child_sect_check_set or cur_section.pos not in self.pftree.key_sect:
            return
        self.child_sect_check_set.add(cur_section.pos)
        loop_candi = []
        for child_pos in cur_section.child:
            child = self.section(*child_pos)
            # if child.value_all >= cur_section.value_all * self.thresh_loop or child.pos in self.sect_in_path or parent_sect_min(child) == cur_section: # the loop section cannot be better
            if child.value_all >= cur_section.value_all * self.thresh_loop:  # the loop section cannot be better
                continue
            cost_sec1 = len(child) // 2  # the cost to move in this sect
            cost_path = self.pftree.cost_path[
                (cur_section.pos, child.pos)]  # the cost from this child to the indirect parent child
            cost_sec2 = len(cur_section) // 2  # the cost to move among the parent sect
            cost_all = cost_sec1 + cost_path + cost_sec2 + child.value_all
            """to be improved"""
            cost_all2 = cost_sec1 + child.depth - cur_section.depth + +cost_sec2 + child.value_all  # in a straight line
            if cost_all <= cur_section.value_all:  # there is almost no doubt that this path is better
                cur_section.loop_section = child
            elif cost_all2 < cur_section.value_all * self.thresh_loop:  # this path may be chosen
                loop_candi.append(child.pos)
                # self.calculate_parent_sect(child)
        # cur_section.loop_candi = [sect.pos for sect in loop_candi if sect.min_sect.pos != cur_section.pos or sect.loop_section is not None]
        cur_section.loop_candi = loop_candi

    def judge_loop_candidate(self, cur_section):
        """in case the child_section will go back to the cur_section"""
        if cur_section.pos in self.child_sect_check_set2 or cur_section.pos not in self.pftree.key_sect:
            return
        self.child_sect_check_set.add(cur_section.pos)
        loop_candi = []
        for child_pos in cur_section.loop_candi:
            child = self.section(*child_pos)
            if child.value_all >= cur_section.value_all:  # in case of a loop
                self.calculate_parent_sect(child)
                if len(child.parent_candi) == 0:
                    continue
                elif all([i in self.sect_in_path for i in child.parent_candi]): # parent sections have been visited
                    continue
            loop_candi.append(child.pos)
        # cur_section.loop_candi = [sect.pos for sect in loop_candi if sect.min_sect.pos != cur_section.pos or sect.loop_section is not None]
        cur_section.loop_candi = loop_candi


    def calculate_parent_sect(self, cur_section):
        """crate section_candi when encountering a loop section"""
        if cur_section.pos in self.parent_sect_check_set:
            return
        self.parent_sect_check_set.add(cur_section.pos)
        """needs improvement, cur_section may be the loop candidate of the parent section"""
        for sect_pos in cur_section.v_all_dict.keys():
            self.calculate_loop_candidate(self.section(*sect_pos))
        cur_section.parent_candi = [sect for sect, sec_value in cur_section.v_all_dict.items()
                                    if sec_value + len(cur_section) // 2 <= cur_section.value_all * self.thresh_parent
                                    and self.section(*sect).loop_section != cur_section
                                    and cur_section not in self.section(*sect).loop_candi]
        cur_section.min_sect = self.section(*min(cur_section.v_all_dict, key=lambda k: cur_section.v_all_dict[k]))

    def path_branching(self, cur_section, loop_section, node, path):
        """initialize path from a branching section to another branching section via a loop section"""
        def choose_branching_section(sect):
            branching_sect = sect.min_sect
            return branching_sect

        # generate path from the loop section to cur_section
        if (cur_section.pos, loop_section.pos) not in self.loop_path_set:
            self.loop_path_set.add((cur_section.pos, loop_section.pos))
            self.path_loop_branching(cur_section, loop_section)

        # if the current node does not lead to the loop section
        # dfs, find out the node in this section that leads to the loop section
        node_searched = set()
        _, node = self.dfs_path_branching(node, loop_section, path, node_searched)
        # move to the loop_section
        for _ in range(loop_section.depth - cur_section.depth):
            around = []
            x, y = node.pos
            for i in self.direction:
                pos = (x + i[0], y + i[1])
                if self.in_map(pos) and self.pmap[pos[0]][pos[1]].value == node.value + 1 and loop_section.pos in \
                        self.pmap[pos[0]][pos[1]].middle_path:
                    around.append(pos)
            node_pos = random.choice(around)
            node = self.pmap[node_pos[0]][node_pos[1]]
            path.append(node_pos)
        original_section = cur_section
        cur_section = loop_section
        self.sect_in_path.add(cur_section.pos)

        # move to the branching section
        # current section has several parent sections (cur_section is loop section)
        """Attention: this loop section may be a branching section"""
        self.calculate_loop_candidate(cur_section)
        if cur_section.loop_section is not None and cur_section.loop_section.pos not in self.sect_in_path:
            cur_section, node = self.path_branching(cur_section, cur_section.loop_section, node, path)
        else:
            cur_section, node = self.move_from_section_with_several_parents(cur_section, node, path)
        return cur_section, node

    def path_from_loop(self, branching_section, node, path, original_section=None):
        """generate the path from a loop section"""
        # choose the branching section and move towards it by one step
        # dfs, find out the node that leads to the branching section
        # original_section is used for path_branching, to prevent going back to the original branching_section
        node_searched = set()
        path.pop()  # the node will be added again in the DFS
        return self.dfs_path_from_loop(node, path, branching_section, node_searched, original_section)

    def dfs_path_from_loop(self, node, path, branching_section, node_searched, original_section=None):
        """move among the section, use DFS to find out the node connected to designated section"""
        node_searched.add(node.pos)
        path.append(node.pos)
        pos1 = (node.mother_node.value, node.mother_node.index_section)  # the position of mother node
        section_next = self.section(pos1[0], pos1[1])
        if branching_section == section_next or (
                len(section_next.parent) == 1 and branching_section == section_next.parent[
            0] and section_next != original_section):
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
            if not self.in_map((x_next, y_next)) or self.pmap[x_next][y_next].value != node.value or (
                    x_next, y_next) in node_searched:  # already searched
                continue
            node_next = self.pmap[x_next][y_next]
            result = self.dfs_path_from_loop(node_next, path, branching_section, node_searched, original_section)
            if result != False:  # has been found
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
                if (x_next, y_next) in node_searched or not (self.in_map((x_next, y_next))) or self.pmap[x_next][
                    y_next].value != node.value:
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
        # generate the path from the loop to the branching loop_section
        for node in loop_section:
            # repeat finding the parent node until reaching the branching loop_section
            flag = False  # whether there is a need to repeatedly find out the mother node
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
            if self.in_map(pos) and self.pmap[pos[0]][pos[1]].value < cur_value:
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
            if self.detect_obstacle(path2[-1], path[index]):
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
                        ax.text(i, j + 0.7, str(self.pmap[i][j].tree_pos))
                    else:
                        ax.text(i, j + 0.7, str(round(self.pmap[i][j].value_all, 1)))
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

    def plot_map_value_all_fast(self, block=True):
        """plot one path
           The color of each node represents its value_all"""
        maxx = max([self.pmap[i][j].value_all for i in range(self.size) for j in range(self.size) if
                    self.pmap[i][j].value_all != float("Inf")])
        map2 = np.array([[min(self.pmap[i][j].value_all, maxx) for j in range(self.size)] for i in range(self.size)])
        fig, ax = plt.subplots()
        map2 = map2.astype("int")
        # Define a custom colormap that goes from white (0) to deep blue (max value)
        cmap = plt.cm.get_cmap("gray_r")
        cmap.set_bad("white")  # Set zero values to white
        cmap.set_over("gray")  # Set zero values to white
        plt.imshow(map2, cmap=cmap)
        plt.show(block=block)
        ax.autoscale()
        plt.subplots_adjust(left=0.04, right=1, bottom=0.04, top=1)
        plt.show(block=block)
        # print(self.size, maxx) # debug

    def plot_map_key_sect(self, block=True, mode="common"):
        """plot one path
           the value_all of key sect is shown in the map"""
        searched = set()
        fig, ax = plt.subplots()
        self.map[config.p_start[0]][config.p_start[1]] = 0
        self.map[config.p_end[0]][config.p_end[1]] = 0
        self.map = self.map.astype("int")
        # Invert the color mapping: 1 to black and 0 to white
        cmap = plt.get_cmap("gray_r")
        plt.imshow(self.map, cmap=cmap)
        key_sect_pass = self.draw_structure(False, "simple")
        if mode == "common":
            for i in range(self.size):
                for j in range(self.size):
                    if self.pmap[i][j].value == float("Inf"):
                        continue
                    tree_pos = self.pmap[i][j].tree_pos
                    if tree_pos not in self.pftree.key_sect or tree_pos in searched:
                        continue
                    searched.add(tree_pos)
                    if tree_pos in key_sect_pass:  # sections that are initialized are drawn in red
                        ax.text(j, i, str(tree_pos) + str(round(self.section(*tree_pos).value_all)), color="red")
                    else:
                        ax.text(j, i, str(tree_pos) + str(round(self.section(*tree_pos).value_all)), color="blue")
        else:
            for i in range(self.size):
                for j in range(self.size):
                    if self.pmap[i][j].value == float("Inf"):
                        continue
                    tree_pos = self.pmap[i][j].tree_pos
                    if tree_pos not in key_sect_pass or tree_pos in searched:
                        continue
                    searched.add(tree_pos)
                    ax.text(j, i, str(tree_pos) + str(round(self.section(*tree_pos).value_all)), color="red",
                            fontsize=10)
        ax.autoscale()
        plt.subplots_adjust(left=0.04, right=1, bottom=0.04, top=1)
        plt.show(block=block)

    def plot_path_single_light(self, path, block=True, title: str = None):
        plt.figure()
        self.map[config.p_start[0]][config.p_start[1]] = 0
        self.map[config.p_end[0]][config.p_end[1]] = 0
        self.map = self.map.astype("int")
        # Invert the color mapping: 1 to black and 0 to white
        cmap = plt.get_cmap("gray_r")
        plt.imshow(self.map, cmap=cmap)
        plt.plot([i[1] for i in path], [i[0] for i in path], "r-", linewidth=3)
        plt.title(title)
        plt.show(block=block)

    def draw_structure(self, block=True, mode="all"):
        """draw the structure of the pftree
           mode: all shows all the sections
           mode: simple only shows sections that can be chosen during initialization"""
        plt.figure()
        if mode == "all":
            G = nx.MultiDiGraph()
            section_searched = set()
            # bfs, search for sections
            section_search = deque([self.section(*self.start.tree_pos)])
            G.add_node(self.start.tree_pos, )
            rn_colmap = []
            node_labels = {}
            while len(section_search) > 0:
                section = section_search.popleft()
                if section.pos in section_searched:
                    continue

                # different color for starting session and end session
                if section.pos == self.start.tree_pos or section.pos == self.end.tree_pos:
                    rn_colmap.append("darkorange")
                elif len(section.parent) > 1:
                    # print(section.pos, [section.parent[i].pos for i in range(len(section.parent))])
                    rn_colmap.append("red")
                else:
                    rn_colmap.append("lightblue")

                for child in section.child:
                    child = self.section(*child)
                    node_labels[child.pos] = [child.pos, round(child.value_all, 1)]
                    G.add_edge(section.pos, child.pos)
                    section_search.append(child)
                section_searched.add(section.pos)
            pos = graphviz_layout(G, prog="dot")  # 描画レイアウトの指定 Graphvizの"dot"というレイアウトを使用
            # for node, (x,y) in pos.items():
            #     pos[node] = (x, -200 * node[0])
            nx.draw_networkx(G, pos, with_labels=True, labels=node_labels, node_color=rn_colmap, node_size=200,
                             font_size=12)
            plt.show(block=block)
        else:
            key_sect_pass = {self.end.tree_pos}  # key sect that can be chosen when initializing the path
            G = nx.MultiDiGraph()
            section_searched = set()
            # bfs, search for sections
            section_search = deque([self.section(*self.end.tree_pos)])
            G.add_node(self.end.tree_pos, )
            rn_colmap = []
            node_labels = {}
            while len(section_search) > 0:
                section = section_search.popleft()
                if section.pos in section_searched:
                    continue
                section_searched.add(section.pos)

                # different color for starting session and end session
                if section.pos == self.start.tree_pos or section.pos == self.end.tree_pos:
                    rn_colmap.append("darkorange")
                elif len(section.parent) > 1:
                    # print(section.pos, [section.parent[i].pos for i in range(len(section.parent))])
                    rn_colmap.append("red")
                else:
                    rn_colmap.append("lightblue")
                if section.pos == (0, 0):
                    continue

                # add new sections
                # branching section
                sect_next_list = []
                self.calculate_loop_candidate(section)
                if section.loop_section is not None:
                    sect_next_list.append(section.loop_section)
                for sect_next in section.loop_candi:
                    sect_next_list.append(self.section(*sect_next))
                self.calculate_parent_sect(section)
                if len(section.parent_candi) > 0:
                    section_candi = section.parent_candi
                else:  # all parent sections can be chosen
                    section_candi = [sect for sect, sec_value in section.v_all_dict.items()]
                for sect_next in section_candi:
                    sect_next_list.append(self.section(*sect_next))

                # plot the structure
                for sect_next in sect_next_list:
                    node_labels[sect_next.pos] = [sect_next.pos, round(sect_next.value_all, 1)]
                    G.add_edge(section.pos, sect_next.pos)
                    section_search.append(sect_next)
                    key_sect_pass.add(sect_next.pos)
            pos = graphviz_layout(G, prog="dot")  # 描画レイアウトの指定 Graphvizの"dot"というレイアウトを使用
            nx.draw_networkx(G, pos, with_labels=True, labels=node_labels, node_color=rn_colmap, node_size=200,
                             font_size=12)
            plt.show(block=block)
            return key_sect_pass

    def in_map(self, pos):
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size and self.map[pos[0]][pos[1]] != 1

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

    def check_sect_history(self, cur_sect, sect_candi) -> section:
        """check the history of choosing sections to diversify the initialized population

        Args:
            cur_sect (tuple):  current section
            sect_candi (list): possible next sections

        Returns:
            section: chosen section
        """
        sect_candi_new = [sect for sect in sect_candi if
                          sect not in self.sect_chosen[cur_sect] and sect not in self.sect_in_path]  # sections that have not been searched in the past
        if len(sect_candi_new) == 0:  # all the sections have been chosen in the past
            sect_pos_not_in_path = [sect for sect in sect_candi if sect not in self.sect_in_path]
            """to be improved (sometimes visit sections repeatedly)"""
            if len(sect_pos_not_in_path) > 0:
                sect_pos = random.choice(sect_pos_not_in_path)
            else: # all sections have been visited, choose one visited section randomly
                sect_pos = random.choice(sect_candi)
            return self.section(*sect_pos)
        sect_pos = random.choice(sect_candi_new)
        self.sect_chosen[cur_sect].add(sect_pos)
        return self.section(*sect_pos)

    # def check_sect_in_path(self, sect_candi) -> bool:
    #     """check whether there all sections have already been visited
    #        True: unvisited sections exist
    #        False: all sections have been visited"""
    #     return not all([sect.pos in self.sect_in_path for sect in sect_candi])

    def print_all_section_path(self, path):
        """print the position of each node belongs to """
        for node in path:
            if self.pmap[node[0]][node[1]].tree_pos in self.pftree.key_sect:
                print(self.pmap[node[0]][node[1]].tree_pos)

    def debug_continuous_path(self, path):
        for i in range(len(path) - 1):
            if abs(path[i + 1][0] - path[i][0]) + abs(path[i + 1][1] - path[i][1]) > 2:
                print(path[i + 1], path[i])

    def print_key_section_path(self, path):
        """print the section each node in the path belongs to"""
        for i in range(len(path) - 1):
            node1, node2 = self.pmap[path[i][0]][path[i][1]], self.pmap[path[i + 1][0]][path[i + 1][1]]
            section1 = self.section(*node1.tree_pos)
            section2 = self.section(*node2.tree_pos)
            if section2 != section1 and section1.pos in self.pftree.key_sect:
                print(section1.pos)

    def reset(self):
        """self.draw_structure may influence the value of some variables"""
        self.child_sect_check_set = set()  # sections that have calculated loop sections
        self.parent_sect_check_set = set()  # sections that have calculated the candidates for parent sections
