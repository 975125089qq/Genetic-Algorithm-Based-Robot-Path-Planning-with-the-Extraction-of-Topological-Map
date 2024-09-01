import copy
from collections import defaultdict
import heapq


class Node:
    def __init__(self, x, y, value=float("Inf"), index=0):
        self.value = value
        self.value_all = value  # value that considers angle
        self.x = x
        self.y = y

        self.index = index  # for UnionFind
        self.mother_node = None
        self.index_section = None  # the index of the section

        self.loop_node = []  # for nodes with the same value
        self.loop_node2 = []  # for nodes with smaller value
        self.middle_path = set()  # set(pos of loop sections)

    @property
    def pos(self):
        return self.x, self.y

    @property
    def tree_pos(self):
        return self.value, self.index_section


class UnionFind():
    """
    Union Find木クラス

    Attributes
    --------------------
    n : int
        要素数
    root : list
        木の要素数
        0未満であればそのノードが根であり、添字の値が要素数
    rank : list
        木の深さ
    """

    def __init__(self, n):
        """
        Parameters
        ---------------------
        n : int
            要素数
        """
        self.n = n
        self.root = [-1] * n
        self.rank = [0] * n

    def find(self, x):
        """
        ノードxの根を見つける

        Parameters
        ---------------------
        x : int
            見つけるノード

        Returns
        ---------------------
        root : int
            根のノード
        """
        if (self.root[x] < 0):
            return x
        else:
            self.root[x] = self.find(self.root[x])
            return self.root[x]

    def unite(self, x, y):
        """
        木の併合

        Parameters
        ---------------------
        x : int
            併合したノード
        y : int
            併合したノード
        """
        x = self.find(x)
        y = self.find(y)

        if (x == y):
            return
        elif (self.rank[x] > self.rank[y]):
            self.root[x] += self.root[y]
            self.root[y] = x
        else:
            self.root[y] += self.root[x]
            self.root[x] = y
            if (self.rank[x] == self.rank[y]):
                self.rank[y] += 1

    def same(self, x, y):
        """
        同じグループに属するか判定

        Parameters
        ---------------------
        x : int
            判定したノード
        y : int
            判定したノード

        Returns
        ---------------------
        ans : bool
            同じグループに属しているか
        """
        return self.find(x) == self.find(y)

    def size(self, x):
        """
        木のサイズを計算

        Parameters
        ---------------------
        x : int
            計算したい木のノード

        Returns
        ---------------------
        size : int
            木のサイズ
        """
        return -self.root[self.find(x)]

    def roots(self):
        """
        根のノードを取得

        Returns
        ---------------------
        roots : list
            根のノード
        """
        return set([i for i, x in enumerate(self.root) if x < 0])

    def group_size(self):
        """
        グループ数を取得

        Returns
        ---------------------
        size : int
            グループ数
        """
        return len(self.roots())

    def group_members(self, node_list, depth):
        """
        全てのグループごとのノードを取得

        Returns
        ---------------------
        group_members : defaultdict
            根をキーとしたノードのリスト
        """
        group_members = {}
        for node in node_list:
            index_section = self.find(node.index)
            if self.find(node.index) not in group_members:
                group_members[index_section] = Section(depth, index_section)
            group_members[index_section].append(node)
            node.index_section = index_section
        return group_members


class Section(list):
    """ basic component of the Pftree"""

    def __init__(self, depth, index):
        super().__init__()
        self.branch = 0
        self.parent = []
        self.child = set()
        self.depth = depth
        self.index = index
        self.value_all = float("Inf")  # value considering smoothness
        self.v_all_dict = {}  # {parent_section.pos:value_all}, record value of nodes from each parent section
        self.v_all_dict_ud = {}  # updated value_all diction
        self.loop_section = None  # loop sections that can be chosen
        self.loop_candi = []  # candidate for possible loop sections
        # self.min_sect = None # parent section with the smallest value_all
        # self.parent_candi = None # candidate for possible parent sections

    @property
    def pos(self):
        return self.depth, self.index


class Pftree():
    """the tree structure to generate the potential field
       needs to manage the relation between different section of the trees"""

    def __init__(self, end, size, MAP, pmap):
        self.depth = 0  # start from 0
        self.union_list = []
        self.sections = []  # [{index:Section()}]
        self.end_node = end  # the end node
        self.end_section = None
        self.direction = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        self.size = size
        self.map = MAP
        self.pmap = pmap
        self.key_sect = set()  # division section or loop section
        self.cost_path = {}  # {(sect_pos1, sect_pos2): value_all}

    def add_section(self, union, node_list):
        """deepen the tree by one """

        def connect_section_calculate_value_all():
            for section in self.sections[-1].values():  # section in {index: section}
                self.key_sect.add(section.pos)
                value_all_mother = {}
                index_parent_set = set()
                for node in section:
                    # smoothness
                    if node.mother_node.tree_pos not in section.v_all_dict or section.v_all_dict[
                        node.mother_node.tree_pos] > node.value_all:
                        section.v_all_dict[node.mother_node.tree_pos] = node.value_all
                    # bug fixed: the section can be too small and the parent section is in loop_node2
                    # its value in the dict is Inf
                    # choose the parent node in loop_node2 to approximate
                    if len(node.loop_node2) > 0:
                        for node2 in node.loop_node2:
                            if node2.tree_pos not in value_all_mother or value_all_mother[
                                node2.tree_pos] > node2.value_all:
                                value_all_mother[node2.tree_pos] = node2.value_all

                    # parent section
                    node_parent = node.mother_node
                    index_parent_list = [self.union_list[-2].find(node_parent.index)]
                    if len(node.loop_node) > 0:
                        for i in node.loop_node:
                            node_parent = i.mother_node
                            index_parent_list.append(self.union_list[-2].find(node_parent.index))
                    if len(node.loop_node2) > 0:
                        for i in node.loop_node2:
                            index_parent_list.append(self.union_list[-2].find(i.index))
                    for index_parent in index_parent_list:
                        if index_parent not in index_parent_set:
                            section_parent = self.sections[-2][index_parent]
                            section.parent.append(section_parent)
                            section_parent.branch += 1
                            section_parent.child.add(section.pos)
                            index_parent_set.add(index_parent)

                # calculate value_all for every parent section
                for section_parent in section.parent:
                    if section_parent.pos not in section.v_all_dict:  # bug fixed: parent section in loop_node2
                        section.v_all_dict[section_parent.pos] = value_all_mother[section_parent.pos]
                section.value_all = min(section.v_all_dict.values())  # value_all for this section

        def check_reach_goal():
            if self.depth == self.end_node.value:
                self.end_section = self.sections[self.depth][self.end_node.index_section]

        def shorten_tree():
            def update_cost_path(sect_parent, sect_middle, sect):
                # delete old paths
                if sect_middle is not None and (sect_parent.pos, sect_middle.pos) in self.cost_path:
                    self.cost_path.pop((sect_parent.pos, sect_middle.pos))

                # add new path
                # check whether there are two paths between sect and sect_parent
                key = (sect_parent.pos, sect.pos)
                if key not in self.cost_path:
                    self.cost_path[key] = sect.v_all_dict[sect_parent.pos] - sect_parent.value_all
                else:
                    self.cost_path[key] = min(self.cost_path[key],
                                              sect.v_all_dict[sect_parent.pos] - sect_parent.value_all)

            for section in self.sections[-1].values():
                if len(section.parent) == 1:
                    # delete section_middle
                    if section.parent[0].branch == 1 and len(section.parent[0].parent) == 1 and section.parent[
                        0] != self.end_section:
                        section_middle = section.parent[0]
                        section.parent = section_middle.parent
                        section.parent[0].child.remove(section_middle.pos)
                        section.parent[0].child.add(section.pos)
                        self.key_sect.remove(section_middle.pos)
                        section.v_all_dict[section.parent[0].pos] = section.v_all_dict.pop(
                            section_middle.pos)  # update v_all_dict
                        update_cost_path(section.parent[0], section_middle, section)
                    # add path to cost_path
                    else:
                        update_cost_path(section.parent[0], None, section)

                # find out section of loop
                elif len(section.parent) >= 2:
                    tem = []
                    for section_parent in section.parent:
                        if section_parent.branch == 1 and len(
                                section_parent.parent) == 1 and section_parent != self.end_section and section.pos not in \
                                section_parent.parent[0].child:  # avoid two paths between two sections
                            section_parent.parent[0].child.remove(section_parent.pos)
                            self.key_sect.remove(section_parent.pos)
                            section_parent.parent[0].child.add(section.pos)
                            tem.append(section_parent.parent[0])
                            section.v_all_dict[section_parent.parent[0].pos] = section.v_all_dict.pop(
                                section_parent.pos)
                            update_cost_path(section_parent.parent[0], section_parent, section)
                        else:
                            section_parent.child.add(section.pos)
                            tem.append(section_parent)
                            update_cost_path(section_parent, None, section)
                    section.parent = tem
                section.v_all_dict_ud = copy.deepcopy(section.v_all_dict)

        def update_value_all():
            """update value_all for current section and parent sections"""

            def update_node():
                """update the value_all for node in each section"""

                def update_node_one_parent():
                    dif = sect_ud.v_all_dict[sect_ud.parent[0].pos] - sect_ud.v_all_dict_ud[sect_ud.parent[0].pos]
                    for node in sect_ud:
                        for dir in self.direction:
                            x_next = node.pos[0] + dir[0]
                            y_next = node.pos[1] + dir[1]
                            if not self.in_map((x_next, y_next)):
                                continue
                            node_next = self.pmap[x_next][y_next]
                            if node_next.mother_node == node:
                                node_next.value_all -= dif

                def update_node_several_parents():
                    for node in sect_ud:
                        sect_par_pos = node.mother_node.tree_pos
                        if sect_par_pos not in self.key_sect:
                            sect_par_pos = self.return_section(*sect_par_pos).parent[0].pos
                        dif1 = sect_ud.v_all_dict[sect_par_pos] - sect_ud.v_all_dict_ud[sect_par_pos]
                        dif2 = sect_ud.v_all_dict[sect_par_pos] - sect_ud.value_all - len(sect_ud) // 2  # traverse sect
                        dif = max(dif1, dif2)
                        if dif <= 0:
                            continue
                        for dir in self.direction:
                            x_next = node.pos[0] + dir[0]
                            y_next = node.pos[1] + dir[1]
                            if not self.in_map((x_next, y_next)):
                                continue
                            node_next = self.pmap[x_next][y_next]
                            if node_next.mother_node == node:
                                node_next.value_all -= dif

                # update nodes found by current section (reduce the value of nodes from sections with bigger value_all)
                for sect_pos in lowest_sect:
                    sect_ud = self.return_section(*sect_pos)
                    if len(sect_ud.parent) <= 1:
                        update_node_one_parent()
                    else:
                        # can traverse the section
                        update_node_several_parents()

            def update_parent_sect(section):
                # update parent section
                cost_sec1 = len(section) // 2  # the cost to move in this sect
                for section_parent, value_all in section.v_all_dict_ud.items():
                    section_parent = self.return_section(*section_parent)
                    cost_path = self.cost_path[
                        (section_parent.pos, section.pos)]  # the cost from this section to the indirect parent section
                    cost_sec2 = len(section_parent) // 2  # the cost to move among the parent sect
                    cost_all = cost_sec1 + cost_path + cost_sec2 + section.value_all
                    if cost_all < section_parent.value_all:  # there is almost no doubt that this path is better
                        section_parent.value_all = cost_all
                        heapq.heappush(update_sect_list, ([section_parent.value_all, section_parent.pos]))

            def update_child_sect(sect_ud):
                """update the value_all for child section of section_update"""
                # if it is the lowest section, update the node
                if len(sect_ud.child) == 0 and sect_ud.depth == self.depth:
                    lowest_sect.add(sect_ud.pos)
                    return

                # update the child_section
                for child_pos in sect_ud.child:
                    sect_child = self.return_section(*child_pos)
                    # print(sect_child.v_all_dict_ud[sect_ud.pos], sect_ud.value_all + self.cost_path[(sect_ud.pos, sect_child.pos)] )
                    sect_child.v_all_dict_ud[sect_ud.pos] = sect_ud.value_all + self.cost_path[
                        (sect_ud.pos, sect_child.pos)]
                    v_all_new = min(sect_child.v_all_dict_ud.values())
                    if v_all_new >= sect_child.value_all:
                        continue
                    sect_child.value_all = v_all_new
                    heapq.heappush(update_sect_list, ([sect_child.value_all, sect_child.pos]))

            """though heapq, sections are seldom updated
               almost no influence to speed"""
            update_sect_list = [[s.value_all, s.pos] for s in self.sections[-1].values() if
                                len(s.parent) != 1]  # sections that need to be updated
            heapq.heapify(update_sect_list)
            lowest_sect = {s[1] for s in update_sect_list}  # sections whose nodes need to be updated
            while len(update_sect_list) > 0:
                _, sect_pos = heapq.heappop(update_sect_list)
                sect = self.return_section(*sect_pos)
                update_parent_sect(sect)
                update_child_sect(sect)
            update_node()

        # ______________________________________________________________
        # ______________________________________________________________
        # ______________________________________________________________
        # ______________________________________________________________
        layer = union.group_members(node_list, self.depth)  # {index: section}
        self.union_list.append(union)
        self.sections.append(layer)
        # start session
        if self.depth == 0:
            self.depth += 1
            self.sections[0][0].value_all = 0
            return

        # connect parent sections and estimate the smoothness of the section
        connect_section_calculate_value_all()

        # check whether goal has been visited
        check_reach_goal()

        # shorten the tree (when it is the situation of one parent section and one child section)
        shorten_tree()

        # update value_all for section and parent sections
        update_value_all()

        self.depth += 1

    def loop_detection(self, node1, node2, depth):
        """detect loop
           True: not loop
           False: loop"""
        if depth > 0:
            return self.union_list[depth].same(node1.index, node2.index)
        else:
            return True

    def in_map(self, pos):
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size and self.map[pos[0]][pos[1]] != 1

    def return_section(self, depth, section_index):
        """return the section"""
        return self.sections[depth][section_index]
