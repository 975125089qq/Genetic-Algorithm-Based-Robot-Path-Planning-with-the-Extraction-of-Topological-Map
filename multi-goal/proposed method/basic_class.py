from collections import deque
from collections import defaultdict

class Node:
    def __init__(self, x, y, value=float("Inf"), index=0):
        self.value = value
        self.value_all = value  # value that considers angle
        self.x = x
        self.y = y

        self.index = index  # for UnionFind
        self.start_id = None # found by which starting point
        self.mother_node = None
        self.index_section = None  # the index of the section

        self.loop_node = []  # for nodes with the same value
        self.loop_node2 = []  # for nodes with smaller value
        self.node_other_start = [] # nodes found by other starting point
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
                group_members[index_section].start_id = node.start_id
            group_members[index_section].append(node)
            node.index_section = index_section
        return group_members


class Section(list):
    """ basic component of the Pftree"""

    def __init__(self, depth, index):
        super().__init__()
        self.start_id = None # the index of the starting point
        self.branch = 0
        self.parent = []
        self.parent_section = None
        self.child = set()
        self.depth = depth
        self.index = index
        self.value_all = float("Inf") # value considering smoothness
        self.value_all_list = [] # record the value_all for each parent section(?)
        self.loop_section = None  # loop sections that can be chosen
        self.loop_candi = [] # candidate for possible loop sections
        self.value_all_dict = defaultdict(lambda: float("Inf")) # {parent_section.pos:value_all}, record value of nodes from each parent section

    @property
    def pos(self):
        return self.depth, self.index


class Pftree():
    """the tree structure to generate the potential field
       needs to manage the relation between different section of the trees"""

    def __init__(self, size, MAP, pmap):
        self.depth = 0  # start from 0
        self.union_list = []
        self.sections = []  # [{index:Section()}]
        self.middle_sec_set = set() # record the middle sections
        self.middle_nodes = set() # record middle nodes
        self.direction = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        self.size = size
        self.map = MAP
        self.pmap = pmap
        self.key_sect = set() # division section or loop section

    def add_section(self, union, node_list, conn):
        """deepen the tree by one """
        section = union.group_members(node_list, self.depth) # {index: section}
        self.union_list.append(union)
        self.sections.append(section)
        # find the parent section and estimate the smoothness of the section
        value_all_dict_all = {}  # {section.pos:value_all_dict}

        if self.depth < 1:
            self.depth += 1
            return

        for section in self.sections[-1].values():  # section in {index: section}
            self.key_sect.add(section.pos)
            value_all_dict = section.value_all_dict
            value_all_mother = defaultdict(lambda : float("Inf"))
            value_all_dict_all[section.pos] = value_all_dict
            index_parent_set = set()

            for node in section:
                # smoothness
                if value_all_dict[node.mother_node.tree_pos] > node.value_all:
                    value_all_dict[node.mother_node.tree_pos] = node.value_all
                if len(node.loop_node2) > 0:
                    # choose the parent node in loop_node2 to approximate
                    for node2 in node.loop_node2:
                        if value_all_mother[node2.tree_pos] > node2.value_all:
                            value_all_mother[node2.tree_pos] = node2.value_all

                # parent section and record the starting point
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

                # section from other starting point
                conn.check_group_node(node, self.sections)

            # calculate value_all for every parent section
            for section_parent in section.parent:
                if section_parent.pos not in value_all_dict: # bug fixed: parent section in loop_node2
                    value_all_dict[section_parent.pos] = value_all_mother[section_parent.pos]
                section.value_all_list.append(value_all_dict[section_parent.pos])
            section.value_all = min(section.value_all_list)  # value_all for this section
            section.parent_section = section.parent[section.value_all_list.index(section.value_all)]

        # shorten the tree (when it is the situation of one parent section and one child section)
        for section in self.sections[-1].values():
            if len(section.parent) == 1:
                if section.parent[0].branch == 1 and len(section.parent[0].parent) == 1:
                    section_middle = section.parent[0]
                    section.parent = section_middle.parent
                    section.parent[0].child.remove(section_middle.pos)
                    section.parent[0].child.add(section.pos)
                    self.key_sect.remove(section_middle.pos)

            # find out section of loop
            elif len(section.parent) >= 2:
                tem = []
                for section_parent in section.parent:
                    if section_parent.branch == 1 and len(section_parent.parent) == 1:
                        section_parent.parent[0].child.remove(section_parent.pos)
                        self.key_sect.remove(section_parent.pos)
                        section_parent.parent[0].child.add(section.pos)
                        tem.append(section_parent.parent[0])
                    else:
                        section_parent.child.add(section.pos)
                        tem.append(section_parent)
                section.parent = tem

                # update value_all for section and parent sections
                node_updated = set()
                self.update_value_all(section, node_updated, value_all_dict_all)
        self.depth += 1

    def update_value_all(self, section, node_updated, value_all_dict_all):
        """update value_all for current section and parent sections"""

        # update nodes found by current section (reduce the value of nodes from sections with bigger value_all)
        for node in section:
            value = value_all_dict_all[section.pos][node.mother_node.tree_pos]
            dif = value - section.value_all - len(section)//2
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

        # update parent section
        section_updated = {section.pos} # sections that have been updated
        cost_sec1= len(section)//2 # the cost to move in this section
        for index, value_all in enumerate(section.value_all_list):
            if value_all == section.value_all: # the smallest one
                continue
            section_parent = section.parent[index]
            value_all2 = section_parent.value_all # the value_all of the indirect parent section
            cost_path = value_all - value_all2 # the cost of the middle path
            cost_sec2 = len(section_parent) // 2 # the cost to move among the parent section
            cost_all = cost_sec1 + cost_path + cost_sec2 + section.value_all
            cost_all2 = cost_sec1 + section.depth-section_parent.depth + cost_sec2 + section.value_all # in a straight line
            # print(section.pos, cost_all, cost_all2, section_parent.value_all)
            if cost_all < section_parent.value_all: # there is almost no doubt that this path is better
                section_parent.loop_section = section
                dif = section_parent.value_all - cost_all  # to change the value_all of other sections
                section_parent.value_all = cost_all
                self.update_section(section_parent, section_updated, dif, node_updated)
            elif cost_all2 < section_parent.value_all: # this path may be chosen
                section_parent.loop_candi.append(section)


    def update_section(self, section, section_updated, dif, node_updated):
        # child
        section_search = deque(section.child)
        while len(section_search) > 0:
            section2 = section_search.pop()
            if section2 in section_updated:
                continue
            section2 = self.sections[section2[0]][section2[1]]
            section2.value_all -= dif
            for key, value in section2.value_all_dict.items():
                section2.value_all_dict[key] = value-dif
            for i in section2.child:
                section_search.append(i)
            section_updated.add(section2.pos)
            # print(section2.child, section2.depth, self.depth, len(section2.child) == 0)

            # update the value_all for node of the lowest sections
            if len(section2.child) == 0 and section2.depth == self.depth:
                for node in section2:
                    for dir in self.direction:
                        pos_next = (node.x + dir[0], node.y + dir[1])
                        # not in map
                        if not self.in_map(pos_next) or pos_next in node_updated:
                           continue
                        node_next = self.pmap[pos_next[0]][pos_next[1]]
                        node_next.value_all -= dif
                        node_updated.add(node_next.pos)

        """TD: parent to be updated"""
        # # parent
        # for index, value_all in enumerate(section.path):
        #     if value_all == section.value_all: # the smallest one
        #         continue
        #     section_parent = section.parent[index]
        #     value_all2 = section_parent.value_all # the value_all of the indirect parent section
        #     cost_path = value_all - value_all2 # the cost of the middle path
        #     cost_sec2 = len(section) // 2 # the cost to move among the parent section
        #     cost_all = cost_sec1 + cost_path + cost_sec2
        #     cost_all = -10
        #     print(section.pos, cost_all)
        #     if cost_all < section_parent.value_all:
        #         dif = section_parent.value_all - cost_all  # to change the value_all of other sections
        #         section_parent.value_all = cost_all
        #         self.update_section(section_parent, section_updated, dif)

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


class Conn:
    """
        record the connection between every pair of targets

        !!to do:
            sort the section by value_all
    """
    def __init__(self):
        self.conn_all = defaultdict(list) # {(two_targets):[nodes]}, connection between targets

    def check_group_node(self, node, sections):
        """check whether connections between targets exist

        Args:
            node: node of the map, node_other_start records the connection with other starting points

        """
        if len(node.node_other_start) == 0:
            return
        start1 = node.start_id
        session1 = sections[node.tree_pos[0]][node.tree_pos[1]]
        value1 = session1.value_all
        for node2 in node.node_other_start:
            start2 = node2.start_id
            if start1 < start2:
                two_targets = (start1, start2)
                node_pair = [node, node2]
            else:
                two_targets = (start2, start1)
                node_pair = [node2, node]
            self.conn_all[two_targets].append(node_pair)