import numpy as np
import time
import random
from draw_path import draw_path

def sgmp(IMAP, sp, dp, setSize, numPt):
    global OBSTACLE_PENALTY
    global mapColSize
    global map
    map = IMAP


    # Start TIC for measuring the execution time.
    start_time = time.time()

    # Initialize "pathSet" according to the number of paths to be created.
    pathSet = [None] * setSize

    # Identify the size of given map specified in "map".
    mapRowSize, mapColSize = map.shape

    # To identify the infeasible edge, we initialize "OBSTACLE_PENALTY"
    # by using a value that cannot be a value of distance from given map.
    # OBSTACLE_PENALTY = mapColSize * 2
    OBSTACLE_PENALTY = np.inf

    # Identify the index of obstacle nodes.
    # It is useful to accelerate the process of creating DAG.
    obstacleRow, obstacleCol = np.where(map == 1)
    obstacle = set(obstacleRow * mapRowSize + obstacleCol)

    # Identify the index of free-space nodes.
    # It is useful to accelerate the process of creating DAG.
    # Because starting node and destination node cannot be included in
    # the path as intermediate nodes, we treat them as obstacles.
    feasibleCell = [i for i in range(mapRowSize*mapColSize) if i not in obstacle and i!= sp and i!= dp]

    # Because the variable "dag" grows according to the number of included
    # nodes, the size of dag will be changed dynamically.
    # However, in Python, we can use a dynamic list, so we don't need to
    # specify a fixed size like in MATLAB.
    dag = []

    # Initialize the size of "dag".
    # "dagSize" can be used as the index of the to-be-included node on "dag".
    dagSize = 0

    # To find the detailed information about "tNode", go to Line 140.
    tNode = [0, 0, 0]

    # To find the detailed information about "knnNodes", go to Line 160.
    knnNodes = np.full(numPt, np.nan)

    # Because DAG always starts from the starting node,
    # it should be included in the first row of "dag".
    tem = indexToRowCol(sp, mapColSize)
    dag.append([int(sp), int(tem[0]), int(tem[1])])

    # Increase the size of "dag" because "sp" was included in "dag".
    dagSize += 1

    # SGMP runs until "dp" is included at the end of DAG.
    # Thus, it will fall into an infinite loop if there is no feasible path
    # connecting the starting and destination nodes. So, be careful.

    while dag[0][-1] != dp:

        # Identify a direction node "tNode" to branch the "dag"
        # by choosing a free-space node from "feasibleCell".
        # The "tNode" variable is a 1 by 3 array that contains
        # [Index of node, Vertical Axis, Horizontal Axis].
        # Note that "tNode" may not be included in the "dag"
        # because there can be obstacle nodes between DAG and "tNode".
        tNode[0] = random.choice(feasibleCell)

        tNode[1], tNode[2] = indexToRowCol(tNode[0], mapColSize)

        # For identifying the nearest nodes of "tNode" that are already
        # included in the "dag", the values of Euclidean distance
        # between "tNode" and all nodes in "dag" are calculated.

        euDist = []
        for i in range(len(dag)):
            dag_x, dag_y = dag[i][1], dag[i][2]
            euDist.append(np.sqrt((dag_x-tNode[1])**2+(dag_y-tNode[2])**2))
        # The variable "idx" contains the position (row) in "dag" of
        # nearest nodes from "tNode".
        idx = np.argsort(euDist)


        # Identify nearest nodes that can be used as candidate parent nodes.
        knnNodes[:min(numPt, dagSize)] = idx[:min(numPt, dagSize)]

        # Identify a node to be actually included in "dag".
        # The function "nextState" returns the furthest node "actualNode"
        # from the nearest node "knnNodes[0]" that is a node between the edge "<nearestNode, tNode>".
        actualNode = nextState(dag[int(knnNodes[0])][1:3], tNode[1:])

        # According to the situation of DAG, there may not exist a node
        # to be included in the DAG for branching it further to "tNode".
        # 1) When the nearest node adjoins the obstacle nodes, so that DAG
        #    cannot be branched any further to "tNode".
        # 2) When DAG already contains the "actualNode".
        # 3) If there are obstacle nodes between the nearest node and the actual node.
        # In those cases, it is failed to find a new node for branching DAG.

        if actualNode is None or actualNode == dag[dagSize - 1][0] or np.isinf(map_dist_realtime(dag[int(knnNodes[0])][0], actualNode)):
            continue

        # Include a new node "actualNode" in "dag".
        dag.append([actualNode, *indexToRowCol(actualNode, mapColSize)])

        # Connect additional parent nodes specified in "numPt" if possible.
        # If it is unable to include additional parent nodes, mark it as "NaN" value.
        for i in range(1, numPt):
            if np.isnan(knnNodes[i]) or np.isinf(map_dist_realtime(dag[int(knnNodes[i])][0], actualNode)):
                knnNodes[i] = np.nan

        # Assign the position of parent nodes in "dag".
        dag[-1].extend(knnNodes.tolist())
        # Increase the size of "dag" because "actualNode" was included in "dag".
        dagSize += 1
        # Test the last-added node "actualNode" can be connected.
        # If it is possible, terminate the while-loop (Line 140).
        if not np.isinf(map_dist_realtime(dag[dagSize - 1][0], dp)):
            dag.append([dp, *indexToRowCol(dp, mapColSize), dagSize - 1])
            dagSize += 1
            break
    # Generate the path based on the obtained "dag".
    for pop in range(setSize):
        pathSet[pop] = createPathFrom(dag, sp)

    execTime = time.time() - start_time
    return pathSet, execTime, dag

def createPathFrom(dag, sp):
        # A function for creating a path from "dag".

        # Because "dp" always has only one parent and the path is created by
        # back-tracking from "dp" to "sp", the variables "node" and "parent"
        # are initialized as follows.
        node = dag[-1][0]
        parent = dag[-1][3]
        path = []

        # In the variable "dag", the parent of "sp" is initialized as the value 0.
        # Thus, if the parent is set to 0, then it means a path is created.
        while parent != 0:
            path = [node] + path
            node = dag[parent][0]

            # Identify parents of "node".
            parents = [p for p in dag[parent][3:] if not np.isnan(p)]
            # Select a parent randomly from specified parents.
            parent = random.choice(parents)
            parent = int(parent)

        # Include "sp" at the head of "path".
        path = [sp] + [node] + path
        return path


def nextState(nearestCell, tNode):
    # A function for identifying "actualNode" that will be included in "dag".
    nRow, nCol = nearestCell
    rRow, rCol = tNode

    x, y = return_before_obstacle((nRow, nCol), (rRow, rCol), map)
    if x == nRow and y == nCol:
        return None
    return rowColToIndex(x, y, mapColSize)


def indexToRowCol(idx, colSize):
    # A function for changing the index of node "idx" to vertical and horizontal coordinates.

    row = idx // colSize
    col = idx % colSize
    return row, col


def rowColToIndex(row, col, colSize):
    # A function for changing the vertical and horizontal coordinates to the index of node "idx".

    return row * colSize + col

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

# find out if two nodes can be connected directly
def return_before_obstacle(node1, node2, MAP):
    """return the node before running into an obstacle
       used in bresenham"""
    row1, line1 = node1
    row2, line2 = node2
    EPS = 1e-6
    res = np.array([row1, line1])
    if row1 != row2:
        k = (line2 - line1) / (row2 - row1)
        b = line1 - k * (row1 + 0.5) + 0.5
        row_min, row_max = min(row1, row2), max(row1, row2)
        rows = [row_min + 0.5] + [i for i in range(row_min + 1, row_max + 1)] + [row_max + 0.5]
        if row1 > row2:
            rows.reverse()
        lines = [i * k + b for i in rows]

        if k > 0:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] + EPS)
                line_2_int = int(lines[i + 1] - EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int + 1):
                    if MAP[row_current][j] == 1:
                        return res
                    else:
                        res[0], res[1] = row_current, j
            res[0], res[1] = row2, line2

        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    if MAP[row_current][j] == 1:
                        return res
                    else:
                        res[0], res[1] = row_current, j
            res[0], res[1] = row2, line2

    else:
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1, line2 + line_direction, line_direction):
            if MAP[row1][i] == 1:
                return res
            else:
                res[0], res[1] = row1, i
        res[0], res[1] = row2, line2
    return res

def bresenham(x1, y1, x2, y2, flag):
    # Create Feasible Edge Function (or Modified Bresenham Algorithm)
    # The function returns (x, y)-coordinates between <(x1, y1), (x2, y2)>.
    # If "flag" is set to "True", the function will terminate immediately
    # when it's stuck by obstacle nodes and returns coordinates identified so far.
    # In the other case, the function will perform a modified Bresenham Algorithm.

    x1, x2 = round(x1), round(x2)
    y1, y2 = round(y1), round(y2)
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    steep = abs(dy) > abs(dx)

    if steep:
        dx, dy = dy, dx

    if dy == 0:
        q = np.zeros(dx + 1)
    else:
        indices = np.arange(np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - dy, -dy)
        mod_result = np.mod(indices, dx)
        mod_diff = np.diff(mod_result) >= 0
        q = np.concatenate(([0], mod_diff))

    if steep:
        if y1 <= y2:
            y = np.arange(y1, y2 + 1).reshape(-1, 1)
        else:
            y = np.arange(y1, y2 - 1, -1).reshape(-1, 1)

        if x1 <= x2:
            x = x1 + np.cumsum(q).reshape(-1, 1)
        else:
            x = x1 - np.cumsum(q).reshape(-1, 1)
    else:
        if x1 <= x2:
            x = np.arange(x1, x2 + 1).reshape(-1, 1)
        else:
            x = np.arange(x1, x2 - 1, -1).reshape(-1, 1)

        if y1 <= y2:
            y = y1 + np.cumsum(q).reshape(-1, 1)
        else:
            y = y1 - np.cumsum(q).reshape(-1, 1)
    if not flag:
        return x, y

    otherQ = np.zeros(q.size + np.count_nonzero(q == 1), dtype=int)
    cum = 0
    newX = np.zeros(otherQ.size, dtype=int)
    newY = np.zeros(otherQ.size, dtype=int)

    if steep:
        for i in range(q.size):
            if q[i] == 1:
                newX[i + cum] = x[i - 1]
                newY[i + cum] = y[i]
                if map[newX[i + cum], newY[i + cum]] == 1:
                    newX = newX[:i - 1 + cum]
                    newY = newY[:i - 1 + cum]
                    break
                cum += 1
                otherQ[i + cum] = 1
                newX[i + cum] = x[i]
                newY[i + cum] = y[i]
                if map[newX[i + cum], newY[i + cum]] == 1:
                    newX = newX[:i + cum]
                    newY = newY[:i + cum]
                    break
            else:
                newX[i + cum] = x[i]
                newY[i + cum] = y[i]
                if map[newX[i + cum], newY[i + cum]] == 1:
                    newX = newX[:i + cum]
                    newY = newY[:i + cum]
                    break
    else:
        for i in range(q.size):
            if q[i] == 1:
                newX[i + cum] = x[i]
                newY[i + cum] = y[i - 1]
                if map[newX[i + cum], newY[i + cum]] == 1:
                    newX = newX[:i + cum]
                    newY = newY[:i + cum]
                    break
                cum += 1
                otherQ[i + cum] = 1
                newX[i + cum] = x[i]
                newY[i + cum] = y[i]
                if map[newX[i + cum], newY[i + cum]] == 1:
                    newX = newX[:i + cum]
                    newY = newY[:i + cum]
                    break
            else:
                newX[i + cum] = x[i]
                newY[i + cum] = y[i]
                if map[newX[i + cum], newY[i + cum]] == 1:
                    newX = newX[:i + cum]
                    newY = newY[:i + cum]
                    break

    return newX, newY


def map_dist_realtime(sp, ep):

    # A function for calculating the Euclidean distance of <sp, ep>.
    # Please do not care about the "realtime".

    # Because we don't use global variables in Python, we removed the global
    # statements for "mapColSize" and "map".
    n = map.shape[0]
    penalty = np.inf
    sp = np.array([sp//n, sp%n], dtype="int")
    ep = np.array([ep//n, ep%n], dtype="int")

    # Distance between two nodes
    sdist = np.linalg.norm(sp - ep)
    # Check if there is any obstacle node intersecting the edge
    if abs((ep[1] - sp[1]) / (ep[0] - sp[0])) == 1:
        startR = min(sp[0], ep[0])
        startC = min(sp[1], ep[1])
        endR = max(sp[0], ep[0])
        endC = max(sp[1], ep[1])
        sdist = max(sdist, np.sum(np.diag(map[startR:endR + 1, startC:endC + 1])) * penalty)
    elif sp[0] == ep[0]:
        sdist = max(sdist, np.sum(map[sp[0], min(sp[1], ep[1]):max(sp[1], ep[1]) + 1]) * penalty)
    elif sp[1] == ep[1]:
        sdist = max(sdist, np.sum(map[min(sp[0], ep[0]):max(sp[0], ep[0]) + 1, sp[1]]) * penalty)
    else:
        # original code
        # obsCount = 0
        # x, y = bresenham(sp[0], sp[1], ep[0], ep[1], False)
        # """The following part is much slower than the original Matlab code.
        #    But likely, it is the characteristic of Python (to run slow) rather than a bug."""
        # for i in range(len(x)):
        #     if map[x[i], y[i]] == 1:
        #         obsCount += 1
        #         # break # this line is added by us to accelerate the program (need testing)
        # sdist = max(sdist, obsCount * penalty)

        """the original code published on github has some bugs. The collision judgement is fixed by the following code"""
        if not no_obstacle(sp, ep, map):
            sdist = np.inf
        # debug
        # if sdist == float("inf") and no_obstacle(sp, ep, map) or sdist != float("inf") and not no_obstacle(sp, ep, map):
        #     print(sdist, no_obstacle(sp, ep, map))
            # draw_path(map, [calculate_number(sp[0], sp[1]), calculate_number(ep[0], ep[1])])
    return sdist
