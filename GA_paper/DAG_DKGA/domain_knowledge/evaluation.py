import numpy as np
from sgmp import no_obstacle
import config


def evaluation(segment, MAP, type="fitness"):
    """input: a path or segment
        tem: fitness of a path or segment"""
    distance = 0
    for i in range(len(segment) - 1):
        distance += map_dist(segment[i], segment[i+1], MAP)
    if type == "fitness":
        distance += evaluation_smooth(segment) * config.SCALE_FACTOR
    return distance

def evaluation_smooth(path):
    """evaluate the smoothness
        the result is unstable, a bug may exist"""
    penalty = 0
    # delete duplicated node
    index = 0
    while index < len(path)-1:
        if path[index] == path[index+1]:
            path.pop(index+1)
            index -= 1
        index += 1

    for i in range(len(path) - 2):
        row1 = path[i] % config.MAP_SIZE
        row2 = path[i + 1] % config.MAP_SIZE
        row3 = path[i + 2] % config.MAP_SIZE
        line1 = path[i] // config.MAP_SIZE
        line2 = path[i + 1] // config.MAP_SIZE
        line3 = path[i + 2] // config.MAP_SIZE

        pos_re1 = np.array([row2 - row1, line2 - line1])
        pos_re2 = np.array([row3 - row2, line3 - line2])
        if pos_re2[1] * pos_re1[0] == pos_re1[1] * pos_re2[0]:  # 0 degree, no punishment
            continue
        cosangle = np.dot(pos_re1, pos_re2) / np.linalg.norm(pos_re1) / np.linalg.norm(pos_re2)
        penalty += np.arccos(cosangle)
    return penalty


def bresenham(x1, y1, x2, y2):
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
    return x, y



def map_dist(sp, ep, MAP):

    # A function for calculating the Euclidean distance of <sp, ep>.
    # Please do not care about the "realtime".

    # Because we don't use global variables in Python, we removed the global
    # statements for "mapColSize" and "map".
    n = MAP.shape[0]
    penalty = config.PENALTY
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
        sdist = max(sdist, np.sum(np.diag(MAP[startR:endR + 1, startC:endC + 1])) * penalty)
    elif sp[0] == ep[0]:
        sdist = max(sdist, np.sum(MAP[sp[0], min(sp[1], ep[1]):max(sp[1], ep[1]) + 1]) * penalty)
    elif sp[1] == ep[1]:
        sdist = max(sdist, np.sum(MAP[min(sp[0], ep[0]):max(sp[0], ep[0]) + 1, sp[1]]) * penalty)
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
        if not no_obstacle(sp, ep, MAP):
            sdist = np.inf
        # debug
        # if sdist == float("inf") and no_obstacle(sp, ep, map) or sdist != float("inf") and not no_obstacle(sp, ep, map):
        #     print(sdist, no_obstacle(sp, ep, map))
            # draw_path(map, [calculate_number(sp[0], sp[1]), calculate_number(ep[0], ep[1])])
    return sdist
