import numpy as np
from common import config


def evaluation(segment, type="length"):
    """input: a path or segment
        tem: fitness of a path or segment"""
    distance = 0
    rows, lines = [], []
    for i in range(len(segment)):
        row, line = segment[i]
        rows.append(row)
        lines.append(line)
    for i in range(len(segment) - 1):
        distance += np.sqrt((rows[i] - rows[i + 1]) ** 2 + (lines[i] - lines[i + 1]) ** 2)
    if type != "length":
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
        row1 = path[i][0]
        row2 = path[i+1][0]
        row3 = path[i+2][0]
        line1 = path[i][1]
        line2 = path[i+1][1]
        line3 = path[i+2][1]

        pos_re1 = np.array([row2 - row1, line2 - line1])
        pos_re2 = np.array([row3 - row2, line3 - line2])
        if pos_re2[1] * pos_re1[0] == pos_re1[1] * pos_re2[0]:  # 0 degree, no punishment
            continue
        cosangle = np.dot(pos_re1, pos_re2) / np.linalg.norm(pos_re1) / np.linalg.norm(pos_re2)
        penalty += np.arccos(cosangle)
    return penalty

# used in A*_8 to calculate the smoothness of each node
def cal_smo(pos_re1, pos_re2):
    penalty = 0
    if pos_re2[1] * pos_re1[0] == pos_re1[1] * pos_re2[0]:  # 0 degree, no punishment
        return penalty
    cosangle = np.dot(pos_re1, pos_re2) / np.linalg.norm(pos_re1) / np.linalg.norm(pos_re2)
    penalty += np.arccos(cosangle) * config.SCALE_FACTOR
    return penalty
