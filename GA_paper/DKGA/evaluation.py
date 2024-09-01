import numpy as np
from functions import calculate_coordinate, detect_obstacle
import config


def evaluation(segment, MAP, type="fitness"):
    """input: a path or segment
        tem: fitness of a path or segment"""
    distance = 0
    rows, lines = [], []
    for i in range(len(segment)):
        row, line = calculate_coordinate(segment[i])
        rows.append(row)
        lines.append(line)
    for i in range(len(segment) - 1):
        distance += np.sqrt((rows[i] - rows[i + 1]) ** 2 + (lines[i] - lines[i + 1]) ** 2)
        # distance += config.PENALTY * tem
    for i in range(len(segment) - 1):
        if not detect_obstacle(segment[i], segment[i + 1], MAP):
            distance += config.PENALTY
            # print("obstacle found")
            # print(distance)
            break
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