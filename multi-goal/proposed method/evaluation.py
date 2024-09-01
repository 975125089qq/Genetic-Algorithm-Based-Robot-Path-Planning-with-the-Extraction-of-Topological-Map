import numpy as np
from functions import no_obstacle
import config


def evaluation(segment, MAP, type="fitness"):
    """input: a path or segment
        tem: fitness of a path or segment"""
    distance = 0
    rows, lines = [], []
    for i in range(len(segment)):
        row, line = segment[i]
        rows.append(row)
        lines.append(line)
    for i in range(len(segment) - 1):
        distance += np.sqrt((rows[i] - rows[i + 1]) ** 2 + (lines[i] - lines[i + 1]) ** 2) / config.TIMES
        # distance += config.PENALTY * tem
    for i in range(len(segment) - 1):
        if not no_obstacle(segment[i], segment[i + 1], MAP):
            distance += config.PENALTY
            # print("obstacle found")
            # print(distance)
            break
    if type == "fitness":
        distance += evaluation_smooth(segment)
    return distance

# tested
def penalty_cal(row1, row2, line1, line2, MAP):
    obstacle_num = 0
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
                    obstacle_num += MAP[row_current][j]
        else:
            for i in range(len(rows) - 1):
                line_1_int = int(lines[i] - EPS)
                line_2_int = int(lines[i + 1] + EPS)
                row_current = int(rows[i] + EPS)
                for j in range(line_1_int, line_2_int - 1, -1):
                    obstacle_num += MAP[row_current][j]
    else:
        line_direction = 1 if line1 < line2 else -1
        for i in range(line1, line2, line_direction):
            obstacle_num += MAP[row1][i]
    obstacle_num -= MAP[row2][line2]
    return obstacle_num


## multi targets self work
def evaluation_segments(target_segments, MAP):
    """ calculate the fitness of segments
        input: path_segments {targets:[[segment]]}
        ouput: dictionary {(targets):[fitness....]}"""
    fitness_segments = {}
    for i in target_segments.keys():
        for j in target_segments[i]:
            fitness_tem = evaluation(j, MAP)
            if i in fitness_segments:
                fitness_segments[i].append(fitness_tem)
            else:
                fitness_segments[i] = [fitness_tem]
    return fitness_segments


def evaluation_order(path_single, fitness_segments):
    """for each order of targets, return the smallest fitness found so far"""
    fitness_path = 0
    for j in range(len(path_single) - 1):
        two_targets = (path_single[j], path_single[j + 1]) if path_single[j] < path_single[j + 1] else (
            path_single[j + 1], path_single[j])
        fitness_path += min(fitness_segments[two_targets])
    return fitness_path

# def evaluation_order(path_single, fitness_segments):
#     """for each order of targets, return the smallest fitness found so far"""
#     fitness_path = 0
#     for j in range(len(path_single) - 1):
#         x1, y1 = calculate_coordinate(path_single[j])
#         x2, y2 = calculate_coordinate(path_single[j+1])
#         fitness_path += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
#     return fitness_path

def evaluation_smooth(path):
    """evaluate the smoothness
        the result is unstable, a bug may exist"""
    penalty = 0
    # delete duplicated node
    index = 0
    while index < len(path)-1:
        if path[index] == path[index+1]:
            path.pop(index+1)
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
        if np.all(pos_re1==0) or np.all(pos_re2==0):
            continue
        cosangle = np.dot(pos_re1, pos_re2) / np.linalg.norm(pos_re1) / np.linalg.norm(pos_re2)
        # angle = np.arccos(min(1,cosangle))
        if cosangle < -0.501:  # 135 degrees
            penalty += 12
        elif cosangle < 0.01:  # 90 degrees
            penalty += 9
        elif cosangle < 0.501:  # 45 degrees (for 45 degrees)
            penalty += 6
        elif cosangle < 0.87:
            penalty += 3
    return penalty