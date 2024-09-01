import math
from . import config
class Node_theta(object):
    """
    f(n) ... startからgoalまでの最短距離
    g(n) ... startからnノードまでの最短距離
    h(n) ... nノードからgoalまでの最短距離
    f(n) = g(n) + h(n)

    関数を推定値にすることにより最短距離を予測する
    h*(n)をnからgoalまでの直線距離と仮定する。

    f*(n) = g*(n) + h*(n)
    """
    start = None    #start位置(x,y)
    goal = None     #goal位置(x,y)

    def __init__(self,x,y):
        self.pos    = (x,y)
        self.hs     = math.sqrt((x-self.goal[0])**2 + (y-self.goal[1])**2)
        self.fs     = float("Inf")
        self.owner_list  = None
        self.parent_node = None

    def isGoal(self):
        return self.goal == self.pos

class Node_A(object):
    """
    f(n) ... startからgoalまでの最短距離
    g(n) ... startからnノードまでの最短距離
    h(n) ... nノードからgoalまでの最短距離
    f(n) = g(n) + h(n)

    関数を推定値にすることにより最短距離を予測する
    h*(n)をnからgoalまでの直線距離と仮定する。

    f*(n) = g*(n) + h*(n)
    """
    start = None  # start位置(x,y)
    goal = None  # goal位置(x,y)

    def __init__(self, x, y):
        self.pos = (x, y)
        self.hs = math.sqrt((x - self.goal[0]) ** 2 + (y - self.goal[1]) ** 2) * config.SCALE_FACTOR
        self.fs = [float("Inf") for _ in range(8)]  # eight directions
        self.parent_node = [None for _ in range(8)]  # eight directions

    def isGoal(self):
        return self.goal == self.pos