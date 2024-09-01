import random
from matplotlib import pyplot as plt

class Graph(object):
    def __init__(self, cost_matrix: list, rank: int):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.rank = rank
        # noinspection PyUnusedLocal
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]


class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int, targets_connection_table):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy
        self.history = []
        self.targets_connection_table = compatibility_targets_connection_table(targets_connection_table)


    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    # noinspection PyProtectedMember
    def solve(self, graph: Graph):
        """
        :param graph:
        """
        best_fitness = float('inf')
        best_solution = []
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            ants = [_Ant(self, graph, self.targets_connection_table) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                # ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_fitness:
                    best_fitness = ant.total_cost
                    best_solution = [] + ant.tabu
                # update pheromone
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_fitness, best_solution))
            self.history.append(best_fitness)

        return best_solution, best_fitness

    def plot_history(self, block=True):
        plt.figure()
        plt.plot(self.history)
        plt.xlabel("iteration")
        plt.ylabel("fitness value")
        plt.title("self_work")
        plt.show(block=block)

class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph, targets_connection_table):
        self.colony = aco
        self.graph = graph
        self.targets_connection_table = targets_connection_table
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = set([i for i in range(graph.rank)])  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information
        start = 0  # start from the beginning point
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        targets_allowed = []
        for i in self.targets_connection_table[self.current]:
            if i in self.allowed:
                denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                                                                                            i] ** self.colony.beta
                targets_allowed.append(i)

        # have targets that have not been visited
        if len(targets_allowed) > 0:
            # noinspection PyUnusedLocal
            probabilities = [0 for i in range(len(targets_allowed))]  # probabilities for moving to a node in the next step
            for index, target in enumerate(targets_allowed):
                if target in self.allowed:
                    probabilities[index] = self.graph.pheromone[self.current][target] ** self.colony.alpha * \
                        self.eta[self.current][target] ** self.colony.beta / denominator
            # select next node by probability roulette
            selected = 0
            rand = random.random()
            for index, probability in enumerate(probabilities):
                rand -= probability
                if rand <= 0:
                    selected = targets_allowed[index]
                    break
            self.allowed.remove(selected)
            self.tabu.append(selected)
            self.total_cost += self.graph.matrix[self.current][selected]
            self.current = selected

        # all nearest targets have been visited
        # """needs improvement, for example using pheromone to decide which path to choose!!!!!!!!"""
        else:
            flag = False
            for target1 in self.targets_connection_table[self.current]:
                if not flag: # no available targets
                    for target2 in self.targets_connection_table[target1]:
                        if target2 in self.allowed:
                            self.allowed.remove(target2)
                            self.tabu.append(target1)
                            self.tabu.append(target2)
                            self.total_cost += self.graph.matrix[self.current][target1]
                            self.total_cost += self.graph.matrix[target1][target2]
                            self.current = target2
                            flag = True
                            break
                else:
                    break

            if not flag:
                # still can not find an available target, choose one randomly
                selected = random.sample(list(self.allowed), 1)[0]
                self.allowed.remove(selected)
                self.tabu.append(selected)
                self.total_cost += self.graph.matrix[self.current][selected]
                self.current = selected


    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost

def compatibility_targets_connection_table(targets_connection_table):
    targets_connection_table_new = [[] for i in range(len(targets_connection_table))]
    for index, targets_list in enumerate(targets_connection_table):
        for target in targets_list:
            targets_connection_table_new[index].append(target + 1)
    return targets_connection_table_new