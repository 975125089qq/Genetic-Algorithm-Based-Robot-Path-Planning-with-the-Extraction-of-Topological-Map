import numpy as np
import matplotlib.pyplot as plt

def draw_dag(map, dag, block=True):
    """
    Draw obtained Directed Acyclic Graph by SMGP method.

    Args:
        map (ndarray): A boolean square matrix specifying the given map.
                       "True" means an obstacle node, "False" means a free-space node.
        dag (ndarray): A variable specifying the obtained DAG by SGMP.

    Example:
        map = np.array([[0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0]])

        dag = np.array([[115, 1, 115, 0, 0],
                        [4120, 9, 120, 1, np.nan],
                        [4624, 10, 124, 2, 1],
                        [5625, 12, 125, 3, 2],
                        [4113, 9, 113, 2, 1]])

    Note: The values in "dag" represent the nodes of the directed acyclic graph
          and their parent nodes.

    For simplicity, we assume the given map is a square.
    If the map is a rectangle, it can be easily transformed to a square by filling obstacle nodes.
    The variable "mapRowSize" is not required for further procedure.
    However, for compatibility issues of MATLAB, we specify "mapRowSize" to the output variables in this implementation.
    """
    global bresenham_map
    bresenham_map = map.copy()  # Create a copy of the map for the Bresenham algorithm
    map_row_size, map_col_size = map.shape

    plt.figure()
    for k in range(1, len(dag)):
        # k = 1: starting node. Thus, there is no parent node
        # and the algorithm does not need to draw the edge.

        # The current node is identified.
        # For clarity, the algorithm saves the node to "node".
        node = dag[k][0]

        for each_parent in range(3, len(dag[k])):
            # Identify the parent node's index in the dag structure
            parent = dag[k][each_parent]

            if np.isnan(parent) or parent == 0:
                # If there is no parent, then the algorithm
                # does not need to draw the edge.
                continue

            # The parent node of the "k"-th node is identified.
            # For clarity, save the index of the parent node to "parent_node".
            parent = int(parent)
            parent_node = int(dag[parent][0])

            # Obtain (vertical, horizontal)-coordinate of current node, "node".
            sp_row, sp_col = index_to_row_col(node, map_col_size)

            # Obtain (vertical, horizontal)-coordinate of parent node, "parent_node".
            dp_row, dp_col = index_to_row_col(parent_node, map_col_size)

            # Draw the intermediate nodes between "node" and "parent_node".
            # "bresenham" algorithm returns (x, y)-coordinate pairs
            # between "node" and "parent_node".
            plt.plot([sp_col, dp_col], [sp_row, dp_row], "r-", linewidth=3)

    plt.imshow(bresenham_map, cmap='gray')
    plt.show(block=block)


def index_to_row_col(idx, col_size):
    """
    A function for changing the index of node "idx"
    to vertical and horizontal coordinates.

    Args:
        idx (int): The index of the node.
        col_size (int): The number of columns in the map.

    Returns:
        row (int): The vertical coordinate of the node.
        col (int): The horizontal coordinate of the node.
    """
    row = idx // col_size
    col = idx % col_size
    return row, col