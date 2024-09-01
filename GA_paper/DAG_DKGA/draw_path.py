import numpy as np
import matplotlib.pyplot as plt

def draw_path(map, path, block=True):
    """
    Draw obtained path on given map.

    Args:
        map (ndarray): A boolean square matrix specifying the given map.
                       "True" means an obstacle node, "False" means a free-space node.
        path (list): An array containing obtained paths.

    Example:
        map = np.array([[0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0]])

        path = [24, 45, 93, 36]

    Note: The values of "path" represent the indices of nodes in the map,
          indicating the path traverses the nodes in the specified order.

    For more detailed information, please visit the homepage:
    http://ai.cau.ac.kr/?f=softwares&m=cave
    """

    colSize = map.shape[1]
    coords = np.zeros((len(path), 2), dtype=int)
    for k in range(len(path)):
        coords[k, 0], coords[k, 1] = index_to_row_col(path[k], colSize)

    plt.figure()
    plt.imshow(map, cmap='gray')
    plt.plot(coords[:, 1], coords[:, 0], 'r-', linewidth=3)
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
