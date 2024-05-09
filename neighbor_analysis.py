# ana.rebeka.kamsek@ki.si, 2022

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np


def calculate_neighbors(columns, cutoff=26):
    """Calculates numbers of nearest neighbors for atomic columns based on the previously determined positions.

    :param columns: array of (x, y) positions of atomic columns in pixels
    :param cutoff: estimated average distance in pixels that would pick out the first neighbors, int
    :return: arrays with numbers of first neighbors for each atomic column and their frequencies
    """

    neighbors = []

    for i in range(len(columns)):
        r = []
        for j in range(len(columns)):
            # each column cannot be its own neighbor
            if i == j:
                continue

            # calculate the distances to other columns
            r_squared = (columns[j, 0] - columns[i, 0]) ** 2 + (columns[j, 1] - columns[i, 1]) ** 2
            r.append(np.sqrt(r_squared))
        r.sort()

        # only consider short enough distances
        neighbors.append(sum(1 for item in r if item < cutoff))

    neighbors = np.asarray(neighbors)
    hist = np.histogram(neighbors, bins=[1, 2, 3, 4, 5, 6, 7])

    return neighbors, hist


def find_nearest(all_points, one_point):
    """Helper function to find the nearest element in an array to a specified point.

    :param all_points: array with coordinates of points to be considered
    :param one_point: array with coordinates of a single point
    :return: index and coordinates of the nearest point in an array containing all points
    """

    centered_abs = np.abs(all_points - one_point)
    distances = np.sqrt(centered_abs[:, 0] ** 2 + centered_abs[:, 1] ** 2)
    idx = np.argmin(distances, axis=0)

    return idx, all_points[idx, :]


def correct_neighbors(columns, neighbors):
    """Allows the user to manually select which columns should have six neighbors.

    Displays numbers of nearest neighbors and allows the user to click on them.
    The chosen columns are declared to have six neighbors. Suitable for columns,
    not forming the surface of fcc-structured nanoparticles in the [110] zone axis.

    :param columns: array with (x, y) positions of atomic columns in pixels
    :param neighbors: initial numbers of first neighbors for each atomic column
    :return: arrays with corrected numbers of first neighbors for each column and their frequencies
    """

    coords = []
    custom = ListedColormap(["darkred", "red", "darkorange", "mediumturquoise", "blue", "whitesmoke"])

    # visualize the preliminary results and allow the user to select the incorrect ones
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.scatter(columns[:, 0], columns[:, 1], marker='o', cmap=custom, c=neighbors, edgecolors='gray')
    plt.clim(1, 6)
    plt.xlim(0, 1024)
    plt.ylim(0, 1024)
    plt.gca().invert_yaxis()
    plt.xticks([]), plt.yticks([])
    ax1.set_aspect('equal')
    plt.colorbar()

    def onclick(event):
        x, y = event.xdata, event.ydata
        if event.button == matplotlib.backend_bases.MouseButton.LEFT:
            coords.append((x, y))
            ax1.plot(event.xdata, event.ydata, 'x', color='r')
        else:
            print("Invalid - use the left mouse button.")
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    coords = np.array(coords)

    # determine the indices of the numbers of nearest neighbors that will be changed
    indices = []
    for j in range(len(coords)):
        index, element = find_nearest(columns, coords[j])
        indices.append(index)
    neighbors[indices] = 6

    # determine the frequencies of each possible number of nearest neighbors
    hist = np.histogram(neighbors, bins=[1, 2, 3, 4, 5, 6, 7])

    return neighbors, hist


def display_neighbors(columns, neighbors):
    """Displays positions of atomic columns, color-coded according to their number of nearest neighbors.

    :param columns: array with positions of atomic columns
    :param neighbors: array with numbers of nearest neighbors for each column
    """

    custom = ListedColormap(["darkred", "red", "darkorange", "mediumturquoise", "blue", "whitesmoke"])

    plt.figure()
    ax = plt.subplot(111)

    plt.scatter(columns[:, 0], columns[:, 1], marker='o', cmap=custom, c=neighbors, edgecolors='gray')

    plt.clim(1, 6)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1024)
    plt.ylim(0, 1024)
    plt.xticks([]), plt.yticks([])
    ax.set_aspect('equal')
    plt.colorbar()
    plt.show()
