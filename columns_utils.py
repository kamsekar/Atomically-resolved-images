# ana.rebeka.kamsek@ki.si, 2022

import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def create_snippet(image, columns):
    """Crops an image to a snippet using coordinates of two user-defined points and offsets column positions.

    Displays the original image and allows the user to click on it. Coordinates of the two
    user-determined points are used to crop the image. Atomic column positions are offset
    by the upper left corner coordinates of the snippet. Returns the snippet and adjusted positions.
    :param image: original image, 2D array
    :param columns: array with (x, y) positions of atomic columns in pixels
    :return: cropped part of the original image (2D array) and array with translated column positions
    """

    bounding_box = []
    print("Click the upper left and lower right bounding box corner to crop the image.")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(image, cmap='gray'), plt.xticks([]), plt.yticks([])

    def onclick(event):
        x, y = event.xdata, event.ydata
        bounding_box.append((x, y))
        ax1.plot(event.xdata, event.ydata, 'x', color='r')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    bounding_box = np.array(bounding_box)

    # define the cropped part of the image and offset the positions of the atomic columns
    snippet = image[int(bounding_box[0, 1]):int(bounding_box[1, 1]), int(bounding_box[0, 0]):int(bounding_box[1, 0])]
    columns -= bounding_box[0, :]

    return snippet, columns


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


def edit_coordinates(image, columns, upper_left=(0, 0)):
    """Allows the user to manually add or delete incorrectly determined atomic columns.

    Displays the original image along with previously determined positions of atomic columns.
    Each left click saves coordinates that should be added to the column positions, and
    each right click saves coordinates that should be deleted from them. Returns the corrected positions.
    :param image: original atomically resolved image or its snippet, 2D array
    :param columns: array with previously determined positions of atomic columns
    :param upper_left: array with coordinates of the upper-left corner of the cropped image
    :return: array with corrected positions of atomic columns
    """

    to_add, to_delete = [], []
    print("Left-click to add a column, right-click to remove one.")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(image, cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.plot(columns[:, 0], columns[:, 1], '.', c='#0A3976')

    def onclick(event):
        x, y = event.xdata, event.ydata

        if event.button == matplotlib.backend_bases.MouseButton.RIGHT:
            to_delete.append((x, y))
            ax1.plot(event.xdata, event.ydata, 'o', color='r')
        elif event.button == matplotlib.backend_bases.MouseButton.LEFT:
            to_add.append((x, y))
            ax1.plot(event.xdata, event.ydata, 'o', color='g')

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    # add selected positions to the existing array
    to_add = np.array(to_add)
    try:
        columns = np.concatenate((columns, to_add))
    except ValueError:
        print("Error when appending new coordinates (or none were found).")

    # delete selected positions from the existing array
    to_delete = np.array(to_delete)
    indices = []
    for j in range(len(to_delete)):
        index, element = find_nearest(columns, to_delete[j])
        indices.append(index)
    columns = np.delete(columns, indices, axis=0)

    # offset the column positions (in case of working with a snippet) to match them to the original image
    columns += upper_left

    return columns


def display_positions(image, columns):
    """Displays positions of atomic columns along with the original image.

    :param image: original image, 2D array
    :param columns: array with positions of atomic columns
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_shape = image.shape

    plt.figure()
    ax = plt.subplot(111)

    ax.imshow(image_rgb, cmap="gray")
    ax.plot(columns[:, 0], columns[:, 1], '.', c='#0A3976')

    plt.gca().invert_yaxis()
    plt.xlim(0, image_shape[0])
    plt.ylim(0, image_shape[1])
    plt.xticks([]), plt.yticks([])
    ax.set_aspect('equal')
    plt.show()
