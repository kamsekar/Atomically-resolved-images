# ana.rebeka.kamsek@ki.si, 2021

import numpy as np
import cv2
import scipy.optimize as opt
from preprocessing import blur_subtract


def positions_centers_of_mass(filtered_image):
    """Finds approximate positions of atomic columns in atomically resolved HAADF-STEM images.

    The input image, ideally previously filtered, is thresholded to reveal only contours which represent
    atomic columns. Positions of atomic columns are then computed as centers of mass of the contours.
    :param filtered_image: filtered image
    :return: coordinates of approximate column positions
    """

    # a small box blur to smoothen the image before thresholding
    kernel = np.ones((5, 5), np.float32) / (5 * 5)
    dst = cv2.filter2D(filtered_image, -1, kernel)

    # thresholding to yield a black-and-white image
    ret, thresh = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal, erosion removes boundary pixels
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # find contours in the image and make masks from them
    (contours, _) = cv2.findContours(opening.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    points = []
    for contour in contours:
        # compute the center of the contour
        moments = cv2.moments(contour)
        if moments["m00"] == 0:  # skip empty contours
            continue
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        points.append((centroid_x, centroid_y))

    points = np.asarray(points)

    return points


def gaussian_2dim(xdata_tuple, xo, yo, sigma_x, sigma_y, theta, offset, amplitude):
    """Two-dimensional asymmetric/bivariate Gaussian distribution.

    :param xdata_tuple: data to be used for the distribution
    :param xo: x coordinate of the center
    :param yo: y coordinate of the center
    :param sigma_x: spread in x direction
    :param sigma_y: spread in y direction
    :param theta: rotation angle
    :param offset: displacement between the distribution and the zero point
    :param amplitude: amplitude/height
    :return: flattened 2D Gaussian distribution
    """

    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)

    # compute standard coefficients from the input parameters
    alpha = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    beta = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    gamma = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)

    # Gaussian distribution formula
    g = offset + \
        amplitude * np.exp(- (alpha * ((x - xo) ** 2) + 2 * beta * (x - xo) * (y - yo) + gamma * ((y - yo) ** 2)))

    return g.ravel()


def positions_2d_gaussian(image, initial_positions, lattice_parameter=14, radius=0.1):
    """Refines initial positions by fitting asymmetric 2D Gaussians on atomic columns.

    Loops over small areas of filtered image belonging to atomic columns and performs a least-squares
    fitting routine. Skips 1-2 % of columns requiring too much computing time.
    :param image: input image
    :param initial_positions: coordinates of approximate column positions
    :param lattice_parameter: approximate distance between two columns in pixels
    :param radius: parameter to regulate spread of Gaussian curve
    :return: refined coordinates of column positions
    """

    final_positions = []

    for idx in range(0, initial_positions.shape[0]):
        # consider one atomic column at a time
        x_0, y_0 = initial_positions[idx]

        # preprocess intensities of an individual atomic column
        data = blur_subtract(image)[int(x_0 - radius * lattice_parameter):int(x_0 + radius * lattice_parameter),
                                    int(y_0 - radius * lattice_parameter):int(y_0 + radius * lattice_parameter)]
        data_shape = data.shape
        data = data.ravel()

        # rough estimate for fitting parameters to help speed up fitting routine
        initial_guess = (int(data_shape[0] / 2), int(data_shape[1] / 2),
                         0.5 * radius * lattice_parameter, 0.5 * radius * lattice_parameter, 1, 10, 100)

        x = np.linspace(0, data_shape[0] - 1, data_shape[0])
        y = np.linspace(0, data_shape[0] - 1, data_shape[0])
        x, y = np.meshgrid(x, y)

        # avoid a narrow elliptical Gaussian when two neighboring columns are equally as prominent
        sigma_min = 0.3 * radius * lattice_parameter
        sigma_max = 0.7 * radius * lattice_parameter

        # fit a 2D Gaussian distribution
        try:
            popt, pcov = opt.curve_fit(gaussian_2dim, (x, y), data, p0=initial_guess, maxfev=15000,
                                       bounds=([0, 0, sigma_min, sigma_min, 0, 0, 0],
                                               [data_shape[0], data_shape[1], sigma_max, sigma_max, 10, 256, 256]))
        except RuntimeError:
            popt = initial_guess
        a, b, c, d, e, f, g = popt

        final_positions.append((x_0 - radius * lattice_parameter + a, int(y_0 - radius * lattice_parameter) + b))

    final_positions = np.asarray(final_positions)
    final_positions = np.reshape(final_positions, (initial_positions.shape[0], 2))

    return final_positions
