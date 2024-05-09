# A script to analyze previously determined positions of atomic columns. The positions are used as data points
# for a Voronoi diagram. The resulting cells are visualized and colorized according to their area.
# An input image, atomic column positions, and a real-space calibration value are needed.
# ana.rebeka.kamsek@ki.si, 2023

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
import matplotlib


def get_average_area(areas_array):
    """
    Visualize a histogram of cell areas and click on the desired value for normalization.

    Parameters
    ----------
    areas_array : numpy array
        Array with area values for the computed Voronoi cells.

    Returns
    -------
    avg_area : numpy array
        A value to be later used for area normalization.

    """

    fig_hist = plt.figure()
    plt.hist(areas_array, bins=50)
    avg_area = []

    def onclick(event):
        if event.button == matplotlib.backend_bases.MouseButton.LEFT:
            avg_area.append(event.xdata)
            print("Determined value:", f'{event.xdata:.4f}', "nm^2")
            plt.plot(event.xdata, event.ydata, 'o', color='r')

        fig_hist.canvas.draw()

    fig_hist.canvas.mpl_connect('button_press_event', onclick)
    plt.xlabel(r"cell area in nm$^2$")
    plt.ylabel("number of cells")
    plt.show()

    avg_area = np.array(avg_area)

    return avg_area


# input_data
image_path = r"\path_to_image.tif"  # original image
coords_path = r"\path_to_column_positions.txt"  # x and y coordinates of atomic columns
calib = 0.014922  # nm-to-px ratio

# read all files
image = cv2.imread(image_path, 0)
calibration = pow(calib, 2)

points = []
with open(coords_path, "r") as f:
    for line in f:
        x, y = line.split()
        points.append((float(x), float(y)))
f.close()
points = np.asarray(points)

# plot the image with overlaid positions of atomic columns
plt.figure()
plt.imshow(image, cmap='gray')
plt.scatter(points[:, 0], points[:, 1], c='red', s=2)
plt.xticks([]), plt.yticks([])
plt.show()

# add 4 distant dummy points to deal with infinite regions in the Voronoi diagram
limit = image.shape[0] * 2
points = np.append(points, [[limit, limit], [-limit, limit], [limit, -limit], [-limit, -limit]], axis=0)

# compute the Voronoi diagram from atomic column positions
vor = Voronoi(points)

# areas of convex Voronoi cells are computed by computing ConvexHull of each cell
areas = np.zeros(vor.npoints)
for i, reg_num in enumerate(vor.point_region):
    indices = vor.regions[reg_num]
    if -1 in indices:
        areas[i] = np.inf
    else:
        areas[i] = ConvexHull(vor.vertices[indices]).volume

# replace very large areas that span to the image limits with a constant value (manual input needed)
replacement = 350
areas = np.where(areas > replacement, replacement, areas)

# calibrate the values in real space
areas_in_nm = areas * calibration

# determine the normalization area value using a histogram of all areas
average_area = get_average_area(areas_in_nm)
areas_normalized = areas_in_nm / average_area

# alternatively: normalize the areas to any constant value
# areas_normalized = areas_in_nm / 0.048866

# determine the maximum deviations among areas, or set a value manually for a better visualization later
# deviation = max(np.abs(min(areas_normalized) - 1), np.abs(max(areas_normalized) - 1))
deviation = 0.1

# plot the image with Voronoi cells colored according to their area
fig = voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_alpha=0)
colormap = matplotlib.colormaps['RdBu']

plt.gca().set_aspect('equal')
plt.xlim([0, image.shape[1]]), plt.ylim([image.shape[0], 0])
plt.xticks([]), plt.yticks([])

# plot each Voronoi cell as a filled polygon
for i, region in enumerate(vor.regions):
    if -1 not in region:
        polygon = [vor.vertices[j] for j in region]
        if len(polygon) > 0:
            # calculate polygon area using the shoelace formula
            coordinates = np.asarray(polygon)
            x, y = coordinates[:, 0], coordinates[:, 1]

            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            calib_area = area * calibration
            norm_area = calib_area / average_area
            color = (norm_area - (1 - deviation)) / (1 + deviation - (1 - deviation))

            try:
                plt.fill(*zip(*polygon), alpha=1, color=colormap(color))
            except IndexError:
                print("An IndexError occurred, check how the data is shaped.")

# create a color scale to go with the image
c_plot = plt.imshow(np.meshgrid(np.linspace(1 - deviation, 1 + deviation, image.shape[0]),
                                np.linspace(1 - deviation, 1 + deviation, image.shape[1]))[0], cmap='RdBu', alpha=1)
cbar = fig.colorbar(c_plot)
cbar.ax.set_ylabel('relative area')

plt.show()
