# ana.rebeka.kamsek@ki.si, 2021

from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import statistics
import matplotlib.pyplot as plt
from preprocessing import blur_subtract, fft_step_filtering
from scipy.signal import argrelextrema


def rotate(image, angle, center=None):
    """A function to rotate an image by a specified angle.

    :param image: input image
    :param angle: rotation angle
    :param center: coordinates of the rotation axis
    :return: rotated image
    """

    height, width = image.shape[:2]

    # if a rotation axis is not specified, rotate around the image center
    if center is None:
        center = (width / 2, height / 2)

    # perform the rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated


def mse(image1, image2):
    """Computes the mean squared error between two images.

    :param image1: first image
    :param image2: second image
    :return: mean squared error value
    """

    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])

    return err


def zncc(image1, image2):
    """Computes the zero mean normalized cross correlation between two images.

    :param image1: first image
    :param image2: second image
    :return: zero mean normalized cross correlation value
    """

    image1 = image1.astype("float")
    image2 = image2.astype("float")

    err = np.sum((image1 - np.mean(image1)) * (image2 - np.mean(image2)))
    err /= float(image1.shape[0] * image2.shape[1] * np.std(image1) * np.std(image2))

    return err


def compute_metrics(image1, image2):
    """Evaluates three different metrics to compare two input images.

    :param image1: first image
    :param image2: second image
    :return: MSE, SSIM, and ZNCC values

    """

    m = mse(image1, image2)
    s = ssim(image1, image2)
    z = zncc(image1, image2)

    return m, s, z


def compare_rotated_images(image1, image2, center=None):
    """Determines the best rotation angle between two images using three different metrics.

    :param image1: first image
    :param image2: second image
    :param center: coordinates of the rotation axis
    :return: chosen rotation angles according to three metrics
    """

    # initialize metric values and angles
    m_min, s_max, z_max = 10000, -1, -1
    best_angle_m, best_angle_s, best_angle_z = 0, 0, 0

    # rotate one image with respect to another in a (-15°, 15°) angle range by 0.1°
    # evaluate MSE, SSIM, and ZNCC for every rotation angle and update their scores
    for i in range(-150, 150, 1):
        rotated = rotate(image2, angle=0.1 * i, center=center)
        m, s, z = compute_metrics(rotated, image1)

        if m <= m_min:
            m_min = m
            best_angle_m = 0.1 * i
        if s >= s_max:
            s_max = s
            best_angle_s = 0.1 * i
        if z >= z_max:
            z_max = z
            best_angle_z = 0.1 * i

    return best_angle_m, best_angle_s, best_angle_z


def align_centers_of_mass(image1, image2):
    """Calculates centers of mass for both input images and their difference.

    :param image1: first image
    :param image2: second image
    :return: center of mass of the first image and the center of mass difference
    """

    # calculate moments of both images
    moments1 = cv2.moments(image1)
    moments2 = cv2.moments(image2)

    # calculate x,y coordinates of their centers of mass
    c_x = int(moments1["m10"] / moments1["m00"])
    c_y = int(moments1["m01"] / moments1["m00"])
    c_x2 = int(moments2["m10"] / moments2["m00"])
    c_y2 = int(moments2["m01"] / moments2["m00"])

    return (c_x, c_y), c_x - c_x2, c_y - c_y2


def auto_alignment(image_b, image_a):
    """Automatic three-step alignment of two identical location HAADF-STEM images.

    The alignment consists of three steps. First, centers of mass of both images are aligned. Second,
    the angle of rotation between them is determined. Lastly, an area in the middle of the image is
    aligned using centers of mass again to avoid the columns being slightly misaligned.
    :param image_b: first or "before" identical location image
    :param image_a: second or "after" identical location image
    :return: alignment parameters and the difference between input images
    """

    # define area of the image
    image_size = image_b.shape
    area_size = int(np.ceil(0.05 * np.asarray(image_size[0])))

    # compensate for stretching with zero padding
    pad = int((image_size[0] - image_a.shape[0]) / 2)
    after_image_new = cv2.copyMakeBorder(image_a, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT)

    # first translation with aligning centers of mass
    coords_b, x_shift, y_shift = align_centers_of_mass(image_b, after_image_new)

    # create a translation matrix and use it on the after image
    T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_a = cv2.warpAffine(after_image_new, T, (image_size[1], image_size[0]))

    # determining the angle of rotation between the first and the transformed second image
    b1, b2, b3 = compare_rotated_images(image_b, translated_a, center=coords_b)
    b4, b5, b6 = compare_rotated_images(blur_subtract(image_b), blur_subtract(translated_a), center=coords_b)
    b7, b8, b9 = compare_rotated_images(fft_step_filtering(image_b), fft_step_filtering(translated_a), center=coords_b)

    # choose the angle value among the reported ones
    try:
        angle = statistics.mode([b1, b2, b3, b4, b5, b6, b7, b8, b9])
    except statistics.StatisticsError:
        angle = np.median((b1, b2, b3, b4, b5, b6, b7, b8, b9))

    # rotation by the chosen angle
    rotated_after = rotate(translated_a, angle=angle, center=coords_b)

    # align centers of mass of an area in the middle of the images to avoid the columns being slightly disaligned
    middle_point = int(image_size[0] / 2)
    middle_area_before = image_b[middle_point - area_size:middle_point + area_size,
                                 middle_point - area_size:middle_point + area_size]
    middle_area_after = rotated_after[middle_point - area_size:middle_point + area_size,
                                      middle_point - area_size:middle_point + area_size]
    coords_second, x_shift_2, y_shift_2 = align_centers_of_mass(middle_area_before, middle_area_after)

    # create another translation matrix and use it on the rotated after image
    T = np.float32([[1, 0, x_shift_2], [0, 1, y_shift_2]])
    result_image = cv2.warpAffine(rotated_after, T, rotated_after.shape)

    # calculate the difference between the aligned images
    difference_image = cv2.subtract(result_image, image_b) + cv2.subtract(image_b, result_image)

    return x_shift, y_shift, angle, x_shift_2, y_shift_2, difference_image


def get_coordinates(image1, image2):
    """Allow the user to determine a common point in two images.

    Displays both input images side by side and allows the user to click on them. The user should click
    once on each image to signify the position of a common point to be used for alignment. Returns the
    coordinates of the chosen common point.
    :param image1: first image
    :param image2: second image
    :return: coordinates of the chosen point
    """

    coords = []

    # create figure with both input images as subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(image1, cmap='gray'), plt.xticks([]), plt.yticks([])
    ax2 = fig.add_subplot(122)
    ax2.imshow(image2, cmap='gray'), plt.xticks([]), plt.yticks([])

    def onclick(event):
        # get the x and y pixel coords upon clicking
        x, y = event.xdata, event.ydata
        coords.append((x, y))

        # plot the chosen point on the image after clicking
        if event.inaxes is ax1:
            ax1.plot(event.xdata, event.ydata, 'o', color='r')
        elif event.inaxes is ax2:
            ax2.plot(event.xdata, event.ydata, 'o', color='r')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    return np.array(coords)


def semi_manual_alignment(image_b, image_a):
    """A semi-manual three-step alignment of two identical location HAADF-STEM images.

    The alignment consists of three steps. First, centers of mass of both images are aligned by user input.
    Second, the angle of rotation between them is determined. Lastly, an area in the middle of the image is
    aligned using centers of mass again to avoid the columns being slightly misaligned.
    :param image_b: first or "before" identical location image
    :param image_a: second or "after" identical location image
    :return: alignment parameters and the difference between input images
    """

    # define area of the image
    image_size = image_b.shape
    area_size = int(np.ceil(0.05 * np.asarray(image_size[0])))

    # compensate for stretching with zero padding
    pad = int((image_size[0] - image_a.shape[0]) / 2)
    after_image_new = cv2.copyMakeBorder(image_a, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT)

    # obtain user input for two sites that should align
    coordinates = get_coordinates(image_b, after_image_new)

    coords_before = np.asarray((coordinates[0, 0], coordinates[0, 1]))
    coords_after = np.asarray((coordinates[1, 0], coordinates[1, 1]))

    # convert to integers so they can be used as indices
    coords_b = (int(coords_before[0]), int(coords_before[1]))
    coords_a = (int(coords_after[0]), int(coords_after[1]))

    # calculate the values for translation
    x_shift = coords_b[0] - coords_a[0]
    y_shift = coords_b[1] - coords_a[1]

    # create a translation matrix and use it on the after image
    T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_a = cv2.warpAffine(after_image_new, T, (image_size[1], image_size[0]))

    # determining the angle of rotation between the first and the transformed second image
    b1, b2, b3 = compare_rotated_images(image_b, translated_a, center=coords_b)
    b4, b5, b6 = compare_rotated_images(blur_subtract(image_b), blur_subtract(translated_a), center=coords_b)
    b7, b8, b9 = compare_rotated_images(fft_step_filtering(image_b), fft_step_filtering(translated_a), center=coords_b)

    # choose the angle value among the reported ones
    try:
        angle = statistics.mode([b1, b2, b3, b4, b5, b6, b7, b8, b9])
    except statistics.StatisticsError:
        angle = np.median((b1, b2, b3, b4, b5, b6, b7, b8, b9))

    # rotation by the chosen angle
    rotated_after = rotate(translated_a, angle=angle, center=coords_b)

    # align centers of mass of an area in the middle of the images to avoid the columns being slightly disaligned
    middle_point = int(image_size[0] / 2)
    middle_area_before = image_b[middle_point - area_size:middle_point + area_size,
                                 middle_point - area_size:middle_point + area_size]
    middle_area_after = rotated_after[middle_point - area_size:middle_point + area_size,
                                      middle_point - area_size:middle_point + area_size]
    coords_second, x_shift_2, y_shift_2 = align_centers_of_mass(middle_area_before, middle_area_after)

    # create another translation matrix and use it on the rotated after image
    T = np.float32([[1, 0, x_shift_2], [0, 1, y_shift_2]])
    result_image = cv2.warpAffine(rotated_after, T, rotated_after.shape)

    # calculate the difference between the aligned images
    difference_image = cv2.subtract(result_image, image_b) + cv2.subtract(image_b, result_image)

    return x_shift, y_shift, angle, x_shift_2, y_shift_2, difference_image


def stretching(image_b, image_a):
    """Computes the ratio between pixel sizes of two images using the line profiles from their Fourier transforms.

    The function computes the Fourier transforms, their angular averages, line profiles through their centers,
    and then determines the 1st order maxima in both line profiles. The ratio between pixel sizes is computed
    as the ratio between the distances from the zero-order to the 1st order maxima.
    :param image_b: first or "before" identical location image
    :param image_a: second or "after" identical location image
    :return: ratio between the pixel sizes of both input images
    """

    # calculate the Fourier transforms of both input images
    before_fourier = np.log(abs(np.fft.fftshift(np.fft.fft2(image_b))))
    after_fourier = np.log(abs(np.fft.fftshift(np.fft.fft2(image_a))))

    # filter the Fourier transforms to enhance lower-intensity maxima
    before_fourier = blur_subtract(before_fourier, input_parameter=30)
    after_fourier = blur_subtract(after_fourier, input_parameter=30)

    # use only the low-frequency part of the Fourier transforms
    size = image_b.shape[0]
    before_fourier_center = before_fourier[int(0.375 * size):int(0.625 * size), int(0.375 * size):int(0.625 * size)]
    after_fourier_center = after_fourier[int(0.375 * size):int(0.625 * size), int(0.375 * size):int(0.625 * size)]

    # upsampling to improve result accuracy
    dim = (4 * before_fourier_center.shape[0], 4 * before_fourier_center.shape[1])
    before_fourier_center = cv2.resize(before_fourier_center, dsize=dim)
    after_fourier_center = cv2.resize(after_fourier_center, dsize=dim)

    # rotate the Fourier transforms to yield a stack
    before_stack, after_stack = [], []
    for i in range(-900, 900, 10):
        rotated_before = rotate(before_fourier_center, angle=0.1 * i)
        before_stack.append(rotated_before)
        rotated_after = rotate(after_fourier_center, angle=0.1 * i)
        after_stack.append(rotated_after)

    before_stack = np.asarray(before_stack)
    after_stack = np.asarray(after_stack)

    # compute the stack averages
    average_before_fourier = np.sum(before_stack, axis=0)
    average_after_fourier = np.sum(after_stack, axis=0)

    # compute the line profile through the center of the patterns
    line_profile_before = average_before_fourier[:, int(0.5 * before_fourier_center.shape[0])]
    line_profile_after = average_after_fourier[:, int(0.5 * before_fourier_center.shape[0])]

    # smoothen the line profiles
    w = 7
    line_before_smooth_whole = np.convolve(line_profile_before, np.ones(w), 'valid') / w
    line_after_smooth_whole = np.convolve(line_profile_after, np.ones(w), 'valid') / w

    # determine the 1st order maxima in both line profiles
    line_length = len(line_before_smooth_whole)
    line_before_smooth = line_before_smooth_whole.copy()[int(0.5 * line_length):]
    line_after_smooth = line_after_smooth_whole.copy()[int(0.5 * line_length):]
    average_before = np.average(line_before_smooth)
    average_after = np.average(line_after_smooth)

    for i in range(len(line_before_smooth)):
        if line_before_smooth[i] < average_before:
            line_before_smooth[i] = 0
        if line_after_smooth[i] < average_after:
            line_after_smooth[i] = 0

    maxima_before = np.asarray(argrelextrema(line_before_smooth, np.greater)).T
    maxima_after = np.asarray(argrelextrema(line_after_smooth, np.greater)).T

    # compute the ratio between the distances to the 1st order maxima
    ratio = maxima_before[1] / maxima_after[1]

    return ratio
