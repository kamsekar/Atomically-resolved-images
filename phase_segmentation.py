# # ana.rebeka.kamsek@ki.si, 2021

import numpy as np
from sklearn import decomposition


def local_fft_transform(image, use_window_function=True):
    """Computes the fast Fourier transform of the input data.

    :param image: input data
    :param use_window_function: a flag for using a cosine window function on the input data
    :return: Fourier transform result
    """

    # initialize an array for a possible Ham filter
    window_function = np.ones_like(image)

    # create the window function
    if use_window_function:
        window_size = image.shape
        x = np.arange(0, window_size[0])
        y = np.arange(0, window_size[1])

        x, y = np.meshgrid(x, y, sparse=True)
        window_function = (np.sin(x / (window_size[0] - 1) * np.pi) * np.sin(y / (window_size[1] - 1) * np.pi)) ** 2

    # calculate the Fourier transform
    fft_window = np.fft.fft2(image * window_function)
    fft_window = np.fft.fftshift(fft_window, axes=(-1, -2))

    return fft_window


def sliding_function(image, window_size=20, step=5):
    """Scans the input image using a sliding window and computes the fast Fourier transform at every step.

    :param image: input image
    :param window_size: size of the sliding window in px
    :param step: distance between steps in px
    :return: a stack of Fourier transforms and its shape
    """

    # add zero padding to the input image
    padded_array = np.zeros([image.shape[0] + 2 * window_size, image.shape[1] + 2 * window_size])
    padded_array[window_size:-window_size, window_size:-window_size] = image

    x = np.arange(0, image.shape[0], step)
    y = np.arange(0, image.shape[1], step)

    # initialize result
    output_shape = (2 * window_size, 2 * window_size)
    transformed_stack = np.zeros([len(x), len(y), output_shape[0] * output_shape[1]], dtype='complex')

    # sliding across the image
    for i in range(len(x)):
        for j in range(len(y)):
            # define what part of the image is currently considered
            window = padded_array[x[i]:2 * window_size + x[i], y[j]:2 * window_size + y[j]]

            # calculate the Fourier transform of the considered part of the image
            transformed_stack[i, j] = np.ndarray.flatten(local_fft_transform(window))

    return transformed_stack, output_shape


def eigenvector_extraction(image, window_size=20, step=5):
    """Performs Principal Component Analysis (PCA) on a stack of local FFTs from the input image.

    PCA is used to find eigenvectors in the FFT stack, i.e. representative FFT patterns of the dataset.
    From those, abundance maps are created to show where the eigenvectors are present in an image and to what extent.
    Returns eigenvectors, maps, the variance for each number of eigenvectors and a proposed cutoff value
    for the number of components that would retain an acceptable amount of information.
    :param image: input image
    :param window_size: size of the sliding window in px
    :param step: distance between steps in px
    :return: eigenvectors, abundance maps, variance, cutoff value
    """

    # create a stack of Fourier transforms from the input image
    sf_result, transformation_shape = sliding_function(image, window_size, step)

    # perform PCA with a large number of components, which will in any case be ranked by variance
    n_components = 100
    pca = decomposition.PCA(n_components=n_components)
    pcs = pca.fit_transform(np.real(sf_result.reshape(sf_result.shape[0] * sf_result.shape[1], sf_result.shape[2])))

    # save eigenvectors
    eigenvectors = pca.components_.reshape(n_components, transformation_shape[0], transformation_shape[1])

    # save corresponding abundance maps
    temp_1 = pcs.reshape(sf_result.shape[0], sf_result.shape[1], n_components)
    temp_2 = np.swapaxes(temp_1, 0, 2)
    maps = np.swapaxes(temp_2, 1, 2)

    # calculate the variance to explain the retained information for different numbers of components
    variance = np.cumsum(pca.explained_variance_ratio_)

    # propose a cutoff value where the variance curve exhibits an elbow
    slope = np.diff(variance)
    ratio = np.divide(slope, variance[:-1])
    cutoff = np.amax(np.argwhere(ratio > np.amax(slope) * 0.05))

    return eigenvectors, maps, variance, cutoff
