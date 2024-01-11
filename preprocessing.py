# ana.rebeka.kamsek@ki.si, 2021

import numpy as np
import cv2


def blur_subtract(image, input_parameter=10):
    """Blurs grayscale image with a box blur and subtracts the blurred image from the original.

    :param image: input image to be filtered
    :param input_parameter: kernel size for blurring
    :return: filtered image
    """

    kernel_size = input_parameter
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred = cv2.filter2D(image, -1, kernel)

    filtered = cv2.subtract(image, blurred)

    return filtered


def fft_step_filtering(image, input_parameter=30):
    """Performs high-pass FFT filtering of a grayscale image using a circular mask.

    :param image: input image to be filtered
    :param input_parameter: ratio of image size versus desired mask size
    :return: filtered image
    """

    mask_ratio = input_parameter

    # calculate the Fourier transform of the image
    masked_fourier = np.fft.fftshift(np.fft.fft2(image))

    # mask the center
    size = int(image.shape[0] / mask_ratio)
    center = int(image.shape[0] / 2)
    y, x = np.ogrid[1: 2 * size + 1, 1:2 * size + 1]

    # create a circular mask
    mask = (x - size) * (x - size) + (y - size) * (y - size) <= size * size
    masked_fourier[center - size:center + size, center - size:center + size] = \
        masked_fourier[center - size:center + size, center - size:center + size] * (1 - mask)

    # transform back to real space
    reconstruction = np.real(np.fft.ifft2(np.fft.ifftshift(masked_fourier)))
    background = np.abs(image - reconstruction)

    filtered = np.subtract(image, background)

    return filtered


def fft_gauss_filtering(image, input_parameter=30):
    """Performs high-pass FFT filtering of a grayscale image using a Gaussian mask.

    :param image: input image to be filtered
    :param input_parameter: ratio of image size versus desired mask size
    :return: filtered image
    """

    mask_ratio = input_parameter

    # calculate the Fourier transform of the image
    gauss_fourier = np.fft.fftshift(np.fft.fft2(image))

    # mask the center
    size = int(image.shape[0] / mask_ratio)
    center = int(image.shape[0] / 2)
    y, x = np.ogrid[1: 2 * size + 1, 1:2 * size + 1]

    # create a 2D Gaussian distribution mask
    gauss = np.sqrt(np.amax(np.abs(gauss_fourier))) * np.exp(- (0.001 * ((x - size) ** 2) + 0.001 * ((y - size) ** 2)))
    gauss_fourier[center - size:center + size, center - size:center + size] = \
        gauss_fourier[center - size:center + size, center - size:center + size] * (-gauss)

    # transform back to real space
    reconstruction = -np.real(np.fft.ifft2(np.fft.fftshift(gauss_fourier)))
    reconstruction /= np.amax(reconstruction)
    image = image / 255.0

    background = np.abs(image - reconstruction)

    filtered = np.subtract(image, background)

    return filtered
