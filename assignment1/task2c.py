import numpy as np
import os
import matplotlib.pyplot as plt
from task2ab import save_im, greyscale


def convolve_im(im, kernel):
    """ A function that convolves im with kernel
    
    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]
    
    Returns:
        [type]: [np.array of shape [H, W, 3]]
    """
    kernel_offset = kernel.shape[0] // 2
    im = np.pad(im, ((kernel_offset, kernel_offset), (kernel_offset, kernel_offset), (0, 0)), mode="constant")
    for x in range(kernel_offset, im.shape[0] - kernel_offset):
        for y in range(kernel_offset, im.shape[1] - kernel_offset):
            for c in range(im.shape[2]):
                x_offset = x + kernel_offset
                y_offset = y + kernel_offset
                conv_val = 0
                for i in range(kernel.shape[0]):
                    conv_val += np.dot(kernel[i], im[x_offset - i, y_offset - kernel.shape[0] + 1:y_offset + 1, c][::-1])
                im[x][y][c] = conv_val
    return im[kernel_offset:-kernel_offset, kernel_offset:-kernel_offset]


if __name__ == "__main__":
    # Read image
    impath = os.path.join("images", "lake.jpg")
    lake_im = plt.imread(impath)

    # Define the convolutional kernels
    h_a = np.ones((3, 3)) / 9
    h_b = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    # Convolve images
    smoothed_im1 = convolve_im(lake_im.copy(), h_a)
    smoothed_im2 = convolve_im(lake_im, h_b)

    save_im("convolved_im_h_a.png", smoothed_im1)
    save_im("convolved_im_h_b.png", smoothed_im2)

    # DO NOT CHANGE
    assert isinstance(smoothed_im1, np.ndarray), \
        "Your convolve function has to return a np.array. " +\
        "Was: {type(smoothed_im1)}"
    assert smoothed_im1.shape == lake_im.shape, \
        "Expected smoothed lake_im ({smoothed_im1.shape}" + \
        "to have same shape as lake_im ({lake_im.shape})"
    assert smoothed_im2.shape == lake_im.shape, \
        "Expected smoothed lake_im ({smoothed_im1.shape}" + \
        "to have same shape as lake_im ({lake_im.shape})"
