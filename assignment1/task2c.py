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
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    kernel_offset_i = kernel.shape[0] // 2
    kernel_offset_j = kernel.shape[1] // 2
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for c in range(im.shape[2]):
                x_offset = x + kernel_offset_i
                y_offset = y + kernel_offset_j
                conv_val = 0
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):
                        if 0 <= x_offset - i < im.shape[0] and 0 <= y_offset - j < im.shape[1]:
                            conv_val += kernel[i, j] * im[x_offset - i, y_offset - j, c]
                im[x][y][c] = conv_val
    return im


if __name__ == "__main__":
    # Read image
    impath = os.path.join("images", "lake.jpg")
    im = plt.imread(impath)

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
    smoothed_im1 = convolve_im(im.copy(), h_a)
    smoothed_im2 = convolve_im(im, h_b)

    save_im("convolved_im_h_a.png", smoothed_im1)
    save_im("convolved_im_h_b.png", smoothed_im2)

    # DO NOT CHANGE
    assert isinstance(smoothed_im1, np.ndarray), \
        "Your convolve function has to return a np.array. " +\
        "Was: {type(smoothed_im1)}"
    assert smoothed_im1.shape == im.shape, \
        "Expected smoothed im ({smoothed_im1.shape}" + \
        "to have same shape as im ({im.shape})"
    assert smoothed_im2.shape == im.shape, \
        "Expected smoothed im ({smoothed_im1.shape}" + \
        "to have same shape as im ({im.shape})"
