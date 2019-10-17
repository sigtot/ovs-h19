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
    K = kernel.shape[0]
    kernel_offset = K // 2
    im = np.pad(im, ((kernel_offset, kernel_offset), (kernel_offset, kernel_offset), (0, 0)), mode="constant")
    for x in range(kernel_offset, im.shape[0] - kernel_offset):
        for y in range(kernel_offset, im.shape[1] - kernel_offset):
            for c in range(im.shape[2]):
                im[x][y][c] = np.dot(np.reshape(im[x-kernel_offset:x+kernel_offset+1, y - kernel_offset:y+kernel_offset+1, c], (K*K))[::-1], np.reshape(kernel, (K*K)))
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
