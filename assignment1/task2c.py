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
    im = np.pad(im, ((kernel_offset, kernel_offset), (kernel_offset, kernel_offset), (0, 0)), mode="edge")
    H, W, num_chans = im.shape[0], im.shape[1], im.shape[2]
    out = im.copy()
    for x in range(kernel_offset, H - kernel_offset):
        for y in range(kernel_offset, W - kernel_offset):
            for c in range(num_chans):
                out[x][y][c] = np.max([np.min([np.dot(
                    np.reshape(im[x-kernel_offset:x+kernel_offset+1, y-kernel_offset:y+kernel_offset+1, c], (K*K))[::-1],
                    np.reshape(kernel, (K*K))), 255]), 0])
    return out[kernel_offset:-kernel_offset, kernel_offset:-kernel_offset]


if __name__ == "__main__":
    # Read image
    lake_im = plt.imread(os.path.join("images", "lake.jpg"))

    # Define the convolutional kernels
    h_a = np.ones((3, 3)) / 9
    h_b = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    h_c = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ])
    h_d = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    # Convolve images
    smoothed_im1 = convolve_im(lake_im, h_a)
    smoothed_im2 = convolve_im(lake_im, h_b)
    edge_im = convolve_im(lake_im, h_c)
    sharpened_im = convolve_im(lake_im, h_d)

    save_im("convolved_im_h_a.png", smoothed_im1)
    save_im("convolved_im_h_b.png", smoothed_im2)
    save_im("edge_detection.png", edge_im)
    save_im("sharpened.png", sharpened_im)

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
