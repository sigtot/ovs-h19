import matplotlib.pyplot as plt
import os
import numpy as np

image_output_dir = "image_processed"
os.makedirs(image_output_dir, exist_ok=True)


def save_im(imname, im, cmap=None):
    impath = os.path.join(image_output_dir, imname)
    plt.imsave(impath, im, cmap=cmap)


def lin_lum(px: np.array(3)) -> int:
    return int(0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2])


def greyscale(rgb_img):
    """ Converts an RGB image to sRGB greyscale
    
    Args:
        rgb_img ([type]): [np.array of shape [H, W, 3]]
    
    Returns:
        grey_img ([type]): [np.array of shape [H, W]]
    """
    grey_img = np.empty((rgb_img.shape[:2]), dtype=int)
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            px = rgb_img[i][j]
            grey_img[i][j] = lin_lum(px)
    return grey_img


def inverse(grey_im):
    """ Finds the inverse of the greyscale image
    
    Args:
        grey_im ([type]): [np.array of shape [H, W]]
    
    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    inv_im = np.empty_like(grey_im)
    for i in range(grey_im.shape[0]):
        for j in range(grey_im.shape[1]):
            inv_im[i][j] = 255 - grey_im[i][j]
    return inv_im


if __name__ == "__main__":
    im = plt.imread("images/lake.jpg")
    gray_im = greyscale(im)
    inverse_im = inverse(gray_im)
    save_im("lake_greyscale.png", gray_im, cmap="gray")
    save_im("lake_inverse.png", inverse_im, cmap="gray")
