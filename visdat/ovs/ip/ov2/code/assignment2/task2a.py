import skimage
import numpy as np
import utils


def MaxPool2d(im: np.array,
              kernel_size: int):
    """ A function that max pools an image with size kernel size.
    Assume that the stride is equal to the kernel size, and that the kernel size is even.

    Args:
        im: [np.array of shape [H, W, 3]]
        kernel_size: integer
    Returns:
        im: [np.array of shape [H/kernel_size, W/kernel_size, 3]].
    """
    H, W = im.shape[0], im.shape[1]
    new_im = np.zeros(((H-1)//kernel_size + 1, (W-1)//kernel_size + 1, 3))
    for i, row in enumerate(im):
        for j, px in enumerate(row):
            for c, val in enumerate(px):
                new_im[i//kernel_size, j//kernel_size, c] = max(new_im[i//kernel_size, j//kernel_size, c], val)

    return new_im


if __name__ == "__main__":

    # DO NOT CHANGE
    im = skimage.data.chelsea()
    print(np.max(im), np.min(im))
    im = utils.uint8_to_float(im)
    max_pooled_image = MaxPool2d(im, 4)

    utils.save_im("chelsea.png", im)
    utils.save_im("chelsea_maxpooled.png", max_pooled_image)

    im = utils.create_checkerboard()
    im = utils.uint8_to_float(im)
    utils.save_im("checkerboard.png", im)
    max_pooled_image = MaxPool2d(im, 2)
    utils.save_im("checkerboard_maxpooled.png", max_pooled_image)