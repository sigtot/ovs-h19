import utils
import skimage
import skimage.morphology
import numpy as np


def remove_noise(im: np.ndarray) -> np.ndarray:
    """
        A function that removes noise in the input image.
        args:
            im: np.ndarray of shape (H, W) with boolean values (dtype=np.bool)
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    selem = skimage.morphology.selem.disk(8)
    im = skimage.morphology.binary_closing(im, selem)  # Close holes in triangle shape
    im = skimage.morphology.binary_opening(im, selem)  # Remove small white fragments
    return im


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    noise_free_image = remove_noise(binary_image)

    assert im.shape == noise_free_image.shape, \
        "Expected image shape ({}) to be same as resulting image shape ({})".format(
            im.shape, noise_free_image.shape)
    assert noise_free_image.dtype == np.bool, \
        "Expected resulting image dtype to be np.bool. Was: {}".format(
            noise_free_image.dtype)

    noise_free_image = utils.to_uint8(noise_free_image)
    utils.save_im("noisy-filtered.png", noise_free_image)

