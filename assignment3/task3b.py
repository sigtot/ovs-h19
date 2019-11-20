import utils
import skimage
import skimage.morphology
import numpy as np
from task3a import remove_noise


def distance_transform(im: np.ndarray) -> np.ndarray:
    """
        A function that computes the distance to the closest boundary pixel.

        args:
            im: np.ndarray of shape (H, W) with boolean values (dtype=np.bool)
        return:
            (np.ndarray) of shape (H, W). dtype=np.int32
    """
    assert im.dtype == np.bool
    selem = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=bool)
    eroded_in = np.copy(im)
    eroded_out = np.copy(im)
    result = np.zeros(im.shape, np.int32)
    while len(np.nonzero(eroded_in)[0]) > 0:
        skimage.morphology.binary_erosion(eroded_in, selem=selem, out=eroded_out)
        result += eroded_out
        eroded_in, eroded_out = eroded_out, eroded_in  # Swap buffers
    return result


if __name__ == "__main__":
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    noise_free_image = remove_noise(binary_image)
    distance = distance_transform(noise_free_image)

    assert im.shape == distance.shape, \
        "Expected image shape ({}) to be same as resulting image shape ({})".format(
            im.shape, distance.shape)
    assert distance.dtype == np.int32, \
        "Expected resulting image dtype to be np.int32. Was: {}".format(
            distance.dtype)

    distance = utils.to_uint8(distance)
    utils.save_im("noisy-distance.png", distance)

    
    



