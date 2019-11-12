import numpy as np
import utils
import pathlib


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    top = 256
    t_max = 0
    var_max = 0
    pdf, bins = np.histogram(im, range(top + 1), density=True)
    omega0 = 0
    omega1 = 1
    sum0 = 0
    sum1 = np.dot(range(top), pdf)
    for t in range(top):
        d_omega = pdf[t]
        omega0 += d_omega
        omega1 -= d_omega

        d_sum = t * pdf[t]
        sum0 += d_sum
        sum1 -= d_sum

        mu0 = sum0 / omega0
        mu1 = sum1 / omega1

        if (var := omega0 * omega1 * (mu0 - mu1) ** 2) > var_max:
            var_max, t_max = var, t

    thresh = t_max
    im[im >= thresh] = 255
    im[im < thresh] = 0
    return thresh


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, \
            "Expected image shape ({}) to be same as thresholded image shape ({})".format(
                im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, \
            "Expected thresholded image dtype to be np.bool. Was: {}".format(
                segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)
