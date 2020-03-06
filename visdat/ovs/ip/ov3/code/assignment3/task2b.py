import utils
import numpy as np


def in_first_bounded_quadrant(x, X, y, Y):
    return 0 <= x < X and 0 <= y < Y


def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness). 
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row_seed, col_seed). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    segmented = np.zeros_like(im).astype(bool)
    discovered = np.empty_like(im).astype(bool)
    H, W = im.shape
    q = []
    for row_seed, col_seed in seed_points:
        discovered[:] = False
        discovered[row_seed][col_seed] = True
        q.append((row_seed, col_seed))
        seed_intensity = im[row_seed][col_seed]
        while q:
            row, col = q.pop(0)
            if abs(seed_intensity - im[row][col]) < T:
                segmented[row][col] = True
                for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    if in_first_bounded_quadrant((r := row + dr), H, (c := col + dc), W) and not discovered[r][c]:
                        discovered[r][c] = True
                        q.append((r, c))

    return segmented


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [ # (row, column)
        [254, 138], # Seed point 1
        [253, 296], # Seed point 2
        [233, 436], # Seed point 3
        [232, 417], # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, \
        "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, \
        "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented.png", segmented_image)

