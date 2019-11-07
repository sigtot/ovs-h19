import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils


def convolve_im(im: np.array, kernel: np.array, verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution.
        
        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    H, W = im.shape[0], im.shape[1]
    K = kernel.shape[0]
    k_padded = np.pad(kernel, ((0, H - K), (0, W - K)), mode="constant")
    if k_padded.shape != im.shape:
        raise Exception("Padded kernel does not match image dimensions {} != {}", k_padded.shape, im.shape)
    fk = np.fft.fft2(k_padded)

    f = np.fft.fft2(im)
    fapplied = np.multiply(f, fk)
    conv_result = np.real(np.fft.ifft2(fapplied))

    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.title("Original")
        plt.subplot(1, 5, 2)
        plt.imshow(np.abs(np.fft.fftshift(fk)), cmap="gray")
        plt.title("Filter")
        plt.subplot(1, 5, 3)
        plt.imshow(20*np.log(np.abs(np.fft.fftshift(f)) + 0.01), cmap="gray")
        plt.title("FT (log)")
        plt.subplot(1, 5, 4)
        plt.imshow(20*np.log(np.abs(np.fft.fftshift(fapplied)) + 0.01), cmap="gray")
        plt.title("FT filtered (log)")
        plt.subplot(1, 5, 5)
        plt.imshow(conv_result, cmap="gray")
        plt.title("Result")
    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
