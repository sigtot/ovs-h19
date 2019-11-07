import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils


def convolve_im(im: np.array, fft_kernel: np.array, verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    if im.shape != fft_kernel.shape:
        raise Exception("Error: image and fft_kernel should have equal dimensions")

    f = np.fft.fft2(im)
    fapplied = np.multiply(f, fft_kernel)
    conv_result = np.real(np.fft.ifft2(fapplied))

    if verbose:
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.title("Original")
        plt.subplot(1, 5, 2)
        plt.imshow(np.fft.fftshift(fft_kernel), cmap="gray")
        plt.title("Filter (shifted)")
        plt.subplot(1, 5, 3)
        plt.imshow(20*np.log(np.abs(np.fft.fftshift(f))), cmap="gray")
        plt.title("FT (shifted)")
        plt.subplot(1, 5, 4)
        plt.imshow(20*np.log(np.abs(np.fft.fftshift(fapplied))), cmap="gray")
        plt.title("FT filtered")
        plt.subplot(1, 5, 5)
        plt.imshow(conv_result, cmap="gray")
        plt.title("Result")

    return conv_result


if __name__ == "__main__":
    verbose = True

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass,
                                 verbose=verbose)
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass,
                                  verbose=verbose)

    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)
