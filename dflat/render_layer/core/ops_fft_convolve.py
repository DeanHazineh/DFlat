import torch
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift, fft2, ifft2, rfft2, irfft2
import numpy as np

from dflat.fourier_layer.core.ops_transform_util import resize_with_crop_or_pad


def general_convolve(image, filter, rfft=False):
    """Runs the Fourier space convolution between an image and filter, where the filter kernels may have a different size from the image shape.

    Args:
        `image` (tf.float or tf.complex): Input image to apply the convolution filter kernel to, of shape [..., Ny, Nx]
        `filter` (tf.float or tf.complex): Convolutional filter kernel, of shape [..., My, Mx], but the same rank as the image input
        `rfft` (bool, optional): Flag to use real rfft instead of the general fft. Defaults to False.

    Returns:
        tf.float or tf.complex: Image with the filter convolved on the inner-most two dimensions
    """
    init_image_shape = image.shape
    im_ny = init_image_shape[-2]
    im_nx = init_image_shape[-1]

    init_filter_shape = filter.shape
    filt_ny = init_filter_shape[-2]
    filt_nx = init_filter_shape[-1]

    # If the image is smaller than the filter size, then we should increase the image
    if im_ny < filt_ny or im_nx < filt_nx:
        image = resize_with_crop_or_pad(image, np.maximum(im_ny, filt_ny), np.maximum(im_nx, filt_nx), False)

    # Zero pad the image with half the filter dimensionality and ensure the image is odd
    padby = (len(init_image_shape) * 2) * [0]
    if np.mod(im_nx, 2) == 0 and np.mod(filt_nx, 2) == 0:
        padby[0:2] = [filt_nx // 2, filt_nx // 2 + 1]
    else:
        padby[0:2] = [filt_nx // 2, filt_nx // 2]

    if np.mod(im_ny, 2) == 0 and np.mod(filt_ny, 2) == 0:
        padby[2:4] = [filt_ny // 2, filt_ny // 2 + 1]
    else:
        padby[2:4] = [filt_ny // 2, filt_ny // 2]
    image = F.pad(image, padby, mode="constant", value=0.0)

    ### Pad the psf to match the new image dimensionality
    image_shape = image.shape
    filter_resh = resize_with_crop_or_pad(filter, image_shape[-2], image_shape[-1], radial_flag=False)

    ### Run the convolution
    convolve_func = fourier_convolve_real if rfft else fourier_convolve
    image = torch.real(convolve_func(image, filter_resh))

    ### Undo odd padding if it was done before FFT
    image = resize_with_crop_or_pad(image, init_image_shape[-2], init_image_shape[-1], False)
    return image


def fourier_convolve(image, filter):
    """Computes the convolution of two signals (real or complex) using frequency space multiplcation. Convolution is done over the two inner-most dimensions.

    Args:
        `image` (tf.float or tf.complex): Image to apply filter to, of shape [..., Ny, Nx]
        `filter` (tf.float or tf.complex): Filter kernel; The kernel must be the same shape as the image

    Returns:
        tf.complex: Image with filter convolved, same shape as input
    """
    # Ensure inputs are complex
    TORCH_ZERO = torch.tensor(0.0).to(image.dtype)
    if not image.is_complex():
        image = torch.complex(image, TORCH_ZERO)
    if not filter.is_complex():
        filter = torch.complex(filter, TORCH_ZERO)

    fourier_product = fft2(ifftshift(image)) * fft2(ifftshift(filter))
    fourier_product = fftshift(ifft2(fourier_product))
    return fourier_product


def fourier_convolve_real(image, filter):
    """Computes the convolution of two, real-valued signals using frequency space multiplication. Convolution is done over the two inner-most dimensions.

    Args:
        image (tf.float): Image to apply filter to, of shape [..., Ny, Nx]
        filter (tf.float):  Filter kernel; The kernel must be the same shape as the image

    Returns:
        tf.float: Image with filter convolved, same shape as input
    """
    TORCH_ZERO = torch.tensor(0.0).to(image.dtype)
    fft_length = image.shape[-1]
    fourier_product = rfft2(ifftshift(image)) * rfft2(ifftshift(filter))
    fourier_product = fftshift(irfft2(fourier_product))
    return fourier_product
