import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift, fft2, ifft2, rfft2, irfft2
from dflat.radial_tranforms import resize_with_crop_or_pad


def general_convolve(image, filter, rfft=False):
    """Runs the Fourier space convolution between an image and filter, where the filter kernels may have a different size from the image shape.

    Args:
        `image` (float or complex): Input image to apply the convolution filter kernel to, of shape [..., Ny, Nx]
        `filter` (float or complex): Convolutional filter kernel, of shape [..., My, Mx], but the same rank as the image input
        `rfft` (bool, optional): Flag to use real rfft instead of the general fft. Defaults to False.

    Returns:
        float or complex: Image with the filter convolved on the inner-most two dimensions
    """
    init_image_shape = image.shape
    imh, imw = init_image_shape[-2:]

    init_filter_shape = filter.shape
    fh, fw = init_filter_shape[-2:]

    # Pad image to the filter size if image is smaller than the filter
    # and pad filter to the image size
    image = resize_with_crop_or_pad(
        image, np.maximum(imh, fh), np.maximum(imw, fw), False
    )

    image_shape = image.shape
    filter_resh = resize_with_crop_or_pad(
        filter, image_shape[-2], image_shape[-1], radial_flag=False
    )

    # Run the convolution in frequency space
    convolve_func = fourier_convolve_real if rfft else fourier_convolve
    image = torch.real(convolve_func(image, filter_resh))

    # Undo odd padding if it was done before FFT
    image = resize_with_crop_or_pad(image, imh, imw, False)
    return image


def fourier_convolve(image, filter):
    """Computes the convolution of two signals (real or complex) using frequency space multiplcation. Convolution is done over the two inner-most dimensions.

    Args:
        `image` (float or complex): Image to apply filter to, of shape [..., Ny, Nx]
        `filter` (float or complex): Filter kernel; The kernel must be the same shape as the image

    Returns:
        complex: Image with filter convolved, same shape as input
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
        image (float): Image to apply filter to, of shape [..., Ny, Nx]
        filter (float):  Filter kernel; The kernel must be the same shape as the image

    Returns:
        float: Image with filter convolved, same shape as input
    """
    TORCH_ZERO = torch.tensor(0.0).to(image.dtype)
    fourier_product = rfft2(ifftshift(image)) * rfft2(ifftshift(filter))
    fourier_product = fftshift(irfft2(fourier_product))
    return fourier_product
