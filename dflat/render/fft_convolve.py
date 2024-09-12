import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.fft import fftshift, ifftshift, fft2, ifft2, rfft2, irfft2
from dflat.radial_tranforms import resize_with_crop_or_pad




def general_convolve(image, filter, rfft=False, mode="valid"):
    """Runs the Fourier space convolution between an image and filter, where the filter kernels may have a different size from the image shape.

    Args:
        `image` (tf.float or tf.complex): Input image to apply the convolution filter kernel to, of shape [..., Ny, Nx]
        `filter` (tf.float or tf.complex): Convolutional filter kernel, of shape [..., My, Mx], but the same rank as the image input
        `rfft` (bool, optional): Flag to use real rfft instead of the general fft. Defaults to False.
        'mode' (str, optional): Choice of valid or full convolution.

    Returns:
        tf.float or tf.complex: Image with the filter convolved on the inner-most two dimensions
    """
    assert mode in ["valid", "full"]
    init_image_shape = image.shape
    im_ny = init_image_shape[-2]
    im_nx = init_image_shape[-1]

    init_filter_shape = filter.shape
    filt_ny = init_filter_shape[-2]
    filt_nx = init_filter_shape[-1]

    # Zero pad the image by half filter
    padby = (len(init_image_shape) * 2) * [0]
    padby[0:2] = [filt_nx // 2, filt_nx // 2]
    padby[2:4] = [filt_ny // 2, filt_ny // 2]
    image = F.pad(image, padby, mode="constant", value=0.0)

    ### Pad the psf to match the new image dimensionality
    image_shape = image.shape
    filter_resh = resize_with_crop_or_pad(filter, *image_shape[-2:], radial_flag=False)

    ### Run the convolution (Defualt to using a checkpoint of the fourier transform)
    image = checkpoint(fourier_convolve, image, filter_resh, rfft)
    image = torch.real(image)

    if mode == "valid":
        image = resize_with_crop_or_pad(image, *init_image_shape[-2:], False)
    return image


def weiner_deconvolve(image, filter, const=1e-4, abs=False):
    init_image_shape = image.shape
    im_ny = init_image_shape[-2]
    im_nx = init_image_shape[-1]

    init_filter_shape = filter.shape
    filt_ny = init_filter_shape[-2]
    filt_nx = init_filter_shape[-1]

    # If the image is smaller than the filter size, then we should increase the image
    if im_ny < filt_ny or im_nx < filt_nx:
        image = resize_with_crop_or_pad(
            image, np.maximum(im_ny, filt_ny), np.maximum(im_nx, filt_nx), False
        )

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
    filter_resh = resize_with_crop_or_pad(filter, *image_shape[-2:], radial_flag=False)

    ### Run the deconvolution
    kft = fft2(filter_resh)
    G = torch.conj(kft) / (torch.conj(kft) * kft + const)
    image = torch.real(ifft2(fft2(image) * G))
    image = torch.abs(image) if abs else image

    #### Undo odd padding if it was done before FFT
    image = resize_with_crop_or_pad(image, *init_image_shape[-2:], False)
    return image


def fourier_convolve(image, filter, rfft=False):
    """Computes the convolution of two signals (real or complex) using frequency space multiplcation. Convolution is done over the two inner-most dimensions.

    Args:
        `image` (float or complex): Image to apply filter to, of shape [..., Ny, Nx]
        `filter` (float or complex): Filter kernel; The kernel must be the same shape as the image

    Returns:
        complex: Image with filter convolved, same shape as input
    """
    # Ensure inputs are complex
    TORCH_ZERO = torch.tensor(0.0).to(dtype=image.dtype, device=image.device)
    if rfft:
        fourier_product = rfft2(ifftshift(image)) * rfft2(ifftshift(filter))
        fourier_product = fftshift(irfft2(fourier_product))
    else:
        image = torch.complex(image, TORCH_ZERO) if not image.is_complex() else image
        filter = (
            torch.complex(filter, TORCH_ZERO) if not filter.is_complex() else filter
        )
        fourier_product = fft2(ifftshift(image)) * fft2(ifftshift(filter))
        fourier_product = fftshift(ifft2(fourier_product))

    return fourier_product
