import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from dflat.render.util_sensor import get_QETrans_Basler_Bayer
from dflat.render.util_spectral import get_rgb_bar_CIE1931, gamma_correction


def hsi_to_rgb(
    hsi,
    wavelength_set_m,
    demosaic=False,
    gamma=False,
    tensor_ordering=False,
    normalize=True,
    projection="Basler_Bayer",
):
    """Converts a batched hyperspectral datacube of shape [minibatch, Height, Width, Channels] to RGB. If tensor_ordering is true,
    input may instead be passed with the more common tensor shape [B, Ch, H, W]. The CIE1931 color matching functions are used by default.

    Args:
        hsi (float): Hyperspectral cube with shsape [B, H, W, Ch] or [B, Ch, H, W] if tensor_ordering is True.
        wavelength_set_m (float): List of wavelengths corresponding to the input channel dimension.
        demosaic (bool, optional): If True, a Bayer filter mask is applied to the RGB images and then interpolation is used to match experiment. Defaults to True.
        gamma (bool, optional): Applies gamma transformation to the input images. Defaults to True.
        tensor_ordering (bool, optional): If True, allows passing in a HSI with the more covenient pytorch to_tensor form. Defaults to False.
        normalize (bool, optional): If true, the returned projection is max normalized to 1.
        projection (str, optional): Either "CIE1931" or "Basler_Bayer". Specifies the color spectral curves.

            Returns:
        RGB: Stack of images with output channels=3
    """
    assert projection in [
        "CIE1931",
        "Basler_Bayer",
    ], "Projection must be one of ['CIE1931', 'Basler_Bayer']."

    input_tensor = torch.is_tensor(hsi)
    if not input_tensor:
        hsi = torch.tensor(hsi)
    if tensor_ordering:
        hsi = hsi.transpose(-3, -1).transpose(-3, -2).contiguous()
    assert (
        len(wavelength_set_m) == hsi.shape[-1]
    ), "List of wavelengths should match the input channel dimension."

    if projection.lower() == "cie1931":
        spec = get_rgb_bar_CIE1931(wavelength_set_m * 1e9)
    elif projection.lower() == "basler_bayer":
        spec, _ = get_QETrans_Basler_Bayer(wavelength_set_m * 1e9)
        spec = np.concatenate([spec[:, 0:1], spec[:, 2:]], axis=-1)
        spec = spec / np.sum(spec, axis=0, keepdims=True)
    spec = torch.tensor(spec).type_as(hsi)

    rgb = torch.matmul(hsi, spec)
    scale = torch.amax(rgb, dim=(-3, -2, -1), keepdim=True)
    if normalize:
        rgb = rgb / scale
    if demosaic:
        rgb = bayer_interpolate(bayer_mask(rgb))
    if gamma:
        rgb = gamma_correction(rgb)
    if tensor_ordering:
        rgb = rgb.transpose(-3, -1).transpose(-2, -1).contiguous()
    if not input_tensor:
        rgb = rgb.cpu().numpy()

    return rgb


def bayer_mask(rgb_img):
    """
    Masks the given RGB image according to a Bayer pattern.

    Arguments:
    rgb_img: Tensor of shape (..., height, width, 3)

    Returns:
    A masked image with the same shape as input.
    """
    img_shape = rgb_img.shape
    height = img_shape[-3]
    width = img_shape[-2]

    # Create the Bayer masks for R, G, and B
    r_mask = torch.tensor([[1, 0], [0, 0]]).type_as(rgb_img)
    g_mask = torch.tensor([[0, 1], [1, 0]]).type_as(rgb_img)
    b_mask = torch.tensor([[0, 0], [0, 1]]).type_as(rgb_img)

    # Tile the masks to cover the entire image
    r_mask = r_mask.repeat(height // 2, width // 2)[:, :, None]
    g_mask = g_mask.repeat(height // 2, width // 2)[:, :, None]
    b_mask = b_mask.repeat(height // 2, width // 2)[:, :, None]

    num_dims_to_add = rgb_img.dim() - r_mask.dim()
    new_shape = (1,) * num_dims_to_add + r_mask.shape
    r_mask = r_mask.reshape(new_shape)
    g_mask = g_mask.reshape(new_shape)
    b_mask = b_mask.reshape(new_shape)

    # Apply the masks
    r_channel = rgb_img[..., 0:1] * r_mask
    g_channel = rgb_img[..., 1:2] * g_mask
    b_channel = rgb_img[..., 2:3] * b_mask
    masked_image = torch.cat([r_channel, g_channel, b_channel], dim=-1)

    return masked_image


def bayer_interpolate(masked_image):
    """
    Interpolates a Bayer-masked image.

    Arguments:
    masked_image: Tensor of shape (..., height, width, 3).

    Returns:
    An interpolated image with the same shape as input.
    """
    img_shape = masked_image.shape
    masked_image = masked_image.reshape(-1, *img_shape[-3:])
    masked_image = rearrange(masked_image, "b h w c -> b c h w")

    # 3x3 interpolation kernel
    kernel = torch.tensor(
        [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]
    ).type_as(masked_image)[None, None, :, :]

    # Interpolate each channel
    r_interp = F.conv2d(masked_image[:, 0:1, :, :], kernel, padding=1)
    g_interp = F.conv2d(masked_image[:, 1:2, :, :], kernel / 2, padding=1)
    b_interp = F.conv2d(masked_image[:, 2:3, :, :], kernel, padding=1)

    interpolated_image = torch.cat([r_interp, g_interp, b_interp], dim=1)
    interpolated_image = rearrange(interpolated_image, "b c h w -> b h w c").reshape(
        *img_shape
    )
    return interpolated_image


def photons_to_ADU(
    image_photons,
    QE=1.0,
    gain=1.0,
    clip_zero=True,
    shot_noise=True,
    dark_noise=True,
    dark_offset=0,
    dark_noise_e=1.0,
):
    """Converts signal in units of photons to detector units. Applies QE to convert photons to electrons, applies shot and dark noise, gain, and clipping.

    Args:
        image_photons (float): Image/signal data in photons.
        QE (float, optional): Quantum Efficiency. Defaults to 1.0.
        gain (float, optional): Gain on electron signal. Defaults to 1.0.
        clip_zero (bool, optional): If true, returns output clipped to zero. Defaults to True.
        shot_noise (bool, optional): If true, applies shot noise via poisson model. Defaults to True.
        dark_noise (bool, optional): if true, applies dark noise via normal [dark_offset, dark_noise_e]. Defaults to True.
        dark_offset (int, optional): Dark noise normal mean. Defaults to 0.
        dark_noise_e (float, optional): dark noise normal std dev in electrons. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    dtype = image_photons.dtype
    electrons_signal = image_photons * QE

    # Shot Noise on electron signal
    if shot_noise:
        noise = torch.poisson(electrons_signal).to(dtype)
        electrons_signal = electrons_signal + noise

    # total dark noise electrons
    if dark_noise:
        gaussian_noise = torch.normal(
            dark_offset, dark_noise_e, size=image_photons.shape
        ).to(dtype)
        electrons_signal = electrons_signal + gaussian_noise

    # Apply gain and zero clipping (units is adu after applying gain, instead of e)
    electrons_signal = gain * electrons_signal

    if clip_zero:
        return torch.clip(electrons_signal, min=0)
    else:
        return electrons_signal
