import torch
import torch.nn.functional as F
from .util_spectral import get_rgb_bar_CIE1931, gamma_correction


def photons_to_ADU(image_photons, sensor_parameters, clip_zero=True):
    dtype = image_photons.dtype

    electrons_signal = image_photons * sensor_parameters["QE"]
    # Shot Noise on electron signal
    if sensor_parameters["shot_noise"]:
        noise = torch.poisson(electrons_signal).to(dtype)
        electrons_signal = electrons_signal + noise
        # electrons_signal.requires_grad_(True)

    # total dark noise electrons
    if sensor_parameters["dark_noise"]:
        gaussian_noise = torch.normal(sensor_parameters["dark_offset"], sensor_parameters["dark_noise_e"], size=image_photons.shape).to(dtype)
        electrons_signal = electrons_signal + gaussian_noise

    # Apply gain and zero clipping (units is adu after applying gain, instead of e)
    electrons_signal = sensor_parameters["gain"] * electrons_signal

    if clip_zero:
        return torch.clip(electrons_signal, min=0)
    else:
        return electrons_signal


def bayer_mask(rgb_img):
    """
    Masks the given RGB image according to a Bayer pattern.

    Arguments:
    rgb_img: Tensor of shape (batch, height, width, 3)

    Returns:
    A masked image with the same shape as input.
    """
    img_shape = rgb_img.shape
    if len(img_shape) != 4:
        raise ValueError("Input RGB image should be a rank 4 tensor")
    height = img_shape[-3]
    width = img_shape[-2]

    # Create the Bayer masks for R, G, and B
    r_mask = torch.tensor([[1, 0], [0, 0]]).type_as(rgb_img)
    g_mask = torch.tensor([[0, 1], [1, 0]]).type_as(rgb_img)
    b_mask = torch.tensor([[0, 0], [0, 1]]).type_as(rgb_img)

    # Tile the masks to cover the entire image
    r_mask = r_mask.repeat(height // 2, width // 2)[None, :, :, None]
    g_mask = g_mask.repeat(height // 2, width // 2)[None, :, :, None]
    b_mask = b_mask.repeat(height // 2, width // 2)[None, :, :, None]

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
    masked_image: Tensor of shape (batch, height, width, 3).

    Returns:
    An interpolated image with the same shape as input.
    """
    img_shape = masked_image.shape
    if len(img_shape) != 4:
        raise ValueError("Input RGB image should be a rank 4 tensor")

    # Reshape the input image to dimensions [minibatch, in_channels, height, width]
    masked_image = torch.permute(masked_image, (0, 3, 1, 2)).contiguous()

    # 3x3 interpolation kernel
    kernel = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]).type_as(masked_image)[None, None, :, :]

    # Interpolate each channel
    r_interp = F.conv2d(masked_image[:, 0:1, :, :], kernel, padding=1)
    g_interp = F.conv2d(masked_image[:, 1:2, :, :], kernel / 2, padding=1)
    b_interp = F.conv2d(masked_image[:, 2:3, :, :], kernel, padding=1)

    interpolated_image = torch.cat([r_interp, g_interp, b_interp], dim=1)
    interpolated_image = torch.permute(interpolated_image, (0, 2, 3, 1)).contiguous()
    return interpolated_image


def layer_output_to_rgb(layer_stack, wavelength_set_m, demosaic=True, gamma=True):
    """Converts the rendered hyperspectral image stack from the Fronto_Planar_renderer_incoherent layer to an RGB projection

    Args:
        layer_stack (torch.float): Tensor of shape [BatchSize, num_profile, num_point_sources, Height, Width, num_wl].
        demosaic (bool, optional): Flag to apply Bayer Demosaic and linear interpolation. Defaults to True.
        gamma (bool, optional): Flag to apply gamma correction to returned image. Defaults to True.
    """
    init_shape = layer_stack.shape
    layer_stack = layer_stack.view(-1, *init_shape[-3:])
    rgb = hsi_to_rgb(layer_stack, wavelength_set_m, demosaic, gamma)
    rgb = rgb.view(*init_shape[:-1], 3)

    return rgb


def hsi_to_rgb(hsi, wavelength_set_m, demosaic=True, gamma=True):
    "Converts a batched hyperspectral datacube of shape [minibatch, Height, Width, Channels] to RGB"
    if not torch.is_tensor(hsi):
        hsi = torch.tensor(hsi)

    cmf_bar = torch.tensor(get_rgb_bar_CIE1931(wavelength_set_m * 1e9)).type_as(hsi)
    rgb = torch.matmul(hsi, cmf_bar)
    rgb = rgb / torch.amax(rgb, dim=(-3, -2, -1), keepdim=True)

    if demosaic:
        rgb = bayer_interpolate(bayer_mask(rgb))
    if gamma:
        rgb = gamma_correction(rgb)

    return rgb
