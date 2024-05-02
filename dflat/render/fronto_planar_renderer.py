import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .fft_convolve import general_convolve
from dflat.radial_tranforms import resize_with_crop_or_pad
from .util_meas import hsi_to_rgb


class Fronto_Planar_Renderer_Incoherent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, psf_intensity, scene_radiance, rfft=True, crop_to_psf_dim=False):
        if len(scene_radiance.shape) == 5:
            scene_radiance = scene_radiance[None]
        psf_shape = psf_intensity.shape
        srad_shape = scene_radiance.shape
        # psf has shape  [B P Z L H W]
        # scene radiance [1 P Z L H W]

        assert len(psf_shape) == 6, "PSF should be rank 6 tensor like [B P Z L H W]."
        assert (
            len(srad_shape) == 6
        ), "Scene radiance should be passed as a rank 5 tensor like [P Z L H W]"
        for i in np.arange(1, 4, 1):
            assert (
                srad_shape[i] == psf_shape[i] or srad_shape[i] == 1
            ), f"PSF tensor and Scene radiance tensor must either have the same length along dimension {i} or must allow for broadcasting with length==1 for the radiance."

        psf_intensity = (
            torch.tensor(psf_intensity, dtype=torch.float32)
            if not torch.is_tensor(psf_intensity)
            else psf_intensity.to(dtype=torch.float32)
        )
        scene_radiance = (
            torch.tensor(scene_radiance, dtype=torch.float32)
            if not torch.is_tensor(scene_radiance)
            else scene_radiance.to(dtype=torch.float32)
        )

        return checkpoint(
            self._forward, psf_intensity, scene_radiance, rfft, crop_to_psf_dim
        )

    def _forward(self, psf_intensity, scene_radiance, rfft, crop_to_psf_dim):
        # Run the spatial convolution
        meas = general_convolve(scene_radiance, psf_intensity, rfft)
        if crop_to_psf_dim:
            resize_with_crop_or_pad(meas, *psf_intensity.shape[-2:], False)

        # Add noise or other image stuff here
        ## TO be updated and added later
        return meas

    def rgb_measurement(self, meas, wavelength_set_m, bayer_mosaic=False, gamma=True):
        B, P, Z, L, H, W = meas.shape
        meas = hsi_to_rgb(
            rearrange(meas, "B P Z L H W -> (B P Z) H W L", B=B, P=P, Z=Z),
            wavelength_set_m,
            bayer_mosaic,
            gamma,
        )
        meas = rearrange(meas, "(B P Z) H W C -> B P Z C H W", B=B, P=P, Z=Z)
        return meas
