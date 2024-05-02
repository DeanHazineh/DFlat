import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .fft_convolve import general_convolve
from dflat.radial_tranforms import resize_with_crop_or_pad
from .util_meas import hsi_to_rgb


class Fronto_Planar_Renderer_Incoherent(nn.Module):
    def __init__(
        self,
        mode="full",
        wavelength_set_m=None,
        bayer_mosaic=False,
        gamma=False,
        rfft=True,
        crop_to_psf_dim=False,
    ):
        super().__init__()
        self.valid_modes = ["full", "grayscale", "rgb"]
        self.mode = mode
        assert mode in self.valid_modes, f"mode must be one of {self.valid_modes}"
        if mode == "rgb":
            assert (
                wavelength_set_m is not None
            ), "Wavelengths must be specified if requesting rgb projection."
        self.wavelength_set_m = wavelength_set_m
        self.bayer_mosaic = bayer_mosaic
        self.gamma = gamma
        self.rfft = rfft
        self.crop_to_psf_dim = crop_to_psf_dim

    def forward(self, psf_intensity, scene_radiance):
        """Render a fronto-planar intensity image given the feature-dependent point-spread function and an all-in-focus intensity map.

        Args:
            psf_intensity (float): point-spread function of shape [B, P, Z, Lam, H W].
            scene_radiance (float): Scene intensity map of shape [B, P, Z, Lam H W], where the first dimension may be excluded. Broadcasting may be used with dim=1.

        Returns:
            float: rendered image, which may be a projection onto a grayscale or rgb sensor if specified by mode initialization.
        """

        if len(scene_radiance.shape) == 5:
            scene_radiance = scene_radiance[None]
        psf_shape = psf_intensity.shape
        srad_shape = scene_radiance.shape

        assert len(psf_shape) == 6, "PSF should be rank 6 tensor like [B P Z L H W]."
        assert (
            len(srad_shape) == 6
        ), "Scene radiance should be passed as a rank 5 tensor like [P Z L H W]"
        for i in [1, 2, 3]:
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

        return checkpoint(self._forward, psf_intensity, scene_radiance)

    def _forward(self, psf_intensity, scene_radiance):
        meas = general_convolve(scene_radiance, psf_intensity, self.rfft)
        if self.crop_to_psf_dim:
            meas = resize_with_crop_or_pad(meas, *psf_intensity.shape[-2:], False)

        # Add sensor noise again here
        # TODO

        if self.mode == "grayscale":
            meas = torch.sum(meas, dim=(1, 2, 3), keepdim=True)
            meas = meas / torch.sum(meas, dim=(-1, -2), keepdim=True)
        elif self.mode == "rgb":
            meas = self.rgb_measurement(
                meas, self.wavelength_set_m, self.bayer_mosaic, self.gamma
            )

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
