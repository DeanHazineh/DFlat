import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from dflat.render.fronto_planar_renderer import (
    Fronto_Planar_Renderer_Incoherent,
)


@pytest.fixture
def psf_intensity_and_scene_radiance():
    # Create dummy data for PSF intensity and scene radiance
    psf_intensity = np.random.rand(2, 3, 4, 5, 128, 128)  # Shape [B P Z L H W]
    scene_radiance = np.random.rand(3, 4, 5, 128, 128)  # Shape [P Z L H W]
    scene_radiance = np.expand_dims(
        scene_radiance, 0
    )  # Adding batch dimension [1 P Z L H W]
    return torch.tensor(psf_intensity, dtype=torch.float32), torch.tensor(
        scene_radiance, dtype=torch.float32
    )


@pytest.fixture
def wavelength_set_m():
    return np.linspace(400e-9, 700e-9, 5)


def test_fronto_planar_renderer_forward(psf_intensity_and_scene_radiance):
    psf_intensity, scene_radiance = psf_intensity_and_scene_radiance
    renderer = Fronto_Planar_Renderer_Incoherent()
    output = renderer.forward(
        psf_intensity, scene_radiance, rfft=True, crop_to_psf_dim=False
    )
    assert (
        output.shape[-3:] == scene_radiance.shape[-3:]
    ), "Output shape should match the scene radiance spatial dimensions."


def test_fronto_planar_renderer_rgb_conversion(
    psf_intensity_and_scene_radiance, wavelength_set_m
):
    psf_intensity, scene_radiance = psf_intensity_and_scene_radiance
    renderer = Fronto_Planar_Renderer_Incoherent()
    meas = renderer.forward(
        psf_intensity, scene_radiance, rfft=True, crop_to_psf_dim=False
    )

    rgb_output = renderer.rgb_measurement(
        meas, wavelength_set_m, bayer_mosaic=True, gamma=True
    )

    expected_shape = meas.shape[:-3] + (3, meas.shape[-2], meas.shape[-1])
    assert (
        rgb_output.shape == expected_shape
    ), "RGB output shape should include color channels and match other dimensions."
