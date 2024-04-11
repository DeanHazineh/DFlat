import pytest
import torch
import numpy as np
from dflat.render.fft_convolve import (
    general_convolve,
    weiner_deconvolve,
    fourier_convolve,
)


def test_fourier_convolve():
    # Create a simple image and filter
    image = torch.rand(1, 100, 100)
    filter = torch.rand(1, 100, 100)

    # Expected behavior: output shape matches input shape
    output = fourier_convolve(image, filter)
    assert output.shape == image.shape, "Output shape should match input shape."


def test_general_convolve():
    # Generate test image and filter with different sizes
    image = torch.rand(1, 120, 120)
    filter = torch.rand(1, 100, 100)

    # Convolve using general_convolve
    output = general_convolve(image, filter)
    assert (
        output.shape[-2:] == image.shape[-2:]
    ), "Output shape should be the same as input image shape."


def test_weiner_deconvolve():
    # Create test data for Wiener deconvolution
    image = torch.rand(1, 120, 120)
    filter = torch.rand(1, 120, 120)

    # Perform Wiener deconvolution
    output = weiner_deconvolve(image, filter, const=1e-3, abs=False)
    assert (
        output.shape[-2:] == image.shape[-2:]
    ), "Output shape should be the same as input image shape."

    # Test with absolute value option
    output_abs = weiner_deconvolve(image, filter, const=1e-3, abs=True)
    assert torch.all(output_abs >= 0), "Output should be non-negative when abs=True."


@pytest.mark.parametrize("rfft", [True, False])
def test_fourier_convolve_rfft_option(rfft):
    image = torch.rand(1, 128, 128)
    filter = torch.rand(1, 128, 128)
    output = fourier_convolve(image, filter, rfft=rfft)
    assert (
        output.shape == image.shape
    ), "Fourier convolve with rfft option should return correct shape."
