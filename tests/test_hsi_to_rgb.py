import pytest
import torch
import numpy as np
from dflat.render import hsi_to_rgb


@pytest.fixture
def hyperspectral_data():
    # Generates a synthetic hyperspectral dataset
    B, H, W, Ch = 1, 10, 10, 5  # Example dimensions
    hsi = np.random.rand(B, H, W, Ch).astype(np.float32)
    wavelengths = np.linspace(
        400e-9, 700e-9, Ch
    )  # Simulated wavelength range from 400nm to 700nm
    return hsi, wavelengths


@pytest.mark.parametrize(
    "demosaic, gamma, normalize",
    [
        (True, True, True),
        (False, False, False),
        (True, False, True),
        (False, True, False),
    ],
)
def test_hsi_to_rgb(hyperspectral_data, demosaic, gamma, normalize):
    hsi, wavelengths = hyperspectral_data
    hsi_tensor = torch.tensor(hsi)
    B, H, W, _ = hsi_tensor.shape  # Define B, H, and W based on tensor shape

    # Convert hyperspectral image to RGB using the provided function
    rgb_image = hsi_to_rgb(
        hsi_tensor,
        wavelengths,
        demosaic=demosaic,
        gamma=gamma,
        tensor_ordering=False,
        normalize=normalize,
        projection="CIE1931",  # You can switch between "CIE1931" and "Basler_Bayer" if needed
    )

    # Check the shape of the output
    expected_shape = (B, H, W, 3)  # Expecting 3 channels for RGB
    assert (
        rgb_image.shape == expected_shape
    ), f"Expected RGB shape {expected_shape}, but got {rgb_image.shape}"

    # Optional: Add additional checks such as dtype or value ranges
    assert rgb_image.dtype == torch.float32, "Output should be of type float32"
    if normalize:
        assert rgb_image.max() <= 1, "Normalized RGB values should not exceed 1"

    # Check that the RGB conversion does not produce any unexpected extremely high or low values
    assert not torch.any(torch.isnan(rgb_image)), "RGB image should not contain NaNs"
    assert not torch.any(
        torch.isinf(rgb_image)
    ), "RGB image should not contain infinities"
