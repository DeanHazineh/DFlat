import pytest
import torch
import numpy as np
import scipy.special
from pyhank import qdht as qdht_reference

from dflat.radial_tranforms import (
    radial_2d_transform,
    radial_2d_transform_wrapped_phase,
    qdht,
    iqdht,
)
from dflat.radial_tranforms.radial_ops import general_interp_regular_1d_grid
from dflat.initialize import focusing_lens


def test_radial_2d_transform():
    # Get a profile to test our transformation
    ampl, phase, aperture = focusing_lens(
        in_size=[101, 101],
        in_dx_m=[1e-6, 1e-6],
        wavelength_set_m=[600e-9],
        depth_set_m=[2e-3],
        fshift_set_m=[[0.0, 0.0]],
        out_distance_m=1e-3,
        aperture_radius_m=51e-6,
        radial_symmetry=False,
    )
    ampl = ampl * aperture
    phase = phase * aperture
    cidx = 101 // 2

    ampl_trans = radial_2d_transform(ampl[:, cidx : cidx + 1, cidx:].squeeze(-2))
    phase_trans = radial_2d_transform_wrapped_phase(
        phase[:, cidx : cidx + 1, cidx:].squeeze(-2)
    )

    amp_mse = np.mean((ampl - ampl_trans) ** 2)
    phase_mse = np.mean((phase - phase_trans) ** 2)
    assert ampl.shape == ampl_trans.shape
    assert phase.shape == phase_trans.shape
    assert amp_mse < 0.1
    assert phase_mse < 0.1


def test_qdht():
    # Define a jinc function
    ri = np.linspace(0, 100, 1024)
    fi = np.zeros_like(ri)
    fi[1:] = scipy.special.jv(1, ri[1:]) / ri[1:]
    fi[ri == 0] = 0.5

    # Compute the dflat QDHT
    f = torch.tensor(fi[None], dtype=torch.float32)
    radial_grid = torch.tensor(ri[None], dtype=torch.float32)
    kr, ht = qdht(radial_grid, f)
    _, f = iqdht(kr, ht)
    ht = ht[0].numpy()
    f = f[0].numpy()

    # Forward and reverse transform should return original data
    assert np.allclose(
        f, fi, atol=0.05
    ), "The returned tensor and the initial tensor disagree by atol."

    # Compare the qdht calculation to pyhank
    _, ht_ref = qdht_reference(ri, fi)
    assert np.allclose(
        ht_ref, ht, atol=0.05
    ), "QDHT calculation disagrees with pyhank reference by atol."

    return


def test_general_interp_regular_1d_grid():
    # Test for general interpolation
    x = torch.linspace(0, 10, steps=11)
    xi = torch.tensor([1, 4, 9])

    y_complex = torch.exp(torch.complex(x, torch.tensor([0.0])))  # Complex tensor
    y_real = torch.abs(y_complex)

    res1 = general_interp_regular_1d_grid(x, xi, y_complex).numpy()
    res2 = general_interp_regular_1d_grid(x, xi, y_real).numpy()

    assert np.allclose(
        np.abs(res1), res2, atol=0.01
    ), "Error on equating real and complex inputs."
