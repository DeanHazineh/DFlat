import pytest
import numpy as np
from dflat.initialize import (
    focusing_lens,
    multiplexing_mask_orthrand,
    multiplexing_mask_sieve,
)


@pytest.fixture
def valid_input_dict():
    return {
        "in_size": [501, 501],
        "in_dx_m": [2e-6, 2e-6],
        "wavelength_set_m": [400e-9, 600e-9, 700e-9],
        "depth_set_m": [10e-3, 20e-3, 30e-3],
        "fshift_set_m": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "out_distance_m": 30e-3,
        "aperture_radius_m": 500e-6,
        "radial_symmetry": False,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"in_size": [0.0]},
        {"in_dx_m": [0.0]},
        {"fshift_set_m": [0.0]},
        {"fshift_set_m": [[0.0]]},
        {"depth_set_m": [0.0], "fshift_set_m": [[0.0]]},
        {"depth_set_m": [0.0], "fshift_set_m": [[0.0, 0.0], [0.0, 0.0]]},
        {"depth_set_m": [0.0, 0.0], "fshift_set_m": [[0.0, 0.0]]},
        {"radial_symmetry": True, "in_size": [4, 4]},
        {"radial_symmetry": True, "in_size": [7, 5]},
        {"radial_symmetry": True, "in_dx_m": [4.0, 5.0]},
        {"out_distance_m": 1},
        {"aperture_radius_m": "str"},
        {"radial_symmetry": 1.0},
    ],
)
def test_focusing_lens_invalid(updates, valid_input_dict):
    valid_input_dict.update(updates)
    with pytest.raises(AssertionError):
        _ = focusing_lens(**valid_input_dict)

    return


@pytest.mark.parametrize(
    "updates",
    [
        {
            "wavelength_set_m": [400e-9],
            "depth_set_m": [10e-3],
            "fshift_set_m": [[0.0, 0.0]],
        },
        {
            "wavelength_set_m": [400e-9, 600e-9, 700e-9],
            "depth_set_m": [10e-3, 20e-3, 30e-3],
            "fshift_set_m": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        },
    ],
)
def test_focusing_lens_valid(updates, valid_input_dict):
    # 2D phase generation output shape validation
    for radial_flag in [True, False]:
        valid_input_dict["radial_symmetry"] = radial_flag
        valid_input_dict.update(updates)

        ampl, phase, aperture = focusing_lens(**valid_input_dict)
        assert ampl.shape == phase.shape
        assert aperture.shape[-2:] == ampl.shape[-2:]

        num_l = len(valid_input_dict["wavelength_set_m"])
        assert ampl.shape[0] == num_l


@pytest.mark.parametrize(
    "num_sets, lens_size",
    [
        (4, (100, 100)),
        (9, (200, 200)),
    ],
)
def test_multiplexing_mask_sieve(num_sets, lens_size):
    masks = multiplexing_mask_sieve(num_sets, lens_size)
    assert masks.shape == (
        num_sets,
        lens_size[0],
        lens_size[1],
    ), "Incorrect mask shape."
    assert np.array_equal(masks, masks.astype(bool)), "Masks are not binary"


def test_multiplexing_mask_sieve_assertions():
    # num_sets is not a perfect square
    with pytest.raises(AssertionError):
        multiplexing_mask_sieve(2, (100, 100))

    # num_sets is less than 1
    with pytest.raises(AssertionError):
        multiplexing_mask_sieve(0, (100, 100))

    # num_sets is not an integer
    with pytest.raises(AssertionError):
        multiplexing_mask_sieve("4", (100, 100))


@pytest.mark.parametrize(
    "num_sets, block_dx, lens_dx, lens_size",
    [
        (4, (20, 20), (10, 10), (200, 200)),
        (3, (100, 100), (10, 10), (200, 200)),
    ],
)
def test_multiplexing_mask_orthrand(num_sets, block_dx, lens_dx, lens_size):
    masks = multiplexing_mask_orthrand(num_sets, block_dx, lens_dx, lens_size)
    assert masks.shape == (
        num_sets,
        lens_size[0],
        lens_size[1],
    ), "Incorrect mask shape."
    assert np.array_equal(masks, masks.astype(bool)), "Masks are not binary"


def test_multiplexing_mask_orthrand_assertions():
    # num_sets is less than 1
    with pytest.raises(AssertionError):
        multiplexing_mask_orthrand(0, (10.0, 10.0), (1.0, 1.0), (100, 100))

    # num_sets is not an integer
    with pytest.raises(AssertionError):
        multiplexing_mask_orthrand("3", (10.0, 10.0), (1.0, 1.0), (100, 100))

    # block_size_m has smaller sampling than the lens grid
    with pytest.raises(AssertionError):
        multiplexing_mask_orthrand(3, (1.0, 1.0), (10.0, 10.0), (100, 100))

    # Block size of mask should be an integer multiple of pixel pitch
    with pytest.raises(AssertionError):
        multiplexing_mask_orthrand(3, (13.0, 13.0), (10.0, 10.0), (100, 100))
