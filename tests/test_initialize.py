import pytest
from dflat.initialize import focusing_lens


@pytest.mark.parametrize(
    "updates",
    [
        {"in_size": [0.0]},
        {"in_dx_m": [0.0]},
        {"fshift_set_m": [0.0]},
        {"depth_set_m": [0.0], "fshift_set_m": [[0.0]]},
        {"depth_set_m": [0.0], "fshift_set_m": [[0.0, 0.0], [0.0, 0.0]]},
        {"depth_set_m": [0.0, 0.0], "fshift_set_m": [[0.0, 0.0]]},
        {"radial_symmetry": True, "in_size": [4, 4]},
        {"radial_symmetry": True, "in_size": [7, 5]},
        {
            "radial_symmetry": True,
            "in_dx_m": [4.0, 5.0],
        },
        {"out_distance_m": 1},
        {"aperture_radius_m": "str"},
        {"radial_symmetry": 1.0},
    ],
)
def test_focusing_lens_invalid(updates):
    valid_input_dict = {
        "in_size": [501, 501],
        "in_dx_m": [2e-6, 2e-6],
        "wavelength_set_m": [400e-9, 600e-9, 700e-9],
        "depth_set_m": [10e-3, 20e-3],
        "fshift_set_m": [[0.0, 0.0], [0.0, 0.0]],
        "out_distance_m": 30e-3,
        "aperture_radius_m": 500e-6,
        "radial_symmetry": False,
    }

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
            "wavelength_set_m": [400e-9, 500e-9],
            "depth_set_m": [10e-3, 20e-3],
            "fshift_set_m": [[0.0, 0.0], [0.0, 0.0]],
        },
    ],
)
def test_focusing_lens_valid(updates):
    valid_input_dict = {
        "in_size": [501, 501],
        "in_dx_m": [2e-6, 2e-6],
        "wavelength_set_m": [400e-9, 600e-9, 700e-9],
        "depth_set_m": [10e-3, 20e-3],
        "fshift_set_m": [[0.0, 0.0], [0.0, 0.0]],
        "out_distance_m": 30e-3,
        "aperture_radius_m": 500e-6,
        "radial_symmetry": False,
    }

    # 2D phase generation output shape validation
    for radial_flag in [True, False]:
        valid_input_dict["radial_symmetry"] = radial_flag
        valid_input_dict.update(updates)

        ampl, phase, aperture = focusing_lens(**valid_input_dict)
        assert ampl.shape == phase.shape
        assert aperture.shape[-2:] == ampl.shape[-2:]

        num_z = len(valid_input_dict["depth_set_m"])
        num_l = len(valid_input_dict["wavelength_set_m"])
        assert list(ampl.shape[0:2]) == list([num_z, num_l])
