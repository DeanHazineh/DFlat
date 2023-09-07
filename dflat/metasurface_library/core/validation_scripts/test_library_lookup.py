import numpy as np
import pickle
import matplotlib.pyplot as plt

import dflat.fourier_layer as df_fourier
import dflat.data_structure as df_struct
import dflat.plot_utilities as gF
import dflat.metasurface_library as df_library
import dflat.neural_optical_layer as df_neural


def check_nanocylinder():
    # Generate user inputs
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 571, "y": 571},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": None,
            "sensor_distance_m": 40e-3,
            "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_number": {"x": 501, "y": 501},
            "radial_symmetry": False,
            "diffractionEngine": "ASM_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )
    focus_trans, focus_phase, _, _ = df_fourier.focus_lens_init(propagation_parameters, [532e-9], [20e-3], [{"x": 1e-3, "y": 0}])
    wavelength = propagation_parameters["wavelength_set_m"]

    # Try out different lookup methods
    shape_vect_min, norm_shape_vect_min = df_library.optical_response_to_param(
        [focus_trans],
        [focus_phase],
        wavelength,
        "Nanocylinders_U180nm_H600nm",
        reshape=True,
        fast=False,
    )

    shape_vect_lookup, norm_shape_vect_lookup = df_library.optical_response_to_param(
        [focus_trans],
        [focus_phase],
        wavelength,
        "Nanocylinders_U180nm_H600nm",
        reshape=True,
        fast=True,
    )
    print(shape_vect_min.shape, shape_vect_lookup.shape)

    mlp_layer = df_neural.MLP_Layer("MLP_Nanocylinders_Dense256_U180_H600_SIREN100")
    trans_min, phase_min = mlp_layer(norm_shape_vect_min, [532e-9])
    trans_look, phase_look = mlp_layer(norm_shape_vect_lookup, [532e-9])
    print(phase_look.shape, phase_min.shape)

    fig = plt.figure()
    ax = gF.addAxis(fig, 2, 2)
    im = ax[0].imshow(shape_vect_min[0, :, :])
    gF.formatPlots(fig, ax[0], im, addcolorbar=True)
    im = ax[1].imshow(shape_vect_lookup[0, :, :])
    gF.formatPlots(fig, ax[1], im, addcolorbar=True)

    im = ax[2].imshow(phase_min[0, 0, :, :])
    gF.formatPlots(fig, ax[2], im, addcolorbar=True)
    im = ax[3].imshow(phase_look[0, 0, :, :])
    gF.formatPlots(fig, ax[3], im, addcolorbar=True)
    plt.show()

    return


def check_nanofin():
    # Generate user inputs
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": [532e-9],
            "ms_samplesM": {"x": 51, "y": 51},
            "ms_dx_m": {"x": 10 * 350e-9, "y": 10 * 350e-9},
            "radius_m": None,
            "sensor_distance_m": 2e-3,
            "initial_sensor_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_size_m": {"x": 3.5e-6, "y": 3.5e-6},
            "sensor_pixel_number": {"x": 501, "y": 501},
            "radial_symmetry": False,
            "diffractionEngine": "ASM_fourier",
            "automatic_upsample": False,
            "manual_upsample_factor": 1,
        }
    )
    focus_trans, focus_phase, _, _ = df_fourier.focus_lens_init(
        propagation_parameters,
        [532e-9, 532e-9],
        [2e-3, 2e-3],
        [{"x": -20e-6, "y": 0}, {"x": -20e-6, "y": 0}],
    )

    # Try out different lookup methods
    shape_vect_min, norm_shape_vect_min = df_library.optical_response_to_param(
        [focus_trans], [focus_phase], propagation_parameters["wavelength_set_m"], "Nanofins_U350nm_H600nm", reshape=True, fast=False
    )

    shape_vect_lookup, norm_shape_vect_lookup = df_library.optical_response_to_param(
        [focus_trans], [focus_phase], propagation_parameters["wavelength_set_m"], "Nanofins_U350nm_H600nm", reshape=True, fast=True
    )

    fig = plt.figure()
    ax = gF.addAxis(fig, 2, 2)
    im = ax[0].imshow(shape_vect_min[0, :, :])
    im = ax[1].imshow(shape_vect_min[1, :, :])
    im = ax[2].imshow(shape_vect_lookup[0, :, :])
    im = ax[3].imshow(shape_vect_lookup[1, :, :])

    mlp_layer = df_neural.MLP_Layer("MLP_Nanofins_Dense1024_U350_H600_SIREN100")
    _, phase_min = mlp_layer(norm_shape_vect_min, [532e-9])
    _, phase_look = mlp_layer(norm_shape_vect_lookup, [532e-9])
    out_shape = phase_min.shape
    cx, cy = out_shape[-1] // 2, out_shape[-2] // 2

    #
    phase_min = phase_min - phase_min[:, :, cy, cx][..., None, None]
    phase_look = phase_look - phase_look[:, :, cy, cx][..., None, None]

    fig = plt.figure()
    ax = gF.addAxis(fig, 3, 2)
    ax[0].imshow(focus_phase[0, :, :])
    ax[1].imshow(focus_phase[1, :, :])
    ax[2].imshow(phase_min[0, 0, :, :])
    ax[3].imshow(phase_min[0, 1, :, :])
    ax[4].imshow(phase_look[0, 0, :, :])
    ax[5].imshow(phase_look[0, 1, :, :])
    plt.show()

    return


if __name__ == "__main__":
    check_nanocylinder()
    check_nanofin()
