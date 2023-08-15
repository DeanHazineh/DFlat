import dflat.data_structure as df_struct
from dflat.fourier_layer.core.compute_psf import psf_measured
from dflat.fourier_layer.ms_initialization_utilities import focus_lens_init
from dflat.plot_utilities import *

import matplotlib.pyplot as plt
import numpy as np
import torch


def call_psf_measured(engine):
    savepath = "dflat/fourier_layer/dev_test/fourier_layers_PSFs/output/"

    propagation_settings = {
        "wavelength_m": 532e-9,
        "ms_samplesM": {"x": 513, "y": 513},
        "ms_dx_m": {"x": 3.5e-6, "y": 3.5e-6},
        "radius_m": 513 * 3.5e-6 / 2,
        "sensor_distance_m": 40e-3,
        "initial_sensor_dx_m": {"x": 2e-6, "y": 2e-6},
        "sensor_pixel_size_m": {"x": 2e-6, "y": 2e-6},
        "sensor_pixel_number": {"x": 1024, "y": 1024},
        "radial_symmetry": False,
        "diffractionEngine": engine,
        ## Optional Keys
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    zfocus = 50e-3
    zoffset = [{"x": 0, "y": 0}]
    point_source_locs = np.array([[0, 0, zo] for zo in [40e-3, 50e-3, 70e-3]])

    radial_settings = [False, True]
    for use_radial in radial_settings:
        propagation_settings["radial_symmetry"] = use_radial
        prop_params = df_struct.prop_params(propagation_settings)
        df_struct.print_full_settings(prop_params)

        lens_trans, lens_phase, _, normby = focus_lens_init(
            prop_params, [532e-9], [zfocus], zoffset
        )
        lens_trans = torch.tensor(lens_trans)
        lens_phase = torch.tensor(lens_phase)

        psf_intensity, psf_phase = psf_measured(
            point_source_locs, lens_trans, lens_phase, prop_params, normby
        )

        fig = plt.figure()
        ax = addAxis(fig, 1, 3)
        for i in range(3):
            ax[i].imshow(psf_intensity[0, i, :, :])
        plt.show()

        print(psf_intensity.shape, psf_phase.shape)

    return


if __name__ == "__main__":
    # call_psf_measured("fresnel_fourier")
    call_psf_measured("ASM_fourier")
