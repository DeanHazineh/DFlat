import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier
import dflat.plot_utilities as df_plot
from dflat.fourier_layer import radial_2d_transform, radial_2d_transform_wrapped_phase

import matplotlib.pyplot as plt
import numpy as np


def plot_psfs(out_int, out_phase, out_int_radial, out_phase_radial, savepath, engine):
    ###
    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(out_int[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "intensity_engine_" + engine + ".png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(out_phase[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "phase_engine_" + engine + ".png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(out_int_radial[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "intensity_engine_" + engine + "_radial.png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(out_phase_radial[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "phase_engine_" + engine + "radial.png")
    plt.close()

    return


def plot_fields(out_int, out_phase, out_int_radial, out_phase_radial, savepath, engine):
    ###
    fig = plt.figure()
    ax = df_plot.addAxis(fig, 2, 3)
    for i in range(3):
        ax[i].imshow(out_int[i, 0, :, :])
        ax[i + 3].imshow(out_phase[i, 0, :, :])
    plt.savefig(savepath + "engine_" + engine + ".png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 2, 3)
    for i in range(3):
        ax[i].imshow(out_int_radial[i, 0, :, :])
        ax[i + 3].imshow(out_phase_radial[i, 0, :, :])
    plt.savefig(savepath + "intensity_engine_" + engine + "_radial.png")

    return


def call_psf_layer(engine, device="cuda", batch_loop=False):
    savepath = "dflat/fourier_layer/test_scripts/output_psfs/"

    ### Define new propagation parameter
    propagation_settings = {
        # Mandatory Keys
        "wavelength_set_m": [450e-9, 532e-9, 700e-9],
        "ms_samplesM": {"x": 513, "y": 513},
        "ms_dx_m": {"x": 1e-6, "y": 1e-6},
        "radius_m": 256e-6,
        "sensor_distance_m": 6e-3,
        "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_number": {"x": 256, "y": 256},
        "radial_symmetry": False,
        "diffractionEngine": engine,
        # Optional Keys
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    prop_params = df_struct.prop_params(propagation_settings)
    propagation_settings["radial_symmetry"] = True
    prop_params_radial = df_struct.prop_params(propagation_settings)

    ### Do PSF Layer Calls
    zfocus = 50e-3
    zlocs = [30e-3, 50e-3, 80e-3]
    zoffset = [{"x": 0, "y": 0}]
    point_source_locs = np.array([[0, 0, zo] for zo in zlocs])

    # run 2D Calculation
    psf_layer = df_fourier.PSF_Layer(prop_params).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params, [532e-9], [zfocus], zoffset)
    out_int, out_phase = psf_layer([lens_trans, lens_phase], point_source_locs, batch_loop=batch_loop)

    # Run radial version Calculation
    psf_layer = df_fourier.PSF_Layer(prop_params_radial).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params_radial, [532e-9], [zfocus], zoffset)
    out_int_radial, out_phase_radial = psf_layer([lens_trans, lens_phase], point_source_locs, batch_loop=batch_loop)

    print(out_int.shape, out_phase.shape, out_int_radial.shape, out_phase_radial.shape)

    # Make plot
    plot_psfs(out_int, out_phase, out_int_radial, out_phase_radial, savepath, engine)

    return


def call_matrixASM_psf_layer(device="cuda", batch_loop=False):
    savepath = "dflat/fourier_layer/test_scripts/output_psfs/"

    ### Define new propagation parameter
    propagation_settings = {
        # Mandatory Keys
        "wavelength_set_m": [450e-9, 532e-9, 700e-9],
        "ms_samplesM": {"x": 513, "y": 513},
        "ms_dx_m": {"x": 1e-6, "y": 1e-6},
        "radius_m": 256e-6,
        "sensor_distance_m": 6e-3,
        "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_number": {"x": 256, "y": 256},
        "radial_symmetry": False,
        "diffractionEngine": "ASM_fourier",
        # Optional Keys
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    prop_params = df_struct.prop_params(propagation_settings)
    propagation_settings["radial_symmetry"] = True
    prop_params_radial = df_struct.prop_params(propagation_settings)

    ### Do PSF Layer Calls
    zfocus = 50e-3
    zlocs = [30e-3, 50e-3, 80e-3]
    zoffset = [{"x": 0, "y": 0}]
    point_source_locs = np.array([[0, 0, zo] for zo in zlocs])

    # run 2D Calculation
    psf_layer = df_fourier.PSF_Layer_MatrixBroadband(prop_params).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params, [532e-9], [zfocus], zoffset)
    out_int, out_phase = psf_layer([lens_trans, lens_phase], point_source_locs, batch_loop=batch_loop)

    # Run radial version Calculation
    psf_layer = df_fourier.PSF_Layer_MatrixBroadband(prop_params_radial).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params_radial, [532e-9], [zfocus], zoffset)
    out_int_radial, out_phase_radial = psf_layer([lens_trans, lens_phase], point_source_locs, batch_loop=batch_loop)

    # # Make plot
    plot_psfs(out_int, out_phase, out_int_radial, out_phase_radial, savepath, "Matrix_ASM_Fourier")

    return


def call_propagation_layer(engine, device="cuda", batch_loop=False):
    savepath = "dflat/fourier_layer/test_scripts/output_propagation/"

    ### Define new propagation parameter
    propagation_settings = {
        # Mandatory Keys
        "wavelength_set_m": [450e-9, 532e-9, 700e-9],
        "ms_samplesM": {"x": 513, "y": 513},
        "ms_dx_m": {"x": 1e-6, "y": 1e-6},
        "radius_m": 256e-6,
        "sensor_distance_m": 6e-3,
        "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_number": {"x": 256, "y": 256},
        "radial_symmetry": False,
        "diffractionEngine": engine,
        # Optional Keys
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    prop_params = df_struct.prop_params(propagation_settings)
    propagation_settings["radial_symmetry"] = True
    prop_params_radial = df_struct.prop_params(propagation_settings)

    # run 2D Calculation
    propagator = df_fourier.Propagate_Planes_Layer(prop_params).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params, [532e-9], [1e6], [{"x": 0, "y": 0}])
    out_int, out_phase = propagator([lens_trans, lens_phase], batch_loop=batch_loop)

    # Run radial version Calculation
    propagator = df_fourier.Propagate_Planes_Layer(prop_params_radial).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params_radial, [532e-9], [1e6], [{"x": 0, "y": 0}])
    out_int_radial, out_phase_radial = propagator([lens_trans, lens_phase], batch_loop=batch_loop)
    out_int_radial = radial_2d_transform(out_int_radial.squeeze(1))
    out_phase_radial = radial_2d_transform_wrapped_phase(out_phase_radial.squeeze(1))

    print(out_int.shape, out_phase.shape, out_int_radial.shape, out_phase_radial.shape)

    # Make plot
    plot_fields(out_int, out_phase, out_int_radial, out_phase_radial, savepath, engine)

    return


def call_matrixASM_propagation_layer(device="cuda", batch_loop=False):
    savepath = "dflat/fourier_layer/test_scripts/output_propagation/"

    ### Define new propagation parameter
    propagation_settings = {
        # Mandatory Keys
        "wavelength_set_m": [450e-9, 532e-9, 700e-9],
        "ms_samplesM": {"x": 513, "y": 513},
        "ms_dx_m": {"x": 1e-6, "y": 1e-6},
        "radius_m": 256e-6,
        "sensor_distance_m": 6e-3,
        "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_number": {"x": 256, "y": 256},
        "radial_symmetry": False,
        "diffractionEngine": "ASM_fourier",
        # Optional Keys
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }
    prop_params = df_struct.prop_params(propagation_settings)
    propagation_settings["radial_symmetry"] = True
    prop_params_radial = df_struct.prop_params(propagation_settings)

    # run 2D Calculation
    propagator = df_fourier.Propagate_Planes_Layer_MatrixBroadband(prop_params).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params, [532e-9], [1e6], [{"x": 0, "y": 0}])
    out_int, out_phase = propagator([lens_trans, lens_phase], batch_loop=batch_loop)

    # Run radial version Calculation
    propagator = df_fourier.Propagate_Planes_Layer_MatrixBroadband(prop_params_radial).to(device)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params_radial, [532e-9], [1e6], [{"x": 0, "y": 0}])
    out_int_radial, out_phase_radial = propagator([lens_trans, lens_phase], batch_loop=batch_loop)
    out_int_radial = radial_2d_transform(out_int_radial.squeeze(1))
    out_phase_radial = radial_2d_transform_wrapped_phase(out_phase_radial.squeeze(1))

    # Make plot
    plot_fields(out_int, out_phase, out_int_radial, out_phase_radial, savepath, "Matrix_ASM_fourier")

    return


if __name__ == "__main__":
    # # Check the standard psf layer for PSF
    batch_loop = False

    call_propagation_layer("fresnel_fourier", batch_loop=batch_loop)
    call_propagation_layer("ASM_fourier", batch_loop=batch_loop)
    call_matrixASM_propagation_layer(batch_loop=batch_loop)

    call_psf_layer("fresnel_fourier", batch_loop=batch_loop)
    call_psf_layer("ASM_fourier", batch_loop=batch_loop)
    call_matrixASM_psf_layer(batch_loop=batch_loop)
