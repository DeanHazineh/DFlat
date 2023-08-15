import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

import dflat.plot_utilities as df_plot
import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier


def call_simulate_heart_singularity(engine):
    savepath = "dflat/fourier_layer/test_scripts/heart_singularity_hologram/"

    ################################################
    ### Make a plot of the experimental measurements
    dataLoc = "dflat/fourier_layer/test_scripts/heart_singularity_hologram/reprocessed_Daniel_experimentalData.mat"
    data = sio.loadmat(dataLoc)
    phases = data["phases"]
    intensity = data["intensity"]

    int_z = np.array(data["int_z"], dtype=float)
    int_x = np.array(data["int_x"], dtype=float) * 1e6 - 36.5
    int_y = np.array(data["int_y"], dtype=float) * 1e6 - 24.5
    phase_x = np.array(data["phase_x"], dtype=float) * 1e6 - 36.5
    phase_y = np.array(data["phase_y"], dtype=float) * 1e6 - 24.5
    num_sensor_distances = len(int_z)

    # Match Daniels normalizations
    _, phase_cx_idx = min((val, idx) for (idx, val) in enumerate(np.abs(phase_x.flatten())))
    _, phase_cy_idx = min((val, idx) for (idx, val) in enumerate(np.abs(phase_y.flatten())))
    plotPhase = phases[:, :, :] - phases[phase_cy_idx, phase_cx_idx, :]
    plotPhase = np.arctan2(np.sin(plotPhase), np.cos(plotPhase))

    ### Generate Plot of Experimental Fields
    fig = plt.figure(figsize=(10, num_sensor_distances * 5))
    ax = df_plot.addAxis(fig, num_sensor_distances, 2)
    axcounter = 0
    for i in range(num_sensor_distances):
        plt_int = ax[axcounter].imshow(
            np.log10(intensity[:, :, i]),
            extent=(np.min(int_x), np.max(int_x), np.min(int_y), np.max(int_y)),
            origin="lower",
            cmap="viridis",
        )
        plt_pha = ax[axcounter + 1].imshow(
            plotPhase[:, :, i],
            extent=(np.min(phase_x), np.max(phase_x), np.min(phase_y), np.max(phase_y)),
            origin="lower",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )

        axcounter += 2
    plt.savefig(savepath + "experiment.png")
    plt.close()

    ################################################
    ### Load in the validated heart singularity metasurface data and parameter file
    params_loc = "dflat/fourier_layer/test_scripts/heart_singularity_hologram/metasurface_heart_parameters.pickle"
    with open(params_loc, "rb") as handle:
        exSimDict = pickle.load(handle)
    ms_phase = np.expand_dims(exSimDict["ms_phase_x"], 0)
    ms_trans = np.expand_dims(exSimDict["ms_trans"], 0)

    ### create a new fourier prop_params using the heart parameters file (matched to experimental data)
    downsample_factor = 4
    simulationSettings = {
        "wavelength_set_m": [532e-9],
        "ms_samplesM": {"x": 101, "y": 101},
        "ms_dx_m": {"x": 8.0e-6, "y": 8.0e-6},
        "radius_m": None,
        "sensor_distance_m": 10e-3,  # placeholder to loop over values
        "initial_sensor_dx_m": {"x": 6.0e-8 * downsample_factor, "y": 6.0e-8 * downsample_factor},
        "sensor_pixel_size_m": {"x": 6.0e-8 * downsample_factor, "y": 6.0e-8 * downsample_factor},
        "sensor_pixel_number": {"x": int(1001 / downsample_factor), "y": int(1001 / downsample_factor)},
        "radial_symmetry": False,
        "diffractionEngine": engine,
        ### Optional
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    }

    ### Propagate the fields, looping over sensor distances
    sensor_distance_m = [9.6e-3, 9.8e-3, 10.0e-3, 10.2e-3, 10.4e-3]
    ampl_stack = []
    phase_stack = []
    for dist in sensor_distance_m:
        print("Calculating: ", dist)
        simulationSettings["sensor_distance_m"] = dist
        parameters = df_struct.prop_params(simulationSettings, verbose=False)

        # Initialize the field propagator layer
        field_propagator = df_fourier.Propagate_Planes_Layer(parameters)
        out = field_propagator([ms_trans, ms_phase])
        ampl_stack.append(out[0])
        phase_stack.append(out[1])

    ampl_stack = np.stack(ampl_stack, axis=0).transpose([0, 1, 2, 4, 3])
    phase_stack = np.stack(phase_stack, axis=0).transpose([0, 1, 2, 4, 3])

    ### Generate Plot of Calculated Fields
    xvec, yvec = df_plot.get_detector_pixel_coordinates(parameters)
    _, phase_cx_idx = min((val, idx) for (idx, val) in enumerate(np.abs(xvec.flatten())))
    _, phase_cy_idx = min((val, idx) for (idx, val) in enumerate(np.abs(yvec.flatten())))
    plotPhase = phase_stack[:, 0, 0, :, :] - np.expand_dims(np.expand_dims(phase_stack[:, 0, 0, phase_cy_idx, phase_cx_idx], 1), 1)
    plotPhase = np.arctan2(np.sin(plotPhase), np.cos(plotPhase))

    fig = plt.figure(figsize=(10, len(sensor_distance_m) * 5))
    ax = df_plot.addAxis(fig, len(sensor_distance_m), 2)
    axcounter = 0
    for i in range(len(sensor_distance_m)):
        plt_int = ax[axcounter].imshow(
            np.log10(ampl_stack[i, 0, 0, :, :] ** 2),
            extent=(np.min(xvec), np.max(xvec), np.min(yvec), np.max(yvec)),
            origin="lower",
            cmap="viridis",
        )
        plt_pha = ax[axcounter + 1].imshow(
            plotPhase[i, :, :],
            extent=(np.min(xvec), np.max(xvec), np.min(yvec), np.max(yvec)),
            origin="lower",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axcounter += 2
    plt.savefig(savepath + "dflat_" + engine + ".png")
    plt.close()

    return


if __name__ == "__main__":
    call_simulate_heart_singularity("fresnel_fourier")
    call_simulate_heart_singularity("ASM_fourier")
