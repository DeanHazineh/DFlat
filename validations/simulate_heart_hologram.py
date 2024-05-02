import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os

from dflat.propagation import ASMPropagation, FresnelPropagation


def call_simulate_heart_singularity(engine):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    savepath = script_dir + "/"

    ################################################
    ### Make a plot of the experimental measurements
    dataLoc = savepath + "heart_singularity_experiment.pickle"
    with open(dataLoc, "rb") as f:
        data = pickle.load(f)

    phases = data["phases"]
    intensity = data["intensity"]

    int_z = np.array(data["int_z"], dtype=float)
    int_x = np.array(data["int_x"], dtype=float) * 1e6 - 36.5
    int_y = np.array(data["int_y"], dtype=float) * 1e6 - 24.5
    phase_x = np.array(data["phase_x"], dtype=float) * 1e6 - 36.5
    phase_y = np.array(data["phase_y"], dtype=float) * 1e6 - 24.5
    num_sensor_distances = len(int_z)

    # Match Daniels normalizations
    _, phase_cx_idx = min(
        (val, idx) for (idx, val) in enumerate(np.abs(phase_x.flatten()))
    )
    _, phase_cy_idx = min(
        (val, idx) for (idx, val) in enumerate(np.abs(phase_y.flatten()))
    )
    plotPhase = phases[:, :, :] - phases[phase_cy_idx, phase_cx_idx, :]
    plotPhase = np.arctan2(np.sin(plotPhase), np.cos(plotPhase))

    ### Generate Plot of Experimental Fields
    fig, ax = plt.subplots(
        2, num_sensor_distances, figsize=(3 * num_sensor_distances, 3 * 2)
    )
    for i in range(num_sensor_distances):
        plt_int = ax[0, i].imshow(
            np.log10(intensity[:, :, i]),
            extent=(np.min(int_x), np.max(int_x), np.min(int_y), np.max(int_y)),
            origin="lower",
            cmap="viridis",
        )
        plt_pha = ax[1, i].imshow(
            plotPhase[:, :, i],
            extent=(np.min(phase_x), np.max(phase_x), np.min(phase_y), np.max(phase_y)),
            origin="lower",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
    for axi in ax.flatten():
        axi.axis("off")
    plt.tight_layout()
    plt.savefig(savepath + "out/experiment.png")
    plt.close()

    ################################################
    ### Load in the validated heart singularity metasurface data and parameter file
    params_loc = savepath + "metasurface_heart_lens.pickle"
    with open(params_loc, "rb") as handle:
        exSimDict = pickle.load(handle)
    ms_phase = exSimDict["ms_phase_x"][None, None]
    ms_trans = exSimDict["ms_trans"][None, None]

    ### create a new fourier prop_params using the heart parameters file (matched to experimental data)
    downsample_factor = 4
    settings = {
        "in_size": [101, 101],
        "in_dx_m": [8.0e-6, 8.0e-6],
        "out_distance_m": 10e-3,
        "out_size": [int(1001 / downsample_factor), int(1001 / downsample_factor)],
        "out_dx_m": [6.0e-8 * downsample_factor, 6.0e-8 * downsample_factor],
        "out_resample_dx_m": None,
        "manual_upsample_factor": 1,
        "radial_symmetry": False,
        "FFTPadFactor": 1.0,
        "wavelength_set_m": [532e-9],
    }

    ### Propagate the fields, looping over sensor distances
    propagator = ASMPropagation if engine == "asm" else FresnelPropagation
    sensor_distance_m = [9.6e-3, 9.8e-3, 10.0e-3, 10.2e-3, 10.4e-3]
    ampl_stack = []
    phase_stack = []
    for dist in sensor_distance_m:
        print("Calculating: ", dist)
        settings["out_distance_m"] = dist

        # Initialize the field propagator layer
        field_propagator = propagator(**settings)
        out = field_propagator(ms_trans, ms_phase)
        ampl_stack.append(out[0].cpu().numpy())
        phase_stack.append(out[1].cpu().numpy())

    ampl_stack = np.stack(ampl_stack, axis=0).squeeze()
    phase_stack = np.stack(phase_stack, axis=0).squeeze()

    ### Generate Plot of Calculated Fields
    out_size = settings["out_size"]
    cidx, cidy = out_size[0] // 2, out_size[1] // 2
    plotPhase = phase_stack - phase_stack[:, cidy : cidy + 1, cidx : cidx + 1]
    plotPhase = np.arctan2(np.sin(plotPhase), np.cos(plotPhase))

    nz = len(sensor_distance_m)
    fig, ax = plt.subplots(2, nz, figsize=(3 * nz, 3 * 2))
    for i in range(len(sensor_distance_m)):
        plt_int = ax[0, i].imshow(
            np.log10(ampl_stack[i] ** 2),
            cmap="viridis",
        )
        plt_pha = ax[1, i].imshow(
            plotPhase[i],
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
    plt.savefig(savepath + "out/dflat_" + engine + ".png")
    plt.close()

    return


if __name__ == "__main__":
    call_simulate_heart_singularity("fresnel")
    call_simulate_heart_singularity("asm")
