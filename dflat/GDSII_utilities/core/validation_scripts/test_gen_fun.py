import numpy as np
import torch

from dflat.neural_optical_layer import load_neuralModel


def call_nanofin_gds():
    device = "cuda"
    this_mlp = load_neuralModel("MLP_Nanofins_Dense1024_U350_H600_SIREN100", torch.float32).to(device)

    Nx, Ny = 5, 5
    param_array = np.random.rand(2, Ny, Nx)
    ms_dx_m = {"x": 1 * 350e-9, "y": 1 * 350e-9}
    savepath = "dflat/GDSII_utilities/core/validation_scripts/output/nanofin_"
    this_mlp.write_param_to_gds(param_array, ms_dx_m, savepath, aperture=None, rotation_array=None, add_markers=True)

    return


def call_nanocylinder_gds():
    device = "cuda"
    this_mlp = load_neuralModel("MLP_Nanocylinders_Dense256_U180_H600_SIREN100", torch.float32).to(device)

    Nx, Ny = 5, 5
    param_array = np.random.rand(1, Ny, Nx)
    ms_dx_m = {"x": 1 * 180e-9, "y": 1 * 180e-9}
    savepath = "dflat/GDSII_utilities/core/validation_scripts/output/nanocylinder_"
    this_mlp.write_param_to_gds(param_array, ms_dx_m, savepath, aperture=None, rotation_array=None, add_markers=True)

    return


if __name__ == "__main__":
    call_nanofin_gds()
    call_nanocylinder_gds()
