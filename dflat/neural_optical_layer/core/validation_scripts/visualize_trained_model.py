import matplotlib.pyplot as plt
import numpy as np
import torch

import dflat.metasurface_library as df_library
import dflat.neural_optical_layer as df_neural
import dflat.plot_utilities as df_plt


def plot_MLP_Nanocylinders(model_name, show=False):
    savepath = "dflat/neural_optical_layer/validation_scripts/output/"
    use_dtype = torch.float64  # This is the output dtype
    neural_model = df_neural.MLP_Layer(model_name, dtype=use_dtype)

    ### Get nanofin library dataset and the corresponding neural model (wrapped as a mlp layer)
    library = df_library.Nanocylinders_U180nm_H600nm()
    lib_trans = library.transmittance
    lib_phase = library.phase
    lib_phase = np.arctan2(np.sin(lib_phase), np.cos(lib_phase))  # this is the phase wrap used during training time
    r, wl = library.param1, library.param2

    ### Get the true data used in training and reformat it back to grid
    wl_use = np.arange(np.min(wl), np.max(wl), 1e-9)
    r_use = np.arange(np.min(r), np.max(r), 1e-9)
    r_use = torch.tensor(r_use, dtype=use_dtype)[None, None]  # The neural layer is used for lenses so lets tweak the shape
    param_vec = neural_model.shape_to_param(r_use)
    trans, phase = neural_model(param_vec, wl_use)
    trans = torch.squeeze(trans)
    phase = torch.squeeze(phase)

    ### make plot
    fig = plt.figure(figsize=(10, 10))
    axisList = df_plt.addAxis(fig, 2, 2)
    axisList[0].imshow(
        lib_trans,
        extent=(np.min(r * 1e9), np.max(r * 1e9), np.max(wl * 1e9), np.min(wl * 1e9)),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    im1 = axisList[1].imshow(
        lib_phase,
        extent=(np.min(r * 1e9), np.max(r * 1e9), np.max(wl * 1e9), np.min(wl * 1e9)),
        vmin=-np.pi,
        vmax=np.pi,
        aspect="auto",
    )

    im2 = axisList[2].imshow(
        trans,
        extent=(np.min(r * 1e9), np.max(r * 1e9), np.max(wl * 1e9), np.min(wl * 1e9)),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    im3 = axisList[3].imshow(
        phase,
        extent=(np.min(r * 1e9), np.max(r * 1e9), np.max(wl * 1e9), np.min(wl * 1e9)),
        vmin=-np.pi,
        vmax=np.pi,
        aspect="auto",
    )

    if show == False:
        plt.savefig(savepath + f"{model_name}.png")
        plt.close()
    else:
        plt.show()

    return


def plot_MLP_Nanofins(model_name, show=False):
    savepath = "dflat/neural_optical_layer/core/validation_scripts/output/"
    use_dtype = torch.float64  # This is the output dtype
    neural_model = df_neural.MLP_Layer(model_name, dtype=use_dtype)

    ### Get nanofin library dataset and the corresponding neural model (wrapped as a mlp layer)
    library = df_library.Nanofins_U350nm_H600nm()
    lib_trans = library.transmittance
    lib_phase = library.phase
    lib_phase = np.arctan2(np.sin(lib_phase), np.cos(lib_phase))  # this is the phase wrap used during training time
    wx, wy, wl = library.param1, library.param2, library.param3

    for wl_use in [532e-9]:
        wl_idx = np.argmin(np.abs(wl - wl_use))
        Lx, Ly = np.meshgrid(np.arange(60e-9, 301e-9, 1e-9), np.arange(60e-9, 301e-9, 1e-9))
        shape_vec = torch.tensor(np.stack((Lx, Ly), axis=0), dtype=use_dtype)
        param_vec = neural_model.shape_to_param(shape_vec)
        trans, phase = neural_model(param_vec, [wl_use])

        fig = plt.figure(figsize=(20, 20))
        ax = df_plt.addAxis(fig, 2, 2)
        im1 = ax[0].imshow(lib_trans[0, :, :, wl_idx], vmin=0, vmax=1)
        im2 = ax[1].imshow(lib_trans[1, :, :, wl_idx], vmin=0, vmax=1)
        im3 = ax[2].imshow(trans[0, 0, :, :], vmin=0, vmax=1)
        im4 = ax[3].imshow(trans[0, 1, :, :], vmin=0, vmax=1)
        df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True)
        df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True, rmvyLabel=True)
        df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9)
        df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9, rmvyLabel=True)
        plt.savefig(savepath + f"{model_name}_spatial_slice_trans_{wl_use*1e9}.png")

        fig = plt.figure(figsize=(20, 20))
        ax = df_plt.addAxis(fig, 2, 2)
        im1 = ax[0].imshow(lib_phase[0, :, :, wl_idx], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        im2 = ax[1].imshow(lib_phase[1, :, :, wl_idx], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        im3 = ax[2].imshow(phase[0, 0, :, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        im4 = ax[3].imshow(phase[0, 1, :, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
        df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True)
        df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wy * 1e9, rmvxLabel=True, rmvyLabel=True)
        df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9)
        df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ly[:, 0] * 1e9, rmvyLabel=True)
        plt.savefig(savepath + f"{model_name}_spatial_slice_phase_{wl_use*1e9}.png")

    ### Plot wavelength slice
    yidx = 24
    Lx, Ly = np.meshgrid(np.arange(60e-9, 301e-9, 1e-9), wy[yidx])
    shape_vec = torch.tensor(np.stack((Lx, Ly), axis=0), dtype=use_dtype)
    param_vec = neural_model.shape_to_param(shape_vec)
    Ll = np.arange(310e-9, 751e-9, 1e-9)
    trans, phase = neural_model(param_vec, Ll)

    fig = plt.figure(figsize=(15, 15))
    ax = df_plt.addAxis(fig, 2, 2)
    im1 = ax[0].imshow(lib_trans[0, 24, :, :].T, vmin=0, vmax=1)
    im2 = ax[1].imshow(lib_trans[1, 24, :, :].T, vmin=0, vmax=1)
    im3 = ax[2].imshow(trans[:, 0, 0, :], vmin=0, vmax=1)
    im4 = ax[3].imshow(trans[:, 1, 0, :], vmin=0, vmax=1)
    df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True)
    df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True, rmvyLabel=True)
    df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9)
    df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9, rmvyLabel=True)
    if not show:
        plt.savefig(savepath + f"{model_name}_spectral_slice_trans.png")

    fig = plt.figure(figsize=(15, 15))
    ax = df_plt.addAxis(fig, 2, 2)
    im1 = ax[0].imshow(lib_phase[0, 24, :, :].T, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    im2 = ax[1].imshow(lib_phase[1, 24, :, :].T, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    im3 = ax[2].imshow(phase[:, 0, 0, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    im4 = ax[3].imshow(phase[:, 1, 0, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    df_plt.formatPlots(fig, ax[0], im1, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True)
    df_plt.formatPlots(fig, ax[1], im2, setAspect="auto", xgrid_vec=wx * 1e9, ygrid_vec=wl * 1e9, rmvxLabel=True, rmvyLabel=True)
    df_plt.formatPlots(fig, ax[2], im3, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9)
    df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=Lx[0, :] * 1e9, ygrid_vec=Ll * 1e9, rmvyLabel=True)
    if not show:
        plt.savefig(savepath + f"{model_name}_spectral_slice_phase.png")
        plt.close()
    else:
        plt.show()

    return


if __name__ == "__main__":
    # Run plotting functions for mlp model
    plot_MLP_Nanofins("MLP_Nanofins_Dense1024_U350_H600_SIREN100", show=True)
    plot_MLP_Nanocylinders("MLP_Nanocylinders_Dense256_U180_H600_SIREN100")
