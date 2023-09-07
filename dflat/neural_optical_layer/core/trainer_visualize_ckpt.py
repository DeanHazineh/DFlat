import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import dflat.plot_utilities as df_plt

from caller_MLP import batched_broadband_MLP


def visualize_nanofins(model, epoch, device):
    wl_use = 532e-9
    lx, ly = np.arange(60e-9, 301e-9, 1e-9), np.arange(60e-9, 301e-9, 1e-9)
    Lx, Ly = np.meshgrid(lx, ly)
    init_shape = Lx.shape
    Lx = torch.tensor(Lx.flatten(), dtype=torch.float32).to(device)
    Ly = torch.tensor(Ly.flatten(), dtype=torch.float32).to(device)

    param_vec = torch.transpose(torch.stack(model.normalizeInput([Lx, Ly]), axis=0), 0, 1)
    torch_ones = torch.ones((param_vec.shape[0], 1), dtype=torch.float32, device=device)
    trans, phase = batched_broadband_MLP(param_vec, model, [wl_use], init_shape, torch_ones)
    trans = trans.cpu().numpy()
    phase = phase.cpu().numpy()

    # Plot the spatial slice
    fig = plt.figure(figsize=(7, 7))
    ax = df_plt.addAxis(fig, 2, 2)
    im1 = ax[0].imshow(trans[0, 0, :, :], vmin=0, vmax=1)
    df_plt.formatPlots(fig, ax[0], im1, ylabel="Spatial Slice Data", title="Transmittance", setAspect="auto", xgrid_vec=lx * 1e9, ygrid_vec=ly * 1e9)
    im2 = ax[1].imshow(phase[0, 0, :, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    df_plt.formatPlots(fig, ax[1], im2, title="Phase", setAspect="auto", xgrid_vec=lx * 1e9, ygrid_vec=ly * 1e9)

    # Plot wavelength slice
    yidx = 24
    wl_use = np.arange(310e-9, 751e-9, 1e-9)
    Lx, Ly = np.meshgrid(lx, ly[yidx])
    init_shape = Lx.shape

    Lx = torch.tensor(Lx.flatten(), dtype=torch.float32).to(device)
    Ly = torch.tensor(Ly.flatten(), dtype=torch.float32).to(device)
    param_vec = torch.transpose(torch.stack(model.normalizeInput([Lx, Ly]), axis=0), 0, 1)
    torch_ones = torch.ones((param_vec.shape[0], 1), dtype=torch.float32, device=device)
    Ll = torch.tensor(wl_use, dtype=torch.float32).to(device)

    trans, phase = batched_broadband_MLP(param_vec, model, Ll, init_shape, torch_ones)
    trans = trans.cpu().numpy()
    phase = phase.cpu().numpy()

    im3 = ax[2].imshow(trans[:, 0, :], vmin=0, vmax=1)
    df_plt.formatPlots(fig, ax[2], im3, ylabel="Spectral Slice Data", setAspect="auto", xgrid_vec=lx * 1e9, ygrid_vec=wl_use * 1e9)
    im4 = ax[3].imshow(phase[:, 0, :], vmin=-np.pi, vmax=np.pi, cmap="hsv")
    df_plt.formatPlots(fig, ax[3], im4, setAspect="auto", xgrid_vec=lx * 1e9, ygrid_vec=wl_use * 1e9)
    plt.tight_layout()
    # print(model._modelSavePath + f"trainingOutput/png_images/vis_{epoch}.png")
    plt.savefig(model._modelSavePath + f"trainingOutput/png_images/vis_{epoch}.png")
    plt.close()
    return


def visualize_nanocylinder(model, epoch, device):
    wl_use = np.arange(310e-9, 750e-9, 1e-9)
    r = np.arange(30e-9, 150e-9, 1e-9)
    r_use = torch.tensor(r[:, None], dtype=torch.float32).to(device)
    param_vec = model.normalizeInput([r_use])[0]

    trans, phase = batched_broadband_MLP(param_vec, model, wl_use, param_vec.shape)
    trans = trans.squeeze().cpu().numpy()
    phase = phase.squeeze().cpu().numpy()

    ### make plot
    fig = plt.figure(figsize=(10, 5))
    ax = df_plt.addAxis(fig, 1, 2)
    im0 = ax[0].imshow(trans, vmin=0, vmax=1)
    df_plt.formatPlots(fig, ax[0], im0, title="Transmittance", setAspect="auto", xgrid_vec=r * 1e9, ygrid_vec=wl_use * 1e9)
    im1 = ax[1].imshow(phase, vmin=-np.pi, vmax=np.pi, cmap="hsv")
    df_plt.formatPlots(fig, ax[1], im1, title="Phase", setAspect="auto", xgrid_vec=r * 1e9, ygrid_vec=wl_use * 1e9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.text(0.5, 0.90, f"Epoch: {epoch}", ha="center", va="center", fontsize=12)
    plt.savefig(model._modelSavePath + f"trainingOutput/png_images/vis_{epoch}.png")
    plt.close()

    return


def save_training_imgs_as_gifs(model_paths=None, model_tag=None):
    if model_paths is None:
        model_paths = "dflat/neural_optical_layer/core/trained_MLP_models/"
        all_models = os.listdir(model_paths)
        if model_tag is not None:
            all_models = [name for name in all_models if model_tag in name]
    else:
        model_paths = model_paths if isinstance(model_paths, list) else [model_paths]

    for fold in all_models:
        print(fold)
        img_path = model_paths + fold + "/trainingOutput/png_images/"
        df_plt.gif_from_saved_images(img_path, "vis", "trainingGif60fps.gif", fps=90, deleteFrames=False, verbose=False)

    return
