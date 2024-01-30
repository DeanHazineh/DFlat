import numpy as np
import matplotlib.pyplot as plt


def vis_hyperspectral_image(hsi_img, rgb, dispN=5, figsize=(5, 5)):
    assert len(hsi_img.shape) == 3, "HSI image should have shape [H, W, C]."
    assert len(rgb.shape) == 3, "RGB image should have shape [H, W, 3]"
    assert rgb.shape[-1] == 3, "RGB image should have shape [H, W, 3]"
    H, W, C = hsi_img.shape

    slices = np.linspace(0, C - 1, dispN).astype(int)
    hsi_img = hsi_img[:, :, slices]
    C = dispN

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for i, pos in enumerate(range(C)):
        X, Y = np.meshgrid(range(W), range(H))
        Z = np.full((H, W), pos - 1)
        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=plt.cm.gray(hsi_img[:, :, i]),
            shade=False,
        )

    X, Y = np.meshgrid(range(W), range(H))
    Z = np.full((W, H), C - 1)  # Position the RGB image at the top
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb, shade=False)

    ax.axis("off")
    ax.view_init(elev=30.0, azim=180)

    return
