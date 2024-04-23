import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from moviepy.editor import ImageSequenceClip


def video_from_saved_images(
    filepath, filetag, savename, fps, deleteFrames=True, verbose=False
):
    print("Call video generator")
    png_files = natsorted(
        [
            os.path.join(filepath, f)
            for f in os.listdir(filepath)
            if f.startswith(filetag) and f.endswith(".png")
        ]
    )
    if verbose:
        for file in png_files:
            print("Adding image file as frame: " + file)

    clip = ImageSequenceClip(png_files, fps=fps)
    clip.write_videofile(
        os.path.join(filepath, savename) + ".mp4", codec="libx264", fps=fps
    )

    if deleteFrames:
        for file in png_files:
            os.remove(file)

    return


def gif_from_saved_images(
    filepath, filetag, savename, fps, deleteFrames=True, verbose=False, loop=0
):
    print("Call GIF generator")
    images = []
    png_files = natsorted(
        [
            f
            for f in os.listdir(filepath)
            if f.startswith(filetag) and f.endswith(".png")
        ]
    )
    for file in png_files:
        file_path = os.path.join(filepath, file)
        images.append(Image.open(file_path))
        if verbose:
            print("Write image file as frame: " + file)
        if deleteFrames:
            os.remove(file_path)

    duration = int(1000 / fps)
    images[0].save(
        filepath + savename + ".gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )
    return


def plot_3d_stack(img, rgb, zoom, save_to, show=True, view=[12, 12], crop_dim=None):
    img = np.clip(np.flipud(img), 0, 1)
    rgb = np.clip(np.flipud(rgb), 0, 1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    H, W, C = img.shape
    Y, Z = np.meshgrid(range(W), range(H))
    Y, Z = Y / W, Z / H
    X = np.ones_like(Y)
    for i in range(C - 1):
        ax.plot_surface(
            X / (C - 1) * i,
            Y,
            Z,
            cstride=1,
            rstride=1,
            facecolors=plt.cm.gray(img[:, :, i]),
            shade=False,
        )
        corners_x = [
            X[0, 0] / (C - 1) * i,
            X[0, -1] / (C - 1) * i,
            X[-1, -1] / (C - 1) * i,
            X[-1, 0] / (C - 1) * i,
            X[0, 0] / (C - 1) * i,
        ]
        corners_y = [Y[0, 0], Y[0, -1], Y[-1, -1], Y[-1, 0], Y[0, 0]]
        corners_z = [Z[0, 0], Z[0, -1], Z[-1, -1], Z[-1, 0], Z[0, 0]]
        ax.plot(corners_x, corners_y, corners_z, color="k")

    ax.plot_surface(X, Y, Z, cstride=1, rstride=1, facecolors=rgb, shade=False)
    corners_x = [X[0, 0], X[0, -1], X[-1, -1], X[-1, 0], X[0, 0]]
    ax.plot(corners_x, corners_y, corners_z, color="k")
    ax.axis("off")
    ax.view_init(elev=view[0], azim=view[1])

    ax.set_xlim([0, zoom])
    ax.set_ylim([0, zoom])
    ax.set_zlim([0, zoom + 0.02])
    ax.set_aspect("equal")

    if show:
        plt.show()
    else:
        plt.savefig(save_to, transparent=True)
        plt.close()

    return ax
