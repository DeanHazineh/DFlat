import os
import numpy as np
from scipy.special import j1
import matplotlib.pyplot as plt
from dflat.propagation import FresnelPropagation, ASMPropagation
from dflat.initialize import focusing_lens
from dflat.plot_utilities import axis_off


def airy_disk(
    wavelength, aperture_radius, sensor_distance, grid_size, grid_discretization
):
    """
    Compute the Airy disk diffraction pattern over a specified grid.

    Parameters:
        wavelength (float): Wavelength of the light (meters).
        aperture_radius (float): Radius of the aperture (meters).
        sensor_distance (float): Distance to the sensor (meters).
        grid_size (int): Number of points in one dimension of the grid.
        grid_discretization (float): Spacing between grid points (meters).

    Returns:
        numpy.ndarray: 2D array representing the Airy disk intensity.
    """
    # Create the spatial grid
    x = np.linspace(
        -grid_size * grid_discretization / 2,
        grid_size * grid_discretization / 2,
        grid_size,
    )
    y = np.linspace(
        -grid_size * grid_discretization / 2,
        grid_size * grid_discretization / 2,
        grid_size,
    )
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)  # Radial distance from the center in meters

    # Compute the Airy disk pattern
    k = 2 * np.pi / wavelength  # Wave number
    beta = k * aperture_radius * r / sensor_distance
    # Handle division by zero for the center point
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = (2 * j1(beta) / beta) ** 2
        intensity[np.isnan(intensity)] = 1.0  # Set the center point intensity to 1

    return intensity


def compare_airy_2D():
    wavelength_set_m = [400e-9, 500e-9, 600e-9, 700e-9]
    nl = len(wavelength_set_m)
    out_distance_m = 30e-3

    # Initialize ideal focusing lenses
    lens_settings = {
        "in_size": [501, 501],
        "in_dx_m": [1e-6, 1e-6],
        "wavelength_set_m": wavelength_set_m,
        "depth_set_m": [1e3 for _ in range(nl)],
        "fshift_set_m": [[0.0, 0.0] for _ in range(nl)],
        "out_distance_m": out_distance_m,
        "aperture_radius_m": 501e-6 / 2,
    }
    lamp, lph, lap = focusing_lens(**lens_settings)
    lamp = lamp * lap

    # Compute Field propagation
    prop_settings = {
        "in_size": [501, 501],
        "in_dx_m": [1e-6, 1e-6],
        "out_distance_m": out_distance_m,
        "out_size": [128, 128],
        "out_dx_m": [2e-6, 2e-6],
        "wavelength_set_m": wavelength_set_m,
    }
    psf_asm = ASMPropagation(**prop_settings, verbose=True)
    psf_fres = FresnelPropagation(**prop_settings, verbose=True)

    out_asm, _ = psf_asm(lamp[None], lph[None])
    int_asm = out_asm[0].cpu().numpy() ** 2
    int_asm = int_asm / np.max(int_asm, axis=(-1, -2), keepdims=True)

    out_fres, _ = psf_fres(lamp[None], lph[None])
    int_fres = out_fres[0].cpu().numpy() ** 2
    int_fres = int_fres / np.max(int_fres, axis=(-1, -2), keepdims=True)

    # Compute airy solution
    sol = []
    for lam in wavelength_set_m:
        sol.append(airy_disk(lam, 501e-6 / 2, out_distance_m, 128, 2e-6))
    sol = np.stack(sol)

    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(nl):
        ax[0, i].set_title(f"wl: {wavelength_set_m[i]*1e9}")
        ax[0, i].imshow(sol[i])
        ax[1, i].imshow(int_asm[i])
        ax[2, i].imshow(int_fres[i])
        axis_off(ax[0, i])
        axis_off(ax[1, i])
        axis_off(ax[2, i])

        ax[3, i].plot(sol[i, 64, :], "rx")
        ax[3, i].plot(int_asm[i, 64, :], "b-")
        ax[3, i].plot(int_fres[i, 64, :], "g-")

    ax[0, 0].set_ylabel("Airy Disk")
    ax[1, 0].set_ylabel("ASM")
    ax[2, 0].set_ylabel("Fresnel")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    savepath = script_dir + "/out/"
    plt.savefig(savepath + "propagation2D_vs_airy.png")
    plt.close()

    return


if __name__ == "__main__":
    compare_airy_2D()
