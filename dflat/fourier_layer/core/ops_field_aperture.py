import numpy as np
import torch
from .ops_transform_util import radial_2d_transform


def gen_aperture_disk(parameters):
    """Generate a circular field aperture at the input plane (metasurface plane) grid specified in parameters.

    If parameters["radius_m"]==None, a unity transmittance aperture matching the metasurface grid is returned.

    Args:
        `parameters` (prop_param): Settings object defining field propagation details.

    Returns:
        `np.float`: Field aperture transmittance with shape (1, ms_samplesM['y'], ms_samplesM['x']) or (1, 1, ms_samplesM['r']).
        `np.float`: Sqrt of the total energy transmitted through the aperture
    """

    # unpack parameters
    radius_um = parameters["radius_um"]  # May be None
    ms_samplesM = parameters["ms_samplesM"]
    ms_dx_um = parameters["ms_dx_um"]
    radial_symmetry = parameters["radial_symmetry"]

    # define coordinate on the metasurface plane
    xx, yy = np.meshgrid(np.arange(ms_samplesM["x"]), np.arange(ms_samplesM["y"]))
    xx = xx - (xx.shape[1] - 1) / 2
    yy = yy - (yy.shape[0] - 1) / 2
    xx = xx * ms_dx_um["x"]
    yy = yy * ms_dx_um["y"]

    # Initialize aperture as uniform square or disk (add small constant to avoid 0)
    aperture_trans = np.ones_like(xx)
    if radius_um is not None:
        aperture_trans = ((np.sqrt(xx**2 + yy**2) <= radius_um)).astype(np.float32) + 1e-6
    aperture_trans = np.expand_dims(aperture_trans, 0)

    # Get the total incident energy for normalization calls downstream
    sqrt_energy_illum = sqrt_energy_illumination(aperture_trans, ms_dx_um, False)

    # return radial vector if radial_symmetry==true
    if radial_symmetry:
        ms_samplesM_r = ms_samplesM["r"]
        aperture_trans = aperture_trans[0:1, ms_samplesM_r - 1 : ms_samplesM_r, ms_samplesM_r - 1 :]

    return aperture_trans, sqrt_energy_illum


def sqrt_energy_illumination(aperture_trans, pixel_size, radial_flag):
    """Approximates the total transmitted energy through a user defined aperture.

    Args:
        `aperture_trans` (np.float): Aperture transmittance of shape (M, Ny, Nx) or (M, 1, Nr)
        `pixel_size` (dict): Pitch (in meters) dx, dy of the aperture profile grid defined via a dict
            {"x" : np.float, "y" : np.float}.
        `radial_flag` (bool): Boolean flag defining if input aperture_trans is a radial profile or 2D.

    Returns:
        `np.float`: Sqrt of the total energy transmitted through the aperture of shape (M, 1, 1)
    """
    # note:
    # one may instead compute the area from a radial vector via a 2*pi*r*dr summation but calling radial_2d_transform
    # is more favorable since the radial psf is later converted to an unnormalized 2D psf by the same method

    if radial_flag:
        aperture_trans = np.squeeze(radial_2d_transform(torch.tensor(aperture_trans)).numpy(), 1)

    return np.expand_dims(
        np.expand_dims(
            np.sqrt(np.sum(aperture_trans**2, axis=(1, 2)) * pixel_size["x"] * pixel_size["y"]),
            -1,
        ),
        -1,
    )
