# The QDHT code included here is a pytorch port/implementation inspired by the
# pyhank algorithm published by (https://github.com/etfrogers/pyhank Edward Rogers)

import numpy as np
import scipy.special as scipy_bessel
import torch
from .radial_ops import general_interp_regular_1d_grid


def iqdht(k_grid, fr, order=0):
    """Computes the inverse quasi-discrete Hankel transform for radial tensor, on the inner-most two dimensions

    Args:
        `k_grid` (tf.float): tensor corresponding to the angular frequency vector
        `fr` (`tf.float` or `tf.complex`): Field values on the radial grid of shape (..., 1, Nx).
        `order` (int, optional): Order of the inverse Hankel transform. Defaults to 0.

    Returns:
        `tf.float`: Radial grid corresponding to the iqdh-transformed data.
        `fr.dtype`: Inverse Hankel transform of the input data fr of shape (..., 1, Nx).
    """
    kr, ht = qdht(k_grid / 2 / np.pi, fr, order)

    return kr / 2 / np.pi, ht


def qdht(radial_grid, fr, order=0):
    """Implements a quasi-discrete Hankel transform for radial data on the inner-most two dimensions of the input signal.

    Args:
        radial_grid (float): 1D tensor of length N corresponding to the radial grid coordinates of length (Nr).
        fr (float or complex): Real or complex field values on the radial grid of shape (.., 1, Nr).
        order (int, optional): Order of the Hankel transform. Defaults to 0.

    Returns:
        float: tensor corresponding to the angular frequency vector
        float or complex: Hankel transform of the input data fr of shape (.., 1, Nr)
    """

    if not torch.is_tensor(fr):
        fr = torch.from_numpy(fr)
    device = fr.device

    if not torch.is_tensor(radial_grid):
        radial_grid = torch.from_numpy(radial_grid)

    if radial_grid.device != fr.device:
        radial_grid = radial_grid.to(device)

    # Assert that the spatial grid is 1D
    radial_grid = torch.squeeze(radial_grid)
    assert radial_grid.dim() == 1, "QDHT: Radial grid should be a 1D vector."

    # If the input signal tensor rank is not 3, then readjust for the calculation. In the end, we reshape back then return.
    input_rank = fr.dim()
    init_shape = fr.shape
    if input_rank == 1:
        fr = fr[None, None]
    elif input_rank == 2:
        fr = fr[None]
    elif input_rank > 3:
        fr = fr.view(-1, *init_shape[-2:])

    ### Create the transformation matrix
    # Calculate N+1 roots; must be calculated before max_radius can be derived from k_grid
    xdtype = radial_grid.dtype
    sdtype = fr.dtype
    n_points = len(radial_grid)
    alpha = scipy_bessel.jn_zeros(order, n_points + 1)
    alpha = alpha[0:-1]
    alpha_n1 = alpha[-1]

    # Calculate coordinate vectors
    max_radius = radial_grid.max().item()
    r = alpha * max_radius / alpha_n1
    v = alpha / (2 * np.pi * max_radius)
    kr = 2 * np.pi * v
    kr = torch.tensor(kr, dtype=xdtype).to(device)
    v_max = alpha_n1 / (2 * np.pi * max_radius)
    S = alpha_n1

    # Calculate hankel matrix and vectors
    jp = scipy_bessel.jv(
        order, np.matmul(np.expand_dims(alpha, -1), np.expand_dims(alpha, 0)) / S
    )
    jp = torch.tensor(jp, dtype=sdtype).to(device)
    jp1 = torch.tensor(np.abs(scipy_bessel.jv(order + 1, alpha)), dtype=sdtype).to(
        device
    )
    T = 2 * jp / torch.matmul(jp1.unsqueeze(-1), jp1.unsqueeze(0)) / S
    JR = jp1 / max_radius
    JV = jp1 / v_max

    # Reinterpolate the input data and compute hankel
    r = torch.tensor(alpha * max_radius / alpha_n1, dtype=xdtype).to(device)
    f_transform = general_interp_regular_1d_grid(radial_grid, r, fr)
    hankel_trans = f_transform / JR
    hankel_trans = JV.unsqueeze(-1) * torch.matmul(T, hankel_trans.unsqueeze(-1))
    hankel_trans = hankel_trans.squeeze(-1)

    # reshape the hankel transformed signal back to the user input batch_size
    if input_rank != 3:
        hankel_trans = hankel_trans.view(*init_shape)

    return kr, hankel_trans
