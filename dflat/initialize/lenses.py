import numpy as np


def focusing_lens(
    in_size,
    in_dx_m,
    wavelength_set_m,
    depth_set_m,
    fshift_set_m,
    out_distance_m,
    aperture_radius_m=None,
    radial_symmetry=False,
):
    """Generate the amplitude and phase profiles for a focusing lens. Returns a different profile for each wavelength and depth condition.

    Args:
        in_size (list): Lens grid size [H, W].
        in_dx_m (list): Lens grid discretizion [dy, dx].
        wavelength_set_m (list): List of wavelengths for each focusing profile.
        depth_set_m (list): List of fronto-planar depths for each focusing profile.
        fshift_set_m (list): Nested list of sensor-side focal spot shifts for each lens [[offy, offx],...].
        out_distance_m (float): Output plane distance from lens.
        aperture_radius_m (float, optional): Radius of an aperture profile to return. Defaults to None.
        radial_symmetry (bool, optional): If true, return radial vector describing the profile. Defaults to False.

    Returns:
        List: Amplitude, Phase, and Aperture profiles of shape [Batch, H, W] where batch equals length of ''_set_m.
    """

    wavelength_set_m = np.array(wavelength_set_m)
    depth_set_m = np.array(depth_set_m)
    fshift_set_m = np.array(fshift_set_m)

    assert (
        len(depth_set_m) == len(fshift_set_m) == len(wavelength_set_m)
    ), "wavelength_set_m, depth_set_m, and fshift_set_m must all have the same number of elements in the list."
    assert len(fshift_set_m.shape) == 2 and fshift_set_m.shape[-1] == 2
    assert len(in_size) == len(in_dx_m) == 2
    assert isinstance(out_distance_m, float)
    assert isinstance(aperture_radius_m, float) or aperture_radius_m is None
    assert isinstance(radial_symmetry, bool)
    if radial_symmetry:
        assert in_size[-1] == in_size[-2]
        assert in_dx_m[-1] == in_dx_m[-2]
        assert in_size[-1] % 2 != 0
        assert in_size[-2] % 2 != 0

    # (Batch, H, W)
    x, y = np.meshgrid(np.arange(in_size[-1]), np.arange(in_size[-2]), indexing="xy")
    x = x - (x.shape[-1] - 1) / 2
    y = y - (y.shape[-2] - 1) / 2
    x = x[None] * in_dx_m[-1]
    y = y[None] * in_dx_m[-2]

    wavelength_set = wavelength_set_m[:, None, None]
    depth_set = depth_set_m[:, None, None]
    offset_x = fshift_set_m[:, -1][:, None, None]
    offset_y = fshift_set_m[:, -2][:, None, None]

    phase = (
        -2
        * np.pi
        / wavelength_set
        * (
            np.sqrt(depth_set**2 + x**2 + y**2)
            + np.sqrt(out_distance_m**2 + (x - offset_x) ** 2 + (y - offset_y) ** 2)
        )
    )
    phase = np.angle(np.exp(1j * phase))
    ampl = np.ones_like(phase)
    aperture = (
        ((np.sqrt(x**2 + y**2) <= aperture_radius_m)).astype(np.float32) + 1e-6
        if aperture_radius_m is not None
        else np.ones_like(phase)
    )

    if radial_symmetry:
        cidx = in_size[-1] // 2
        ampl = ampl[:, cidx : cidx + 1, cidx:]
        phase = phase[:, cidx : cidx + 1, cidx:]
        aperture = aperture[:, cidx : cidx + 1, cidx:]

    return ampl, phase, aperture


## This old code was confusing and produced unnecssary batch dimensions that didn't match the ultimate use case
## So it is replaced with the above code
# def focusing_lens_all(
#     in_size,
#     in_dx_m,
#     wavelength_set_m,
#     depth_set_m,
#     fshift_set_m,
#     out_distance_m,
#     aperture_radius_m=None,
#     radial_symmetry=False,
# ):
#     """Generate the amplitude and phase profiles for a focusing lens. Returns a different profile for each wavelength and depth condition.

#     Args:
#         in_size (list): Lens grid size [H, W].
#         in_dx_m (list): Lens grid discretizion [dy, dx].
#         wavelength_set_m (list): List of wavelengths to be used in focusing profiles.
#         depth_set_m (list): List of frontoplanar depths to be used in focusing profiles.
#         fshift_set_m (list): Nested list of sensor-side focal spot shifts for each lens [[offy, offx],...].
#         out_distance_m (list): Output plane distance from lens.
#         aperture_radius_m (float, optional): Radius of an aperture profile to return. Defaults to None.
#         radial_symmetry (bool, optional): If true, return radial vector describing the profile. Defaults to False.

#     Returns:
#         List: Amplitude, Phase, and Aperture profiles of shape [Z, Lam, H, W].
#     """
#     depth_set_m = np.array(depth_set_m)
#     fshift_set_m = np.array(fshift_set_m)
#     wavelength_set_m = np.array(wavelength_set_m)

#     assert len(fshift_set_m.shape) == 2 and fshift_set_m.shape[-1] == 2
#     assert len(depth_set_m) == len(fshift_set_m)
#     assert len(in_size) == len(in_dx_m) == 2
#     assert isinstance(out_distance_m, float)
#     assert isinstance(aperture_radius_m, float) or aperture_radius_m is None
#     assert isinstance(radial_symmetry, bool)
#     if radial_symmetry:
#         assert in_size[-1] == in_size[-2]
#         assert in_dx_m[-1] == in_dx_m[-2]
#         assert in_size[-1] % 2 != 0
#         assert in_size[-2] % 2 != 0

#     # (L, Z, H, W)
#     x, y = np.meshgrid(np.arange(in_size[-1]), np.arange(in_size[-2]), indexing="xy")
#     x = x - (x.shape[-1] - 1) / 2
#     y = y - (y.shape[-2] - 1) / 2
#     x = x[None, None] * in_dx_m[-1]
#     y = y[None, None] * in_dx_m[-2]

#     wavelength_set = wavelength_set_m[:, None, None, None]
#     depth_set = depth_set_m[None, :, None, None]
#     offset_x = fshift_set_m[:, -1][None, :, None, None]
#     offset_y = fshift_set_m[:, -2][None, :, None, None]

#     phase = (
#         -2
#         * np.pi
#         / wavelength_set
#         * (
#             np.sqrt(depth_set**2 + x**2 + y**2)
#             + np.sqrt(out_distance_m**2 + (x - offset_x) ** 2 + (y - offset_y) ** 2)
#         )
#     )
#     ampl = np.ones_like(phase)
#     phase = np.angle(ampl * np.exp(1j * phase))
#     aperture = (
#         ((np.sqrt(x**2 + y**2) <= aperture_radius_m)).astype(np.float32) + 1e-6
#         if aperture_radius_m is not None
#         else np.ones_like(phase)
#     )

#     ampl = np.transpose(ampl, [1, 0, 2, 3])
#     phase = np.transpose(phase, [1, 0, 2, 3])
#     if radial_symmetry:
#         cidx = in_size[-1] // 2
#         ampl = ampl[:, :, cidx : cidx + 1, cidx:]
#         phase = phase[:, :, cidx : cidx + 1, cidx:]
#         aperture = aperture[:, :, cidx : cidx + 1, cidx:]

#     return ampl, phase, aperture
