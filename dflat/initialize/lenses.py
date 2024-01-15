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
    depth_set_m = np.array(depth_set_m)
    fshift_set_m = np.array(fshift_set_m)
    wavelength_set_m = np.array(wavelength_set_m)

    assert len(fshift_set_m.shape) == 2 and fshift_set_m.shape[-1] == 2
    assert len(depth_set_m) == len(fshift_set_m)
    assert len(in_size) == len(in_dx_m) == 2
    assert isinstance(out_distance_m, float)
    assert isinstance(aperture_radius_m, float)
    assert isinstance(radial_symmetry, bool)
    if radial_symmetry:
        assert in_size[-1] == in_size[-2]
        assert in_dx_m[-1] == in_dx_m[-2]
        assert in_size[-1] % 2 != 0
        assert in_size[-2] % 2 != 0

    # (L, Z, H, W)
    x, y = np.meshgrid(np.arange(in_size[-1]), np.arange(in_size[-2]), indexing="xy")
    x = x - (x.shape[-1] - 1) / 2
    y = y - (y.shape[-2] - 1) / 2
    x = x[None, None] * in_dx_m[-1]
    y = y[None, None] * in_dx_m[-2]

    wavelength_set = wavelength_set_m[:, None, None, None]
    depth_set = depth_set_m[None, :, None, None]
    offset_x = fshift_set_m[:, -1][None, :, None, None]
    offset_y = fshift_set_m[:, -2][None, :, None, None]

    phase = (
        -2
        * np.pi
        / wavelength_set
        * (
            np.sqrt(depth_set**2 + x**2 + y**2)
            + np.sqrt(out_distance_m**2 + (x - offset_x) ** 2 + (y - offset_y) ** 2)
        )
    )
    ampl = np.ones_like(phase)
    phase = np.angle(ampl * np.exp(1j * phase))
    aperture = (
        ((np.sqrt(x**2 + y**2) <= aperture_radius_m)).astype(np.float32) + 1e-6
        if aperture_radius_m is not None
        else np.ones_like(phase)
    )

    # reshape the output
    ampl = np.transpose(ampl, [1, 0, 2, 3])
    phase = np.transpose(phase, [1, 0, 2, 3])
    if radial_symmetry:
        cidx = in_size[-1] // 2
        ampl = ampl[:, :, cidx : cidx + 1, cidx:]
        phase = phase[:, :, cidx : cidx + 1, cidx:]
        aperture = aperture[:, :, cidx : cidx + 1, cidx:]

    return ampl, phase, aperture
