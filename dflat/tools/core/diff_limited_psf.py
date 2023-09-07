import numpy as np
from scipy.special import j1
from dflat.fourier_layer.core.ops_hankel import radial_2d_transform
from dflat.plot_utilities.coordinate_vector import get_detector_pixel_coordinates


def airy_disk(propagation_parameters, normalize=True):
    wavelength_set_m = propagation_parameters["wavelength_set_m"]
    radius_m = propagation_parameters["radius_m"]
    sensor_distance_m = propagation_parameters["sensor_distance_m"]
    sensor_pixel_number = propagation_parameters["sensor_pixel_number"]
    rpix = int((sensor_pixel_number["x"] - 1) / 2)

    x, _ = get_detector_pixel_coordinates(propagation_parameters)
    x = x[rpix:][None, :]
    x = 2 * np.pi / wavelength_set_m[:, None] * radius_m * np.sin(x / np.sqrt(sensor_distance_m**2 + x**2)) + 1e-6
    airy_profile = (2 * j1(x) / x) ** 2
    airy_profile = radial_2d_transform(airy_profile)

    if normalize:
        airy_profile = airy_profile / np.sum(airy_profile, axis=(-1, -2), keepdims=True)

    return airy_profile
