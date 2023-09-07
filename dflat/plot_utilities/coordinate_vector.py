import numpy as np


def get_detector_pixel_coordinates(propagation_parameters):
    sensor_pixel_size_m = propagation_parameters["sensor_pixel_size_m"]
    sensor_pixel_number = propagation_parameters["sensor_pixel_number"]

    x = (np.arange(0, sensor_pixel_number["x"], 1) - (sensor_pixel_number["x"] - 1) / 2) * sensor_pixel_size_m["x"]
    y = (np.arange(0, sensor_pixel_number["y"], 1) - (sensor_pixel_number["y"] - 1) / 2) * sensor_pixel_size_m["y"]
    return x, y


def get_lens_pixel_coordinates(propagation_parameters):
    ms_dx_m = propagation_parameters["ms_dx_m"]
    ms_samplesM = propagation_parameters["ms_samplesM"]
    radial_symmetry = propagation_parameters["radial_symmetry"]

    if radial_symmetry:
        r = np.arange(0, ms_samplesM["r"], 1) * ms_dx_m["x"]
        return r, np.array([0])
    else:
        x = (np.arange(0, ms_samplesM["x"], 1) - (ms_samplesM["x"] - 1) / 2) * ms_dx_m["x"]
        y = (np.arange(0, ms_samplesM["y"], 1) - (ms_samplesM["y"] - 1) / 2) * ms_dx_m["y"]
        return x, y
