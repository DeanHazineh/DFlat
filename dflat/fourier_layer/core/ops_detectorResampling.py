import torch
import torch.nn.functional as F
import numpy as np

from .ops_transform_util import resize_with_crop_or_pad


def resize_area(image, resize_ratio):
    """Autodifferentiable area resize implementation based on a strided convolution using F.conv2d.

    Currently, this implementation is only valid for integer resize, not fractional.

    Args:
        `image` (torch.Tensor): Image to be resampled and box averaged, of shape (batch_size, channel_size, Ny, Nx)
        `Resize_ratio` (tuple): Resize factor along y and x, defined by (scaley, scalex)

    Returns:
        `torch.Tensor`: Area-resized image of shape (batch_size, channel_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"])
    """
    # Define the filter which requires shape [out_channels, in_channels, filter_height, filter_width]
    # Use a simple box filter then do a 2D strided convolution with image
    rectFilter = torch.ones((1, 1, resize_ratio[0], resize_ratio[1]), dtype=image.dtype)
    outimage = F.conv2d(image, rectFilter, stride=resize_ratio, padding=0)

    return outimage


def resample_intensity_sensor(intensity, resize_ratio):
    """Helper function calling the desired resize_area method to process the passed in intensity.

    Currently, only an autodifferentiable, non-fractional box-filter area-resize method is encoded and used.
    In the future, this function may be expanded to take other options.

    Args:
        `Intensity` (tf.float64): Field intensity to be resampled and box integrated, of shape (batch_size, channel_size, Ny, Nx)
        `Resize_ratio` (tuple): Resize factor along y and x, defined by (scaley, scalex)

    Returns:
        `tf.float`: Area-resized intensity of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size).
    """
    return resize_area(intensity, resize_ratio)


def resample_phase_sensor(phase, resize_ratio):
    """Helper function calling the desired resize_area average method to process the passed in phase profile.

    Args:
        `Phase` (tf.float64): Field phase profile to be resampled and box integrated, of shape (batch_size, channel_size, Ny, Nx)
        `Resize_ratio` (tuple): Resize factor along y and x, defined by (scaley, scalex)

    Returns:
        `tf.float`: Area-resized phase of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size).
    """
    phasex = resize_area(torch.cos(phase), resize_ratio) / resize_ratio[0] / resize_ratio[1]
    phasey = resize_area(torch.sin(phase), resize_ratio) / resize_ratio[0] / resize_ratio[1]
    return torch.atan2(phasey, phasex)


def sensorMeasurement_intensity_phase(sensor_intensity, sensor_phase, parameters, use_radial):
    """Returns both the measured intensity on the detector and the averaged phase on the detector pixels, given the
    intensity and phase on a grid just above the detector plane.

    Note: In the current version, the measurement implementation does NOT enable fractional resize treatment. As an
    important consquence, the integer rounded resize will be slightly incorrect.

    Args:
        `sensor_intensity` (tf.float64): Field intensity at the sensor plane, of shape (..., calc_samplesN["y"], calc_samplesN["x"]).
        `sensor_phase` (tf.float64): Field phase at the sensor plane, of shape (..., calc_samplesN["y"], calc_samplesN["x"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Intensity measured on the detector pixel array, of shape (..., sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float64`: Average phase measured on the detector pixel array, of shape (..., sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    if not (sensor_intensity.shape == sensor_phase.shape):
        raise ValueError("intensity and phase must be the same shape")

    ## Handle multi-dimension input
    input_rank = sensor_intensity.dim()
    init_shape = sensor_intensity.size()
    if input_rank == 1:
        raise ValueError("Input tensors must have a rank \geq 2")
    elif input_rank == 2:
        sensor_intensity = sensor_intensity.unsqueeze(0)
        sensor_phase = sensor_phase.unsqueeze(0)
    elif input_rank > 3:
        sensor_intensity = sensor_intensity.view(-1, init_shape[-2], init_shape[-1])
        sensor_phase = sensor_phase.view(-1, init_shape[-2], init_shape[-1])

    ## Unpack parameters
    # If we are doing PSFs, then we would have passed use_radial = False because we always want 2D PSFs
    # IF we are doing field propagation, we might instead want to keep vectors as radial
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    sensor_pixel_number = parameters["sensor_pixel_number"]
    radial_flag = use_radial

    ### Only do convolutional downdampling if the grid size and the sensor sizes are not the same (up to a tolerance)
    tol = 0.2e-6
    diff_x = abs(sensor_pixel_size_m["x"] - calc_sensor_dx_m["x"])
    diff_y = abs(sensor_pixel_size_m["y"] - calc_sensor_dx_m["y"])
    if diff_x > tol or diff_y > tol:
        # Because we will do convolutional downsampling, we first want to take the field and reinterpolate it onto a new grid
        # so that the downsample mount is close to an integer instead of a float
        scalex = sensor_pixel_size_m["x"] / calc_sensor_dx_m["x"]
        scaley = sensor_pixel_size_m["y"] / calc_sensor_dx_m["y"]
        upsample_size = (
            int(np.round(scaley) / scaley * init_shape[-2]),
            int(np.round(scalex) / scalex * init_shape[-1]),
        )
        resize_ratio = (int(np.round(scaley)), int(np.round(scalex)))

        # Increase the tensor dimensionality to the form of [Batch x channels x Height x Width]
        sensor_intensity = sensor_intensity.unsqueeze(1)
        sensor_phase = sensor_phase.unsqueeze(1)

        sensor_phase_real = F.interpolate(torch.cos(sensor_phase), size=upsample_size, mode="bicubic")
        sensor_phase_imag = F.interpolate(torch.sin(sensor_phase), size=upsample_size, mode="bicubic")
        sensor_intensity = F.interpolate(sensor_intensity, size=upsample_size, mode="bicubic")
        sensor_phase = torch.atan2(sensor_phase_imag, sensor_phase_real)

        # Now call the area resize methods for intensity and phase.
        sensor_intensity = resample_intensity_sensor(sensor_intensity, resize_ratio).squeeze(1)
        sensor_phase = resample_phase_sensor(sensor_phase, resize_ratio).squeeze(1)

    # Crop or pad with zeros to the sensor size
    sensor_intensity = resize_with_crop_or_pad(
        sensor_intensity,
        1 if radial_flag else sensor_pixel_number["y"],
        sensor_pixel_number["r"] if radial_flag else sensor_pixel_number["x"],
        radial_flag,
    )
    sensor_phase = resize_with_crop_or_pad(
        sensor_phase,
        1 if radial_flag else sensor_pixel_number["y"],
        sensor_pixel_number["r"] if radial_flag else sensor_pixel_number["x"],
        radial_flag,
    )

    # Return with the same batch_size shape
    new_shape = sensor_intensity.shape
    if input_rank != 3:
        sensor_intensity = sensor_intensity.reshape(*init_shape[:-2], *new_shape[-2:])
        sensor_phase = sensor_phase.reshape(*init_shape[:-2], *new_shape[-2:])

    return sensor_intensity, sensor_phase
