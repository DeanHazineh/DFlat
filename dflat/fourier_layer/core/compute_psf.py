import torch
import numpy as np

from .ops_calc_ms_regularizer import regularize_ms_calc_tf
from .ops_transform_util import radial_2d_transform, radial_2d_transform_wrapped_phase
from .method_fresnel_integral import fresnel_diffraction_coeffs, fresnel_diffraction_fft
from .method_angular_spectrum import (
    transfer_function_diffraction,
    transfer_function_Broadband,
)
from .ops_detectorResampling import sensorMeasurement_intensity_phase
from .ops_grid_util import torch_coordinate_grid

#####################################
### Faster (but memory intensive) matrix broadband implementation which is appropriate for ASM only
#####################################


def psf_measured_MatrixASM(
    sim_wavelengths_m,
    point_source_locs,
    ms_modulation_trans,
    ms_modulation_phase,
    parameters,
    normby_transmittance,
):
    """Computes the point-spread function at the sensor-plane then resamples and integrates to yield measurement on a
    user-specified detector pixels. This call directly computes the broadband optical response using large tensor operations
    making the operations faster but memory intensive. This broadband implementation is only valid for ASM propagation.

    Generally, one should set the sensor plane grid to be finer than the detector pixel grid. Area integration is then
    used over the detector pixel area for measured intensity while area averaging is used for measured phase.

    Args:
        `sim_wavelengths_m` (tf.float): Set of wavelengths to compute with.
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (N,3) to compute the PSF for
        `ms_modulation_trans` (tf.float): Metasurface transmittance profile of shape (len(wavelength_set_m) or 1, batch_size, ms_samplesM['y'], ms_samplesM['x'])
            or (len(wavelength_set_m) or 1, batch_size 1, ms_samplesM["r"]).
        `ms_modulation_phase` (tf.float): Metasurface phase profile, the same shape as ms_modulation_trans.
        `parameters` (prop_param):  Settings object defining field propagation details.
        `normby_transmittance` (tf.float): Scalar normalization factor for PSF field.

    Returns:
        `float`: Field intensity measured on the detector, of shape (len(sim_wavelengths_m), batch_size, Nps, sensor_pixel_number["y"], sensor_pixel_number["x"]).
        `float`: Fied phase measured on the detector, of shape (len(sim_wavelengths_m), batch_size, Nps, sensor_pixel_number["y"], sensor_pixel_number["x"]).
    """

    # compute the PSF at the sensor plane -- note that psf_sensor returns
    # torch.abs(field)**2 already with appropriate psf normalization on energy!
    calc_modulation_intensity, calc_modulation_phase = psf_sensor_MatrixASM(
        sim_wavelengths_m,
        point_source_locs,
        ms_modulation_trans,
        ms_modulation_phase,
        parameters,
        normby_transmittance,
    )

    # Predict the measurement on specified detector pixel size and shape
    (
        calc_modulation_intensity,
        calc_modulation_phase,
    ) = sensorMeasurement_intensity_phase(calc_modulation_intensity, calc_modulation_phase, parameters, False)

    return calc_modulation_intensity, calc_modulation_phase


def psf_sensor_MatrixASM(
    sim_wavelengths_m,
    point_source_locs,
    ms_modulation_trans,
    ms_modulation_phase,
    parameters,
    normby_transmittance,
    convert_2D=True,
):
    """Computes the point-spread function on a unifrom grid at the sensor-plane, given a metasurface phase and transmittance.

    This call directly computes the broadband optical response using large tensor operations, making the operations
    faster but memory intensive. This broadband implementation is only valid for ASM propagation.

    Args:
        `sim_wavelengths_m` (tf.float): List/rank1 tensor of wavelengths to simulate for. This is used instead of wavelength_set_m in prop_params
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (Nps,3) to compute the PSF for
        `ms_modulation_trans` (tf.float): Metasurface transmittance of shape (len(sim_wavelength_m) or 1, batch_size, Ny, Nx)
            or (len(sim_wavelength_m) or 1, batch_size, 1, Nr)
        `ms_modulation_phase` (tf.float): Metasurface transmittance of shape (len(sim_wavelength_m) or 1, batch_size, Ny, Nx)
            or (len(sim_wavelength_m) or 1, batch_size, 1, Nr)
        `parameters` (prop_param):  Settings object defining field propagation details.
        `normby_transmittance` (tf.float): Scalar normalization factor for PSF field.
        `convert_2D` (bool, optional): Flag whether or not to convert radially computed PSFs to 2D. Defaults to True.

    Returns:
        `tf.float`: Field intensity at the sensor-plane grid of shape (Nwl, batch_size, Nps, calc_ms_samplesM['y'], calc_ms_samplesM['x'])
        `tf.float`: Field phase at the sensor-plane grid of shape (Nwl, batch_size, Nps, calc_ms_samplesM['y'], calc_ms_samplesM['x']).
    """

    ## For an accurate calculation, resample ms and add the padding as defined during prop_param initialization
    calc_modulation_trans, calc_modulation_phase = regularize_ms_calc_tf(ms_modulation_trans, ms_modulation_phase, parameters)

    ## Get the field after the metasurface, given a point-source spherical wave origin
    (
        calc_modulation_trans,
        calc_modulation_phase,
    ) = wavefront_pointSources_afterms_MatrixASM(
        sim_wavelengths_m,
        point_source_locs,
        calc_modulation_trans,
        calc_modulation_phase,
        parameters,
    )

    ## Propagate the field with the modified, broadband ASM Method
    calc_modulation_trans, calc_modulation_phase = wavefront_afterms_sensor_MatrixASM(
        calc_modulation_trans, calc_modulation_phase, parameters, sim_wavelengths_m
    )

    # After calculation is done, if radial symmetry was used, convert back to 2D unless override return radial
    if parameters["radial_symmetry"] and convert_2D:
        calc_modulation_trans = radial_2d_transform(calc_modulation_trans).squeeze(-3)
        calc_modulation_phase = radial_2d_transform_wrapped_phase(calc_modulation_phase).squeeze(-3)

    ### Normalize by input source energy factor
    calc_modulation_trans /= normby_transmittance
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]

    return (
        torch.abs(calc_modulation_trans) ** 2 * calc_sensor_dx_m["y"] * calc_sensor_dx_m["x"],
        calc_modulation_phase,
    )


def field_propagation_MatrixASM(field_amplitude, field_phase, sim_wavelengths_m, modified_parameters):
    """Takes a batch of field amplitudes and field phases at an input plane (of a single wavelength) and propagates the
    field to an output plane. This routine uses the transfer_function_broadband implementation of field propagation.

    The input to output field distances is defined by parameters["sensor_distance_m"].
    The input grid is defined by parameters["ms_samplesM"] and parameters["ms_dx_m"].
    The output grid is defined via discretization of parameters["sensor_dx_m"] and number points of
    parameters["sensor_pixel_number"]; these variable names are used as the architecture builds/reuses the functions
    initially written for computing psfs.

    Args:
        `field_amplitude` (tf.float): Initial plane field amplitude, of shape
            (len(sim_wavelengths) or 1, batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (len(sim_wavelengths) or 1, batch_size, 1, ms_samplesM["r"])
        `field_phase` (tf.float): Initial plane field phase, the same shape as field_amplitude
        `sim_wavelengths_m` (tf.float): List of simulation wavelengths to propagate the field with
        `modified_parameters` (tf.float): Modified propagation parameters, like generated in Propagate_Planes_Layer_MatrixBroadband

    Returns:
        `tf.float`: Output plane field amplitude, of shape
            ( len(sim_wavelengths), batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"] )
        `tf.float`: Output plane field phase, of shape
            ( len(sim_wavelengths), batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    # For an accurate calculation, resample field and add the padding as defined during prop_param initialization
    calc_field_amplitude, calc_field_phase = regularize_ms_calc_tf(field_amplitude, field_phase, modified_parameters)

    # Propagate the field, using the  with the modified, broadband ASM Method
    distance_m = modified_parameters["sensor_distance_m"]
    calc_samplesN = modified_parameters["calc_samplesN"]
    calc_ms_dx_m = modified_parameters["calc_ms_dx_m"]
    calc_sensor_dx_m = modified_parameters["calc_sensor_dx_m"]
    dtype = modified_parameters["dtype"]
    radial_symmetry = modified_parameters["radial_symmetry"]
    ASM_Pad_opt = modified_parameters["ASM_Pad_opt"]

    # The fields passed in are expected to be of form (batch, Nwl, Ny, Nx) so we need to transpose the field from
    # current shape of (Nwl, batch_size, y, x)
    calc_field_amplitude, calc_field_phase = transfer_function_Broadband(
        calc_field_amplitude.transpose(0, 1).contiguous(),
        calc_field_phase.transpose(0, 1).contiguous(),
        sim_wavelengths_m,
        distance_m,
        calc_ms_dx_m,
        calc_samplesN,
        calc_sensor_dx_m,
        dtype,
        radial_symmetry,
        ASM_Pad_opt,
    )
    calc_field_amplitude = calc_field_amplitude.transpose(0, 1).contiguous()
    calc_field_phase = calc_field_phase.transpose(0, 1).contiguous()

    # Reinterpolate to the user specified grid and also ensure resize
    sensor_pixel_size_m = modified_parameters["sensor_pixel_size_m"]
    (
        calc_field_amplitude,
        calc_field_phase,
    ) = sensorMeasurement_intensity_phase(
        calc_field_amplitude**2 * calc_sensor_dx_m["x"] * calc_sensor_dx_m["y"],
        calc_field_phase,
        modified_parameters,
        radial_symmetry,
    )

    calc_field_amplitude = torch.sqrt(calc_field_amplitude / sensor_pixel_size_m["x"] / sensor_pixel_size_m["y"])

    return calc_field_amplitude, calc_field_phase


def wavefront_pointSources_afterms_MatrixASM(
    sim_wavelengths_m,
    point_sources_locs,
    calc_modulation_trans,
    calc_modulation_phase,
    parameters,
):
    """Computes the set of complex fields after a metasurface, resulting from the illuminated, upsampled/padded phase
    and transmittance modulation profiles. The incident wavefront at the metasurface corresponds to spherical wavefronts
    originating at the point-source locations.

    This function differes from the wavefront_pointSources_afterms as it computes a larger tensor for multiple wavelengths
    and applies it to the broadband input signal.

    Args:
        `sim_wavelengths_m` (tf.float): List/rank1 tensor of wavelengths to simulate for. This is used instead of wavelength_set_m in prop_params
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (Nps,3) to compute the PSF for
        calc_modulation_trans (tf.float): Metasurface transmittance (upsampled/padded) of shape (Nwl or 1, Batch_size, calc_samplesN['y'], calc_samplesN['x'])
            or (Nwl or 1, Batch_size, 1, calc_samplesN["r"]).
        `calc_modulation_phase` (tf.float): Metasurface phase (upsampled/padded) of shape (Nwl or 1, Batch_size, calc_samplesN['y'], calc_samplesN['x'])
            or (Nwl or 1, Batch_size, 1, calc_samplesN["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.

    Returns:
        `tf.float`: Field amplitude after the metasurface, of shape (Nwl, Batch_size, Nps, calc_samplesN['y], calc_samplesN['x'])
            or  (Nwl, Batch_size, Nps, 1, calc_samplesN["r"]).
        `tf.float`: Field phase after the metasurface, of shape (Nwl, Batch_size, Nps, calc_samplesN['y], calc_samplesN['x'])
            or  (Nwl, Batch_size, Nps, 1, calc_samplesN["r"]).
    """

    # trans and phase input has shape Nwl, Nbatch, Ny, Nx
    # Want to return the signal in shape Nwl, Nbatch, Nps, Ny, Nx
    calc_modulation_trans = calc_modulation_trans.unsqueeze(2)
    calc_modulation_phase = calc_modulation_phase.unsqueeze(2)

    angular_wave_number = 2 * np.pi / sim_wavelengths_m
    angular_wave_number = angular_wave_number[:, None, None, None, None]

    # create the metasurface grid
    radial_symmetry = parameters["radial_symmetry"]
    dtype = parameters["dtype"]
    calc_samplesN = parameters["calc_samplesN"]
    calc_ms_dx_m = parameters["calc_ms_dx_m"]

    calc_pixel_x, calc_pixel_y = torch_coordinate_grid(calc_samplesN, calc_ms_dx_m, radial_symmetry, dtype)
    calc_pixel_x = calc_pixel_x[None]
    calc_pixel_y = calc_pixel_y[None]

    # index of the points
    point_source_loc_x = point_sources_locs[:, 0][:, None, None]
    point_source_loc_y = point_sources_locs[:, 1][:, None, None]
    point_source_loc_z = point_sources_locs[:, 2][:, None, None]

    # computation of distance
    torch_zero = parameters["torch_zero"]
    distance_point_ms = torch.sqrt((calc_pixel_x - point_source_loc_x) ** 2 + (calc_pixel_y - point_source_loc_y) ** 2 + point_source_loc_z**2)
    distance_point_ms = distance_point_ms[None, None, :, :, :]

    ## As in wavefront_pointSources_afterms, we remove the 1/r and 1/lambda dependence to aid in normalized psf downstream
    wavefront = torch.complex(calc_modulation_trans, torch_zero) * torch.exp(
        torch.complex(torch_zero, calc_modulation_phase + angular_wave_number * distance_point_ms)
    )

    return torch.abs(wavefront), torch.angle(wavefront)


def wavefront_afterms_sensor_MatrixASM(calc_modulation_trans, calc_modulation_phase, parameters, sim_wavelengths_m):
    ## Propagate the field with the modified, broadband ASM Method
    distance_m = parameters["sensor_distance_m"]
    calc_samplesN = parameters["calc_samplesN"]
    calc_ms_dx_m = parameters["calc_ms_dx_m"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    dtype = parameters["dtype"]
    radial_symmetry = parameters["radial_symmetry"]
    ASM_Pad_opt = parameters["ASM_Pad_opt"]
    init_shape = calc_modulation_trans.shape

    # The fields passed in are expected to be of form (batch, Nwl, Ny, Nx) so we need some reshaping from the
    # prevous shape of (Nwl, Nbatch, Nps, Ny, Nx)
    calc_modulation_trans = torch.permute(calc_modulation_trans, (1, 2, 0, 3, 4)).contiguous().view(-1, init_shape[0], init_shape[3], init_shape[4])
    calc_modulation_phase = torch.permute(calc_modulation_phase, (1, 2, 0, 3, 4)).contiguous().view(-1, init_shape[0], init_shape[3], init_shape[4])
    calc_modulation_trans, calc_modulation_phase = transfer_function_Broadband(
        calc_modulation_trans,
        calc_modulation_phase,
        sim_wavelengths_m,
        distance_m,
        calc_ms_dx_m,
        calc_samplesN,
        calc_sensor_dx_m,
        dtype,
        radial_symmetry,
        ASM_Pad_opt,
    )

    # reshape back to the original input format
    calc_modulation_trans = torch.permute(
        calc_modulation_trans.view(init_shape[1], init_shape[2], init_shape[0], init_shape[3], init_shape[4]),
        [2, 0, 1, 3, 4],
    ).contiguous()
    calc_modulation_phase = torch.permute(
        calc_modulation_phase.view(init_shape[1], init_shape[2], init_shape[0], init_shape[3], init_shape[4]),
        [2, 0, 1, 3, 4],
    ).contiguous()

    return calc_modulation_trans, calc_modulation_phase


#####################################
### Single wavelength implementation
#####################################
def psf_measured(
    point_source_locs,
    ms_modulation_trans,
    ms_modulation_phase,
    parameters,
    normby_transmittance,
):
    """Computes the point-spread function at the sensor-plane then resamples and integrates to yield measurement on a
    user-specified detector pixels.

    Generally, one should set the sensor plane grid to be finer than the detector pixel grid. Area integration is then
    used over the detector pixel area for measured intensity while area averaging is used for measured phase.

    Args:
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (Nps, 3) to compute the PSF for
        `ms_modulation_trans` (tf.float): Metasurface transmittance profile of shape (batch_size, ms_samplesM['y'], ms_samplesM['x'])
            or (batch_size, 1, ms_samplesM["r"]).
        `ms_modulation_phase` (tf.float): Metasurface phase profile of shape (batch_size, ms_samplesM['y'], ms_samplesM['x']) or
            (batch_size, 1, ms_samplesM["r"])
        `parameters` (prop_param):  Settings object defining field propagation details.
        `normby_transmittance` (tf.float): Scalar normalization factor for PSF field.

    Returns:
        `tf.float`: Field intensity measured on the detector, of shape (batch_size, Nps, sensor_pixel_number["y"], sensor_pixel_number["x"]).
        `tf.float`: Fied phase measured on the detector, of shape (batch_size, Nps, sensor_pixel_number["y"], sensor_pixel_number["x"]).
    """

    # compute the PSF at the sensor plane -- note that psf_sensor returns
    # tf.math.abs(field)**2 already with appropriate psf normalization on energy!
    calc_modulation_intensity, calc_modulation_phase = psf_sensor(
        point_source_locs,
        ms_modulation_trans,
        ms_modulation_phase,
        parameters,
        normby_transmittance,
    )

    # Predict the measurement on specified detector pixel size and shape
    (
        calc_modulation_intensity,
        calc_modulation_phase,
    ) = sensorMeasurement_intensity_phase(calc_modulation_intensity, calc_modulation_phase, parameters, False)

    return calc_modulation_intensity, calc_modulation_phase


def psf_sensor(
    point_source_locs,
    ms_modulation_trans,
    ms_modulation_phase,
    parameters,
    normby_transmittance,
    convert_2D=True,
):
    """Computes the point-spread function on a uniform grid at the sensor-plane, given a metasurface phase and transmittance.

    Args:
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (Nps,3) to compute the PSF for
        `ms_modulation_trans` (tf.float): Metasurface transmittance profile of shape (batch_size, ms_samplesM['y'], ms_samplesM['x'])
            or (batch_size, 1, ms_samplesM["r"]).
        `ms_modulation_phase` (tf.float): Metasurface phase profile of shape (batch_size, ms_samplesM['y'], ms_samplesM['x'])
            or (batch_size, 1, ms_samplesM["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.
        `normby_transmittance` (tf.float): Scalar normalization factor for PSF field.
        `convert_2D` (bool, optional): Flag whether or not to convert radially computed PSFs to 2D. Defaults to True.

    Returns:
        `tf.float`: Field intensity at the sensor-plane grid of shape (batch_size, Nps, calc_ms_samplesM['y'], calc_ms_samplesM['x'])
        `tf.float`: Field phase at the sensor-plane grid of shape (batch_size, Nps, calc_ms_samplesM['y'], calc_ms_samplesM['x']).
    """

    # For an accurate calculation, resample ms and add the padding as defined during prop_param initialization
    calc_modulation_trans, calc_modulation_phase = regularize_ms_calc_tf(ms_modulation_trans, ms_modulation_phase, parameters)

    # Get the field after the metasurface, given a point-source spherical wave origin
    calc_modulation_trans, calc_modulation_phase = wavefront_pointSources_afterms(point_source_locs, calc_modulation_trans, calc_modulation_phase, parameters)

    # get finely sampled field just above the sensor
    (calc_modulation_trans, calc_modulation_phase) = wavefront_afterms_sensor(calc_modulation_trans, calc_modulation_phase, parameters)

    # After calculation is done, if radial symmetry was used, convert back to 2D unless override return radial
    if parameters["radial_symmetry"] and convert_2D:
        calc_modulation_trans = radial_2d_transform(calc_modulation_trans.squeeze(2))
        calc_modulation_phase = radial_2d_transform_wrapped_phase(calc_modulation_phase.squeeze(2))

    ### Normalize by input source energy factor
    calc_modulation_trans /= normby_transmittance
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]

    return (
        torch.abs(calc_modulation_trans) ** 2 * calc_sensor_dx_m["y"] * calc_sensor_dx_m["x"],
        calc_modulation_phase,
    )


def wavefront_pointSources_afterms(
    point_sources_locs,
    calc_modulation_trans,
    calc_modulation_phase,
    parameters,
):
    """Computes the set of complex fields after a metasurface, resulting from the illuminated, upsampled/padded phase
    and transmittance modulation profiles. The incident wavefront at the metasurface corresponds to spherical wavefronts
    originating at the point-source locations.

    Args:
        `point_sources_locs` (tf.float): Set of point-source coordinates to compute PSF for, of shape (Nz,3).
        `calc_modulation_trans` (tf.float): Metasurface transmittance (upsampled/padded) of shape (Batch_size, calc_samplesN['y'], calc_samplesN['x'])
            or (Batch_size, 1, calc_samplesN["r"]).
        `calc_modulation_phase` (tf.float): Metasurface phase (upsampled/padded) of shape (Batch_size, calc_samplesN['y'], calc_samplesN['x'])
            or (Batch_size, 1, calc_samplesN["r"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float`: Field amplitude after the metasurface, of shape (Batch_size, Nz, calc_samplesN['y'], calc_samplesN['x']) or  (Batch_size, Nz, 1, calc_samplesN["r"]).
        `tf.float`: Field phase after the metasurface, , of shape (Batch_size, Nz, calc_samplesN['y'], calc_samplesN['x']) or  (Batch_size, Nz, 1, calc_samplesN["r"]).
    """

    # unpack the parameters
    wavelength_m = parameters["wavelength_m"]
    radial_symmetry = parameters["radial_symmetry"]
    dtype = parameters["dtype"]
    calc_samplesN = parameters["calc_samplesN"]
    calc_ms_dx_m = parameters["calc_ms_dx_m"]
    angular_wave_number = 2 * np.pi / wavelength_m
    torch_zero = parameters["torch_zero"]

    # create the metasurface grid
    calc_pixel_x, calc_pixel_y = torch_coordinate_grid(calc_samplesN, calc_ms_dx_m, radial_symmetry, dtype)
    calc_pixel_x = calc_pixel_x.unsqueeze(0)
    calc_pixel_y = calc_pixel_y.unsqueeze(0)

    # index of the points
    point_source_loc_x = point_sources_locs[:, 0][:, None, None]
    point_source_loc_y = point_sources_locs[:, 1][:, None, None]
    point_source_loc_z = point_sources_locs[:, 2][:, None, None]

    # computation of distance
    distance_point_ms = torch.sqrt((calc_pixel_x - point_source_loc_x) ** 2 + (calc_pixel_y - point_source_loc_y) ** 2 + point_source_loc_z**2).unsqueeze(0)

    ## Compute product of spherical wavefront and metasurface
    ## For the spherical wave, one could place 1/(i*lambda*r) instead of 1/r
    ## since intensity and energy in fact requires the extra term as in
    # wavefront = tf.complex(calc_modulation_trans / distance_point_ms, TF_ZERO) * tf.exp(
    #     tf.complex(TF_ZERO, calc_modulation_phase + angular_wave_number * distance_point_ms)
    # )
    ## However we remove the 1/r and 1/lambda dependence to aid in normalized psf downstream
    calc_modulation_trans = torch.unsqueeze(calc_modulation_trans, 1)
    calc_modulation_phase = torch.unsqueeze(calc_modulation_phase, 1)
    wavefront = torch.complex(calc_modulation_trans, torch_zero) * torch.exp(
        torch.complex(torch_zero, calc_modulation_phase + angular_wave_number * distance_point_ms)
    )

    return torch.abs(wavefront), torch.angle(wavefront)


def wavefront_afterms_sensor(
    calc_modulation_trans,
    calc_modulation_phase,
    parameters,
):
    """Propagate the complex field from after the ms to just above the sensor plane

    Args:
        `calc_modulation_trans` (tf.float): Field amplitude just after the metasurface (upsampled/padded), of shape
            (batch_size, Nps, calc_samplesN['y'], calc_samplesN['x']) or (batch_size, Nps, 1, calc_samplesN["r"]).
        `calc_modulation_phase` (tf.float): Field phase just after the metasurface (upsampled/padded), of shape
            (batch_size, Nps, calc_samplesN['y'], calc_samplesN['x']) or (batch_size, Nps, 1, calc_samplesN["r"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float`: Field amplitude at the sensor plane grid, of shape (batch_size, Nps, calc_samplesN['y'], calc_samplesN['x']) or (batch_size, Nps, 1, calc_samplesN["r"]).
        `tf.float`: Field phase at the sensor plane grid, of shape (batch_size, Nps, calc_samplesN['y'], calc_samplesN['x']) or (batch_size, Nps, 1, calc_samplesN["r"]).
    """

    # propagate the field using a specified engine
    wavelength_m = parameters["wavelength_m"]
    sensor_distance_m = parameters["sensor_distance_m"]
    calc_samplesN = parameters["calc_samplesN"]
    calc_ms_dx_m = parameters["calc_ms_dx_m"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    dtype = parameters["dtype"]
    radial_symmetry = parameters["radial_symmetry"]
    diffractionEngine = parameters["diffractionEngine"]

    if diffractionEngine == "fresnel_fourier":
        propagator = fresnel_diffraction_fft
    elif diffractionEngine == "ASM_fourier":
        propagator = transfer_function_diffraction

    # Note when we pass the profiles to the propagator, we regroup the batch_size and Nps dimensions to be one large batch
    # size and we can reshape it after the calculation back to the user input. In terms of propagations, both input
    # dimensions are to the same effect
    init_shape = calc_modulation_trans.shape
    wavefront_trans, wavefront_phase = propagator(
        calc_modulation_trans.view(init_shape[0] * init_shape[1], init_shape[2], init_shape[3]),
        calc_modulation_phase.view(init_shape[0] * init_shape[1], init_shape[2], init_shape[3]),
        wavelength_m,
        sensor_distance_m,
        calc_ms_dx_m,
        calc_samplesN,
        calc_sensor_dx_m,
        dtype,
        radial_symmetry,
        parameters["ASM_Pad_opt"],
    )

    # When the fresnel transform calculation is done, coefficients need to be added back in
    # this is done here rather than in the propagator call so that all propagator engines have same
    # inputs to function. No Coefficients are missing in the transfer_function_diffraction propagator
    if diffractionEngine == "fresnel_fourier":
        calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
        wavefront_trans, wavefront_phase = fresnel_diffraction_coeffs(
            wavefront_trans,
            wavefront_phase,
            wavelength_m,
            sensor_distance_m,
            calc_sensor_dx_m,
            calc_samplesN,
            dtype,
            radial_symmetry,
        )

    # new_shape = wavefront_trans.shape
    wavefront_trans = wavefront_trans.view(*init_shape)
    wavefront_phase = wavefront_phase.view(*init_shape)

    return wavefront_trans, wavefront_phase


def field_propagation(field_amplitude, field_phase, parameters):
    """Takes a batch of field amplitudes and field phases at an input plane (of a single wavelength) and propagates the
    field to an output plane.

    The input to output field distances is defined by parameters["sensor_distance_m"].
    The input grid is defined by parameters["ms_samplesM"] and parameters["ms_dx_m"].
    The output grid is defined via discretization of parameters["sensor_dx_m"] and number points of
    parameters["sensor_pixel_number"]; these variable names are used as the architecture builds/reuses the functions
    initially written for computing psfs.

    Args:
        `field_amplitude` (tf.float): Initial plane field amplitude, of shape
            (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])
        `field_phase` (tf.float): Initial plane field phase, of shape (batch_size, ms_samplesM["y"], ms_samplesM["x"])
            or (batch_size, 1, ms_samplesM["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.

    Returns:
        `tf.float`: Output plane field amplitude, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float`: Output plane field phase, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    # For an accurate calculation, resample field and add the padding as defined during prop_param initialization
    field_amplitude, field_phase = regularize_ms_calc_tf(field_amplitude, field_phase, parameters)

    # Propagate the field, piggy-back off the psf derived functions
    field_amplitude, field_phase = wavefront_afterms_sensor(field_amplitude.unsqueeze(1), field_phase.unsqueeze(1), parameters)
    field_amplitude = field_amplitude.squeeze(1)
    field_phase = field_phase.squeeze(1)

    # Reinterpolate to the user specified grid and also ensure resize
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    radial_symmetry = parameters["radial_symmetry"]
    field_amplitude, field_phase = sensorMeasurement_intensity_phase(
        field_amplitude**2 * calc_sensor_dx_m["x"] * calc_sensor_dx_m["y"],
        field_phase,
        parameters,
        radial_symmetry,
    )
    field_amplitude = torch.sqrt(field_amplitude / sensor_pixel_size_m["x"] / sensor_pixel_size_m["y"])

    return field_amplitude, field_phase
