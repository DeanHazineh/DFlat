import torch
from .compute_psf import (
    psf_measured,
    psf_measured_MatrixASM,
    field_propagation,
    field_propagation_MatrixASM,
)


#################################
def loopWavelength_psf_measured(ms_trans, ms_phase, normby, point_source_locs, parameters_list):
    """Loops the PSF measured routine over wavelength. If one wants to compute over wavelengths without batching, the X
    caller should be used instead which implements a matrix broadband calculation using the ASM engine.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and need not be trivially redefined. Natural broadcasting will be used. The output
    structure is the same but this function is overloaded on input.

    Args:
        `ms_trans` (tf.float): Metasurface transmittance profiles, of shape
            (len(wavelength_set_m), num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), num_profiles, 1, ms_samplesM["r"]). Alternatively, the transmittance of the optical
            element may be assumed wavelength-independent and given rank 3 tensor of shape
            (num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or
            (num_profiles, 1, ms_samplesM["r"]).
        `ms_phase` (tf.float):  Metasurface phase profiles on each wavelength channel. The shape is the same as ms_trans input.
        `normby` (tf.float): Scalar tensor corresponding to a normalization factor to dive the computed psf by,
            (ideally to make the total psf energy unity).
        `point_source_locs` (tf.float): Tensor of point-source coordinates, of shape (N,3)
        `parameters_list` (list): List of prop_param objects, each being initialized (in order of wavelength_set_m) for
            a different wavelength_m value.

    Returns:
        `tf.float`: Batched PSF intensity of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float`: Batched PSF phase of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    # Unpack Parameters
    num_wavelengths = len(parameters_list)
    input_rank = len(ms_trans.shape)
    holdPSF_int_list = []
    holdPSF_phase_list = []

    if input_rank == 3:
        # the metasurface modulation profiles are the same across wavelength
        for idx in range(num_wavelengths):
            psfs_int, psfs_phase = psf_measured(point_source_locs, ms_trans, ms_phase, parameters_list[idx], normby)
            holdPSF_int_list.append(psfs_int.unsqueeze(0))
            holdPSF_phase_list.append(psfs_phase.unsqueeze(0))
    elif input_rank == 4:
        # The metasurface modulations are defined for each wavelength channel
        for idx in range(num_wavelengths):
            psfs_int, psfs_phase = psf_measured(
                point_source_locs,
                ms_trans[idx],
                ms_phase[idx],
                parameters_list[idx],
                normby,
            )
            holdPSF_int_list.append(psfs_int.unsqueeze(0))
            holdPSF_phase_list.append(psfs_phase.unsqueeze(0))
    else:
        raise ValueError("Rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor.")

    return torch.cat(holdPSF_int_list, dim=0), torch.cat(holdPSF_phase_list, dim=0)


def batch_loopWavelength_psf_measured(ms_trans, ms_phase, normby, point_source_locs, parameters_list):
    """Loops the PSF measured routine over wavelength and also adds a batch loop over the metasurface profiles dimension.
    To not explicitly batch over metasurfaces, use loopWavelength_psf_measured instead. See there for more details.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and need not be trivially redefined. Natural broadcasting will be used. The output
    structure is the same but this function is overloaded on input.

    Args:
        `ms_trans` (tf.float): Metasurface transmittance profiles, of shape
            (len(wavelength_set_m), num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), num_profiles, 1, ms_samplesM["r"]). Alternatively, the transmittance of the optical
            element may be assumed wavelength-independent and given rank 3 tensor of shape
            (num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or
            (num_profiles, 1, ms_samplesM["r"]).

        `ms_phase` (tf.float):  Metasurface phase profiles on each wavelength channel. The shape is the same as ms_trans input.

        `normby` (tf.float): Scalar tensor corresponding to a normalization factor to dive the computed psf by,
            (ideally to make the total psf energy unity).

        `point_source_locs` (tf.float): Tensor of point-source coordinates, of shape (N,3)

        `parameters_list` (list): List of prop_param objects, each being initialized (in order of wavelength_set_m) for
            a different wavelength_m value.

    Returns:
        `tf.float`: Batched PSF intensity of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float`: Batched PSF phase of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    # unpack parameters
    input_rank = ms_trans.dim()
    holdPSF_int_list = []
    holdPSF_phase_list = []

    # Check input tensor dimensions
    if input_rank == 3:
        num_ms = ms_trans.shape[0]
        for idx in range(num_ms):
            psfs_int, psfs_phase = loopWavelength_psf_measured(
                ms_trans[idx : idx + 1],
                ms_phase[idx : idx + 1],
                normby,
                point_source_locs,
                parameters_list,
            )
            holdPSF_int_list.append(psfs_int)
            holdPSF_phase_list.append(psfs_phase)
    elif input_rank == 4:
        num_ms = ms_trans.shape[1]
        for idx in range(num_ms):
            psfs_int, psfs_phase = loopWavelength_psf_measured(
                ms_trans[:, idx : idx + 1, :, :],
                ms_phase[:, idx : idx + 1, :, :],
                normby,
                point_source_locs,
                parameters_list,
            )
            holdPSF_int_list.append(psfs_int)
            holdPSF_phase_list.append(psfs_phase)
    else:
        raise ValueError("Rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor.")

    return torch.cat(holdPSF_int_list, dim=1), torch.cat(holdPSF_phase_list, dim=1)


def batch_psf_measured_MatrixASM(
    sim_wavelengths_m,
    point_source_locs,
    ms_trans,
    ms_phase,
    modified_parameters,
    normby,
):
    """Loops the psf_measured_Matrix_ASM routine by for-loop batching over the ms_batch dimension.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and need not be trivially redefined. Natural broadcasting will be used. The output
    structure is the same but this function is overloaded on input.

    Args:
        `sim_wavelengths_m` (tf.float): Tensor of simulation wavelengths to compute the psf for.
        `point_source_locs` (tf.float): Tensor of shape (Nps, 3) corresponding to point-source locations to compute psf for
        `ms_trans` (tf.float): Transmittance profile(s) of the optical metasurface(s), of shape
            (len(wavelength_set_m) or 1, profile_batch, ms_samplesM['y'], ms_samplesM['x']),
            or (len(wavelength_set_m) or 1, profile_batch, 1, ms_samplesM['r']). Rank 3 allowed instead of wl=1.
        `ms_phase` (tf.float): Phase profiles of the metasurface, same shape as ms_trans.
        `modified_parameters` (prop_params): Modified prop_params as generated in PSF_Layer_MatrixBroadband
        `normby` (tf.float): Scalar tensor corresponding to a normalization factor to dive the computed psf by,
            (ideally to make the total psf energy unity).

    Returns:
        `tf.float`: PSF intensity of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float`: PSF phase of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    # unpack parameters and handle overloading
    input_rank = len(ms_trans.shape)
    if input_rank == 3:
        ms_trans = ms_trans[None]
        ms_phase = ms_phase[None]
    elif input_rank != 4:
        raise ValueError("Rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor.")

    # Placeholder lists for the results
    holdPSF_int_list = []
    holdPSF_phase_list = []

    num_ms = ms_trans.shape[1]
    for idx in range(num_ms):
        psfs_int, psfs_phase = psf_measured_MatrixASM(
            sim_wavelengths_m,
            point_source_locs,
            ms_trans.unsqueeze[:, idx : idx + 1, :, :],
            ms_phase.unsqueeze[:, idx : idx + 1, :, :],
            modified_parameters,
            normby,
        )

        holdPSF_int_list.append(psfs_int)
        holdPSF_phase_list.append(psfs_phase)

    return torch.cat(holdPSF_int_list, dim=1), torch.cat(holdPSF_phase_list, dim=1)


### Propagation Layer Routines
def loopWavelength_field_propagation(field_amplitude, field_phase, parameters_list):
    """Loops the field propagation routine over wavelength. If one wants to compute over wavelength without batching, the
    X caller should be used instead which implements a matrix broadband calculation using the ASM engine.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and are thus need not be trivially redefined in the input. The PSF in either case
    is still wavelength-dependent (diffraction propagation is inherrently wavelength-dependent).
    The output structure is the same but this function is overloaded on input.

    Args:
        `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field amplitude
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (profile_batch, 1, ms_samplesM["r"]).

        `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field phase
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (batch_size, 1, ms_samplesM["r"]).

        `parameters_list` (list): List of prop_param objects, each being initialized (in order of wavelength_set_m) for
            a different wavelength_m value.

    Returns:
        `tf.float`: Field amplitude of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
        `tf.float`: Field phase of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"])."""

    # unpack parameters
    num_wavelengths = len(parameters_list)
    input_rank = len(field_amplitude.shape)

    # Placeholder lists for the results
    holdField_ampl = []
    holdField_phase = []
    if input_rank == 3:
        # the metasurface modulation profiles are the same across wavelength
        for idx in range(num_wavelengths):
            ampl, phase = field_propagation(field_amplitude, field_phase, parameters_list[idx])
            holdField_ampl.append(ampl.unsqueeze(0))
            holdField_phase.append(phase.unsqueeze(0))
    elif input_rank == 4:
        # The metasurface modulations are defined for each wavelength channel
        for idx in range(num_wavelengths):
            ampl, phase = field_propagation(field_amplitude[idx], field_phase[idx], parameters_list[idx])
            holdField_ampl.append(ampl.unsqueeze(0))
            holdField_phase.append(phase.unsqueeze(0))
    else:
        raise ValueError("rank of input profiles are incorrect. must be rank 3 or rank 4 tensor.")

    return torch.cat(holdField_ampl, dim=0), torch.cat(holdField_phase, dim=0)


def batch_loopWavelength_field_propagation(field_amplitude, field_phase, parameters_list):
    """Loops the field propagation routine over wavelength and also adds a batch loop over the field input profiles dimension.
    To not explicitly batch over the stack of input fields, use loopWavelength_field_propagation instead. See there for more details.

    When the profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and are thus need not be trivially redefined in the input. The PSF in either case
    is still wavelength-dependent (diffraction propagation is inherrently wavelength-dependent).
    The output structure is the same but this function is overloaded on input.

    Args:
        `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field amplitude
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (profile_batch, 1, ms_samplesM["r"]).

        `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field phase
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (batch_size, 1, ms_samplesM["r"]).

        `parameters_list` (list): List of prop_param objects, each being initialized (in order of wavelength_set_m) for
            a different wavelength_m value.

    Returns:
        `tf.float`: Field amplitude of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
        `tf.float`: Field phase of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
    """

    # unpack parameters
    input_rank = field_amplitude.dim()
    holdField_ampl = []
    holdField_phase = []

    if input_rank == 3:
        num_prof = field_amplitude.shape[0]
        for idx in range(num_prof):
            ampl, phase = loopWavelength_field_propagation(
                field_amplitude[idx : idx + 1],
                field_phase[idx : idx + 1],
                parameters_list,
            )
            holdField_ampl.append(ampl)
            holdField_phase.append(phase)
    elif input_rank == 4:
        num_prof = field_amplitude.shape[1]
        for idx in range(num_prof):
            ampl, phase = loopWavelength_field_propagation(
                field_amplitude[:, idx : idx + 1, :, :],
                field_phase[:, idx : idx + 1, :, :],
                parameters_list,
            )
            holdField_ampl.append(ampl)
            holdField_phase.append(phase)
    else:
        raise ValueError("rank of input profiles are incorrect. must be rank 3 or rank 4 tensor.")

    return torch.cat(holdField_ampl, dim=1), torch.cat(holdField_phase, dim=1)


def batch_field_propagation_MatrixASM(field_amplitude, field_phase, sim_wavelengths_m, modified_parameters):
    """Helper function to batch the field propagation via Matrix ASM method over the profile batch dimension. Wavelength
    calculations are done efficiently without loops. If one does not want to batch profiles, just call the field propagation
    method alone.

    When the input profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and need not be trivially redefined. Natural broadcasting will be used. The output
    structure is the same but this function is overloaded on input.

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

    # Handle Overloading
    input_rank = len(field_amplitude.shape)
    if input_rank == 3:
        field_amplitude = field_amplitude[None]
        field_phase = field_phase[None]
    elif input_rank != 4:
        raise ValueError("Rank of inputs are incorrect. must be rank 3 or rank 4 tensor.")

    holdField_ampl = []
    holdField_phase = []
    num_prof = field_amplitude.shape[1]

    for idx in range(num_prof):
        ampl, phase = field_propagation_MatrixASM(
            field_amplitude[:, idx : idx + 1, :, :],
            field_phase[:, idx : idx + 1, :, :],
            sim_wavelengths_m,
            modified_parameters,
        )
        holdField_ampl.append(ampl)
        holdField_phase.append(phase)

    return torch.cat(holdField_ampl, dim=1), torch.cat(holdField_phase, dim=1)
