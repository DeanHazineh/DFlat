import torch
import torch.nn.functional as F


def condResizeFn_true(ms_modulation_trans, ms_modulation_phase, parameters):
    # method = "nearest"  # Do NOT change; This is a meaningful choice over interp methods!
    # Nearest matches openCV's implementation but torch recommends using nearest-exact
    method = "nearest-exact"
    ms_modulation_trans = ms_modulation_trans.unsqueeze(0)  # Make room for the channels dimensions and a 4D input
    ms_modulation_phase = ms_modulation_phase.unsqueeze(0)

    # handle radial flag conditional
    calc_samplesM = parameters["calc_samplesM"]
    radial_flag = parameters["radial_symmetry"]
    resizeTo = [1, calc_samplesM["r"]] if radial_flag else [calc_samplesM["y"], calc_samplesM["x"]]

    # Resize transmittance of the field -- Just upsampling so nearest interp required.
    calc_modulation_trans = F.interpolate(ms_modulation_trans, size=resizeTo, mode=method)

    # Resize phase of the field -- Just upsampling so nearest interp required.
    calc_modulation_phase_real = F.interpolate(torch.cos(ms_modulation_phase), size=resizeTo, mode=method)
    calc_modulation_phase_imag = F.interpolate(torch.sin(ms_modulation_phase), size=resizeTo, mode=method)
    calc_modulation_phase = torch.atan2(calc_modulation_phase_imag, calc_modulation_phase_real)

    return calc_modulation_trans.squeeze(0), calc_modulation_phase.squeeze(0)


def condPad_true(calc_modulation_trans, calc_modulation_phase, parameters):
    # Get paddings from parameters
    padms_half = parameters["padms_half"]
    padhalfx = padms_half["x"]
    padhalfy = padms_half["y"]

    # handle radial flag conditional
    radial_flag = parameters["radial_symmetry"]
    paddings = [0, padhalfx, 0, 0, 0, 0] if radial_flag else [padhalfx, padhalfx, padhalfy, padhalfy, 0, 0]

    calc_modulation_trans = F.pad(calc_modulation_trans, paddings, mode="constant", value=0)
    calc_modulation_phase = F.pad(calc_modulation_phase, paddings, mode="constant", value=0)
    return calc_modulation_trans, calc_modulation_phase


def regularize_ms_calc_tf(
    ms_modulation_trans,
    ms_modulation_phase,
    parameters,
):
    """Given an input amplitude and phase profile defined on the grid specified in the parameters object, upsample the
    field and pad according to the computed dimensions in prop_params object.

    Args:
        `ms_modulation_trans` (tf.float64): Metasurface transmittance on the user specified grid of shape
            (..., ms_samplesM['y'], ms_samplesM['x']) or (..., 1, ms_samplesM['r']).
        `ms_modulation_phase` (tf.float64): Metasurface phase on the user specified grid of shape
            (..., ms_samplesM['y'], ms_samplesM['x']) or (..., 1, ms_samplesM['r']).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Upsampled and padded metasurface transmittance of shape (..., calc_samplesN['y'], calc_samplesN['x'])
            or (..., 1, calc_samplesN['r']).
        `tf.float64`: Upsampled and padded metasurface phase of shape (..., calc_samplesN['y'], calc_samplesN['x'])
            or (..., 1, calc_samplesN['r'])
    """

    if not (ms_modulation_trans.shape == ms_modulation_phase.shape):
        raise ValueError("transmittance and phase must be the same shape")

    ### Handle multi-dimension input for different downstream use cases
    input_rank = ms_modulation_phase.dim()
    init_shape = ms_modulation_phase.size()
    if input_rank == 1:
        raise ValueError("amplitude and phase input tensors must have a rank \geq 2")
    elif input_rank == 2:
        ms_modulation_phase = ms_modulation_phase.unsqueeze(0)
        ms_modulation_trans = ms_modulation_trans.unsqueeze(0)
    elif input_rank > 3:
        ms_modulation_phase = ms_modulation_phase.view(-1, init_shape[-2], init_shape[-1])
        ms_modulation_trans = ms_modulation_trans.view(-1, init_shape[-2], init_shape[-1])

    ### unpack parameters
    dtype = parameters["dtype"]
    calc_samplesM = parameters["calc_samplesM"]
    ms_samplesM = parameters["ms_samplesM"]

    ### Resample the metasurface via nearest neighbors if required
    if (calc_samplesM["x"] > ms_samplesM["x"]) or (calc_samplesM["y"] > ms_samplesM["y"]):
        calc_modulation_trans, calc_modulation_phase = condResizeFn_true(ms_modulation_trans, ms_modulation_phase, parameters)
    else:
        calc_modulation_trans, calc_modulation_phase = (
            ms_modulation_trans,
            ms_modulation_phase,
        )

    ### Pad the array if samplesN (padded) is larger than samplesM (unpadded)
    calc_samplesN = parameters["calc_samplesN"]
    if (calc_samplesN["x"] > calc_samplesM["x"]) or (calc_samplesN["y"] > calc_samplesM["y"]):
        calc_modulation_trans, calc_modulation_phase = condPad_true(calc_modulation_trans, calc_modulation_phase, parameters)

    # Return with the same batch_size shape
    new_shape = calc_modulation_trans.shape
    if input_rank != 3:
        calc_modulation_trans = calc_modulation_trans.view(*init_shape[:-2], *new_shape[-2:])
        calc_modulation_phase = calc_modulation_phase.view(*init_shape[:-2], *new_shape[-2:])

    return calc_modulation_trans.type(dtype), calc_modulation_phase.type(dtype)
