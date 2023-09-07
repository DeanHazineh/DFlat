import torch


def batched_broadband_MLP(norm_param, mlp_model, wavelength_m_asList, gridShape=None, torch_ones=None, override_WL_to_False=False):
    """Returns the transmittance. Utilizes a for-loop to batch over input wavelengths in order to not overload users memory

    Args:
        norm_param (tf.tensor): MLP input, normalized [0,1] of shape (N, D), where N is the number of cells and D is the shape parameter
        mlp_model (_type_): _description_
        wavelength_m_asList (_type_): _description_
        gridShape (_type_): _description_
    Returns:
        _type_: _description_
    """

    ## Initially localize and save the torch_ones to cache (This will be an issue if the length of wavelength list changes in the same call)
    wavelength_input_flag = False if override_WL_to_False else mlp_model.get_wavelengthFlag()
    if wavelength_input_flag:
        torch_ones = torch.ones((norm_param.shape[0], 1), dtype=torch.float32, device=norm_param.device)

    hold_trans = []
    hold_phase = []
    for idx in range(len(wavelength_m_asList)):
        if wavelength_input_flag:  # Add the normalized wavelength to the last column of the input tensor
            norm_wl = mlp_model.normalizeWavelength(wavelength_m_asList[idx])
            use_param = torch.cat((norm_param, norm_wl * torch_ones), axis=-1)
        else:
            use_param = norm_param
        trans, phase = mlp_model.convert_output_complex(mlp_model(use_param), reshapeToSize=gridShape)

        hold_trans.append(trans.unsqueeze(0))
        hold_phase.append(phase.unsqueeze(0))

    hold_trans = torch.cat(hold_trans, axis=0)
    hold_phase = torch.cat(hold_phase, axis=0)

    return hold_trans, hold_phase
