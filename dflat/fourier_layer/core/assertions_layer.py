import torch


def check_input_type(inputs, use_dtype, device):
    # We will allow None type because point-source-locs can be None!
    modification_flag = False
    for idx, input_dat in enumerate(inputs):
        if input_dat is None:
            continue
        else:
            if not isinstance(input_dat, torch.Tensor):
                try:
                    modification_flag = True
                    inputs[idx] = torch.as_tensor(input_dat, dtype=use_dtype).to(device)
                except ValueError:
                    raise ValueError("Input data at index {} can't be converted to a PyTorch tensor on specified device".format(idx))
            elif input_dat.dtype != use_dtype:
                modification_flag = True
                inputs[idx] = input_dat.to(dtype=use_dtype)
            elif input_dat.device != torch.device(device):
                modification_flag = True
                input_dat = input_dat.to(device)

    if len(inputs) > 1:
        return inputs, modification_flag
    else:
        return inputs[0], modification_flag


def check_broadband_wavelength_parameters(parameters):
    if not ("wavelength_set_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_set_m")

    return


def check_single_wavelength_parameters(parameters):
    if not ("wavelength_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_m")

    if not (type(parameters["wavelength_m"]) == float):
        raise ValueError("Wavelength should be a single float")
    return
