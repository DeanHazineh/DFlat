import torch
import warnings


def check_inputs(inputs, device):
    modification_Flag = False

    # Ensure inputs are tensors
    for idx, input_dat in enumerate(inputs):
        if not torch.is_tensor(input):
            modification_Flag = True
            inputs[idx] = torch.as_tensor(input_dat).to(device)
        elif input_dat.device != torch.device(device):
            modification_Flag = True
            input_dat = input_dat.to(device)

    # Ensure same datatypes
    if inputs[0].dtype != inputs[1].dtype:
        modification_Flag = True
        warnings.warn("Inputs must have the same datatype. Cast second input to the first datatype")
        inputs[1] = inputs[1].type(inputs[0].dtype)

    if inputs[0].dim() != inputs[1].dim():
        raise ValueError("PSF and AIF Image should have the same tensor rank")

    return inputs, modification_Flag
