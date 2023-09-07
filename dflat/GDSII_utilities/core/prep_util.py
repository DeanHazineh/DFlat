import warnings
import cv2
import numpy as np
import torch
from dflat.fourier_layer import radial_2d_transform


def upsample_with_cv2(inputs, upsample_factor):
    # Each input should have the shape (D, Ny, Nx)
    outputs = []
    for input in inputs:
        this = cv2.resize(np.transpose(input, [1, 2, 0]), None, fx=upsample_factor["x"], fy=upsample_factor["y"], interpolation=cv2.INTER_NEAREST)
        # cv2 resize squeezes one dimension so we should put that back
        if len(this.shape) != 3:
            this = np.expand_dims(this, -1)
        outputs.append(np.transpose(this, [2, 0, 1]))

    return outputs


def check_inputs(input_list):
    for iter, this_input in enumerate(input_list):
        if this_input is None:
            continue
        else:
            if torch.is_tensor(this_input):
                this_input = this_input.cpu().detach().numpy()
            if len(this_input.shape) != 3:
                raise ValueError("All inputs tensors must be a rank 3 tensor like [D (or 1), Ny, Nx]")
            if this_input.shape[1] == 1:  # Convert radial tensor to 2D
                this_input = np.squeeze(radial_2d_transform(this_input), 1)
            input_list[iter] = this_input

    return input_list


def prepare_gds(shape_array, ms_dx_m, cell_size, aperture=None, rotation_array=None):
    # For GDS utilities we need to convert to numpy instead of torch
    shape_array, aperture, rotation_array = check_inputs([shape_array, aperture, rotation_array])
    grid_shape = shape_array.shape[-2:]
    aperture = np.ones((1, *grid_shape), dtype=np.uint8) if aperture is None else aperture.astype(np.uint8)
    rotation_array = np.zeros((1, *grid_shape)) if rotation_array is None else rotation_array

    # Upsample tensors
    print(ms_dx_m, cell_size)

    upsample_dict = {"x": ms_dx_m["x"] / cell_size["x"], "y": ms_dx_m["y"] / cell_size["y"]}
    if not (np.isclose(upsample_dict["x"] % 1, 0.0, 1e-5) or np.isclose(upsample_dict["y"] % 1, 0.0, 1e-5)):
        print(upsample_dict)
        warnings.warn("WARNING: The lens discretization ms_dx_m should be a multiple integer of the cell_size to ensure accuracy")

    out = upsample_with_cv2([shape_array, rotation_array, aperture], upsample_dict)
    return out
