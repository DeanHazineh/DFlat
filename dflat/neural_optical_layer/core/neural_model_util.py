import torch
import numpy as np
from .models_DNN import *
from .caller_MLP import *

listModelNames = mlp_model_names


def list_models():
    print(listModelNames)
    return


def load_neuralModel(model_selection_string, dtype=torch.float64):
    if model_selection_string not in listModelNames:
        raise ValueError("mlp_layer: requested MLP is not one of the supported libraries")
    else:
        mlp = globals()[model_selection_string]
        mlp = mlp(dtype)
        mlp.customLoadCheckpoint()

    # Freeze the weights
    for param in mlp.parameters():
        param.requires_grad = False
    mlp.eval()

    print("Loaded (frozen) Model: ", mlp.get_model_name())
    return mlp


def init_norm_param(init_type, dtype, gridShape, mlp_input_shape, init_args=[]):
    if init_type == "uniform":
        norm_param = 0.5 * torch.ones(mlp_input_shape - 1, gridShape[-2], gridShape[-1], dtype=dtype)

    elif init_type == "random":
        norm_param = torch.rand(mlp_input_shape - 1, gridShape[-2], gridShape[-1], dtype=dtype)

    else:
        raise ValueError("initialize_norm_param: invalid init_type string;")

    return norm_param


def flatten_reshape_shape_parameters(shape_vector):
    """Takes a shape/param vector of (D, PixelsY, PixelsX) and flattens to shape (PixelsY*PixelsX,D)"""
    init_shape = shape_vector.shape
    shape_vector = torch.permute(shape_vector, (1, 2, 0)).contiguous()
    return shape_vector.view(init_shape[1] * init_shape[2], -1)
