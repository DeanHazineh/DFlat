import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from siren_pytorch import SirenNet

from .arch_Parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600

mlp_model_names = [
    "MLP_Nanocylinders_Dense256_U180_H600_SIREN100",
    "MLP_Nanofins_Dense1024_U350_H600_SIREN100",
]


class MLP_Nanocylinders_Dense256_U180_H600_SIREN100(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)
        self.set_model_name("MLP_Nanocylinders_Dense256_U180_H600_SIREN100")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense256_U180_H600_SIREN100/")

        # Define a new architecture
        input_dim = self.get_input_shape()
        output_dim = self.get_output_shape()
        arch = SirenNet(
            dim_in=input_dim,  # input dimension, ex. 2d coor
            dim_hidden=256,  # hidden dimension
            dim_out=output_dim,  # output dimension, ex. rgb value
            num_layers=2,  # number of layers
            final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
            w0_initial=100.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        self.set_arch(arch)


class MLP_Nanofins_Dense1024_U350_H600_SIREN100(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)
        self.set_model_name("MLP_Nanofins_Dense1024_U350_H600_SIREN100")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense1024_U350_H600_SIREN100/")

        # Define a new architecture
        input_dim = self.get_input_shape()
        output_dim = self.get_output_shape()
        arch = SirenNet(
            dim_in=input_dim,  # input dimension, ex. 2d coor
            dim_hidden=1024,  # hidden dimension
            dim_out=output_dim,  # output dimension, ex. rgb value
            num_layers=2,  # number of layers
            final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
            w0_initial=100.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        self.set_arch(arch)
