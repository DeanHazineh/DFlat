import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

import dflat.plot_utilities as graphFunc

### DO NOT ALTER ANYTHING IN THIS FILE IF YOU DON"T KNOW WHAT YOU ARE DOING ELSE IT WILL BREAK THINGS


def get_current_path(folder_name: str):
    resource_path = Path(__file__).parent
    return str(resource_path.joinpath(folder_name)) + "/"


class MLP_Object(nn.Module):
    def __init__(self):
        super(MLP_Object, self).__init__()

        # Define class variables
        self._modelSavePath = ""
        self._dtype = torch.float32  # Output dtype while model itself is kept in float32 as standard
        self.trainingLoss = []
        self.testLoss = []
        self.__model_name = ""
        self.__accepts_wavelength = True
        self.__input_dim = 1
        self.__output_dim = 1
        self.__output_pol_state = 1

        # parameter limits wrapped into a list for generalized model usage
        self.__preprocessDataBounds = []
        self.__dataBoundsLabel = []
        self.__arch = None

    def __call__(self, y):
        return self.forward(y)

    def forward(self, y):
        # Default nn are trained in float32 and we want to keep it this way
        # It is more efficient to use mixed datatypes as our paradigm
        # if self._dtype != torch.float32:
        #     y = y.to(torch.float32)
        # Note in Neural layers, input are automatically casted to float32 already so this is unecessary

        y = self.__arch(y)

        if self._dtype != torch.float32:
            y = y.to(self._dtype)

        return y

    ###
    def set_arch(self, torch_sequential):
        self.__arch = torch_sequential
        return

    def set_model_dtype(self, dtype):
        self._dtype = dtype
        return

    def get_model_dtype(self):
        return self._dtype

    def set_modelSavePath(self, modelSavePath):
        self._modelSavePath = get_current_path(modelSavePath)

        if not os.path.exists(self._modelSavePath):
            os.makedirs(self._modelSavePath, exist_ok=True)
            os.makedirs(self._modelSavePath + "/trainingOutput/", exist_ok=True)

        if not os.path.exists(self._modelSavePath + "trainingOutput/png_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/png_images/", exist_ok=True)

        if not os.path.exists(self._modelSavePath + "trainingOutput/pdf_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/pdf_images/", exist_ok=True)

        return

    def set_preprocessDataBounds(self, preprocessDataBounds, boundLabels):
        self.__preprocessDataBounds = preprocessDataBounds
        self.__dataBoundsLabel = boundLabels
        return

    def get_preprocessDataBounds(self):
        return self.__preprocessDataBounds

    def set_model_name(self, name):
        self.__model_name = name
        return

    def get_model_name(self):
        return self.__model_name

    def set_wavelengthFlag(self, boolFlag):
        self.__accepts_wavelength = boolFlag
        return

    def get_wavelengthFlag(self):
        return self.__accepts_wavelength

    def set_input_shape(self, input_dim):
        self.__input_dim = input_dim
        return

    def get_input_shape(self):
        return self.__input_dim

    def set_output_shape(self, output_dim):
        self.__output_dim = output_dim
        return

    def get_output_shape(self):
        return self.__output_dim

    def set_output_pol_state(self, output_stack_num):
        self.__output_pol_state = output_stack_num
        return

    def get_output_pol_state(self):
        return self.__output_pol_state

    def customSaveCheckpoint(self, test_loss=[], training_loss=[], checkpoint_dictionary=None, verbose=False):
        # save weights to checkpoint file
        model_state = self.state_dict()
        if checkpoint_dictionary == None:
            checkpoint_dictionary = {"model_state_dict": model_state}
        else:
            checkpoint_dictionary["model_state_dict"] = model_state

        # if Losses are passed then manually update by concatenating
        if training_loss:
            self.trainingLoss = np.concatenate((self.trainingLoss, training_loss))
        if test_loss:
            self.testLoss = np.concatenate((self.testLoss, test_loss))

        checkpoint_dictionary["loss"] = self.trainingLoss
        checkpoint_dictionary["test_loss"] = self.testLoss

        fig = plt.figure(figsize=(10, 5))
        ax = graphFunc.addAxis(fig, 1, 2)
        ax[0].plot(self.trainingLoss, "b-.", label="training loss")
        ax[0].plot(self.testLoss, "r-.", label="test loss")
        graphFunc.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss", addLegend=True)

        ax[1].plot(np.log10(self.trainingLoss), "b-.")
        ax[1].plot(np.log10(self.testLoss), "r-.")
        graphFunc.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Traning Log(Loss)")

        plt.savefig(self._modelSavePath + "/trainingOutput/png_images/trainingLog_traininghistory.png")
        plt.savefig(self._modelSavePath + "/trainingOutput/pdf_images/trainingLog_traininghistory.pdf")
        plt.close()

        model_path = self._modelSavePath + "checkpoint.pth"
        torch.save(checkpoint_dictionary, model_path)

        if verbose:
            print(f"Model checkpoint saved to: ", model_path)

        return

    def customLoadCheckpoint(self, optimizer=None, scheduler=None):
        model_path = self._modelSavePath + "checkpoint.pth"
        print("Checking for model checkpoint at: ", model_path)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            print("\n Model Checkpoint Loaded \n")
            self.trainingLoss = checkpoint["loss"]
            self.testLoss = checkpoint["test_loss"]

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("\n Optimizer State Reloaded")

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        else:
            print("\n no model checkpoint found at\n", model_path)

        return len(self.trainingLoss)

    ####
    def normalizeInput(self, shapeList):
        # take in a list of shape parameters and normalize
        # based on the class pre-defined parameter limits
        # Ensures stability with NN initialization and compatability with constrained optimization
        outParams = []
        for counter, thisParam in enumerate(shapeList):
            parameterBounds = self.__preprocessDataBounds[counter]
            outParams.append((thisParam - parameterBounds[0]) / (parameterBounds[1] - parameterBounds[0]))

        return outParams

    def normalizeWavelength(self, wavelength_m):
        wavelength_preprocessBounds = self.__preprocessDataBounds[-1]
        wavelength_mlp = (wavelength_m - wavelength_preprocessBounds[0]) / (wavelength_preprocessBounds[1] - wavelength_preprocessBounds[0])

        return wavelength_mlp

    def convert_vectorParam_toMLPInput(self, shapeList_asvector):
        ### Sometimes desire mlp output in meshgrid form from a set of shape vectors.
        # this is just a convenient wrapper to call mlp output on a grid
        shapelist_asgrid = np.meshgrid(*shapeList_asvector)
        outParams = self.normalizeInput(shapelist_asgrid)

        return np.stack([param.flatten() for param in outParams], -1)

    def convert_shape_to_param(self, shape_vector):
        """(Helper function for neural optical models) Given an unormalized shape vector and an MLP model,
        return the normalized params, in [0,1].

        Args:
            `shape_vector` (np.float): Unnormalized shape vector for a cell, of form (N, D) where D is the shape degree.
            `MLP_model` (MLP_Object): A pre-trained neural optical model in DFlat.

        Returns:
            `float`: Model normalized parameter vector suitable for mlp input
        """
        paramList = [shape_vector[:, i : i + 1] for i in range(shape_vector.shape[1])]
        return torch.transpose(torch.stack(self.normalizeInput(paramList)), 0, 1).contiguous()

    def convert_param_to_shape(self, norm_param):
        """(Helper function for neural optical models) Given a model normalized parameter array, return the unnormalized
        shape vector which corresponds to structure lengths in m.

        Args:
            `norm_param` (np.float): Normalized parameters for a cell, of form (N, D) where D is the shape degree.
            `MLP_model` (MLP_Object): A pre-trained neural optical model in DFlat.

        Returns:
            `np.float`: The unnormalized parameter vector, where lengths are back in meaningful units of m.
        """
        is_tensor_flag = torch.is_tensor(norm_param)

        databounds = self.get_preprocessDataBounds()
        shapeDegree = len(databounds) - 1
        if is_tensor_flag:
            return torch.cat([norm_param[:, i : i + 1] * (databounds[i][1] - databounds[i][0]) + databounds[i][0] for i in range(shapeDegree)], dim=1)
        else:
            return np.array([norm_param[:, i] * (databounds[i][1] - databounds[i][0]) + databounds[i][0] for i in range(shapeDegree)]).T
