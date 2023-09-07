import torch
import numpy as np

from dflat.metasurface_library import libraryClass as library
from .arch_Core_class import MLP_Object
from dflat.GDSII_utilities.core.gen_gds_fun import assemble_nanofins_gds, assemble_nanocylinder_gds
from dflat.GDSII_utilities.core.prep_util import *


## SUB-BASE: (CHILD - DO NOT ALTER THIS UNLESS YOU KNOW THE DETAILS; ADD NEW CHILDREN FOR DIFFERENT METASURFACE LIBRARIES)


class MLP_Nanofins_U350_H600(MLP_Object):
    def __init__(self, dtype=torch.float64):
        super().__init__()

        # Define model input normalization during training/inference
        # Units in m; These are private class variables and should not be altered unless
        # the corresponding library class was altered
        # NOTE: this is hardcoded here rather than loading directly from library because we
        # do not want the computational/memory cost of loading the library when model is
        # used for inference only!
        self.cell_size = {"x": 350e-9, "y": 350e-9}
        __param1Limits = [60e-9, 300e-9]  # corresponds to length x m for data
        __param2Limits = [60e-9, 300e-9]  # corresponds to length y m for data
        __param3Limits = [310e-9, 750e-9]  # corresponds to wavelength m
        __paramLimit_labels = ["lenx_m", "leny_m", "wavelength_m"]

        self.set_preprocessDataBounds([__param1Limits, __param2Limits, __param3Limits], __paramLimit_labels)
        self.set_model_dtype(dtype)
        self.set_input_shape(3)
        self.set_output_shape(6)
        self.set_output_pol_state(2)
        self.set_wavelengthFlag(True)

        return

    def returnLibraryAsTrainingData(self):
        # FDTD generated data loaded from library class file
        useLibrary = library.Nanofins_U350nm_H600nm()
        params = useLibrary.params
        transmittance = useLibrary.transmittance
        phase = useLibrary.phase

        # Normalize inputs (always done based on self model normalize function)
        normalizedParams = self.normalizeInput(params)
        trainx = np.stack([param.flatten() for param in normalizedParams], -1)
        trainy = np.stack(
            [
                np.cos(phase[0, :, :, :]).flatten(),  # cos of phase x polarized light
                np.sin(phase[0, :, :, :]).flatten(),  # sin of phase x polarized light
                np.cos(phase[1, :, :, :]).flatten(),  # cos of phase y polarized light
                np.sin(phase[1, :, :, :]).flatten(),  # sin of phase y polarized light
                transmittance[0, :, :, :].flatten(),  # x transmission
                transmittance[1, :, :, :].flatten(),  # y transmission
            ],
            -1,
        )

        return trainx, trainy

    def get_trainingParam(self):
        useLibrary = library.Nanofins_U350nm_H600nm()
        return useLibrary.params

    def convert_output_complex(self, y_model, reshapeToSize=None):
        phasex = torch.atan2(y_model[:, 1], y_model[:, 0])
        phasey = torch.atan2(y_model[:, 3], y_model[:, 2])
        transx = y_model[:, 4]
        transy = y_model[:, 5]

        # allow an option to reshape to a grid size
        if reshapeToSize is not None:
            phasex = torch.reshape(phasex, reshapeToSize)
            transx = torch.reshape(transx, reshapeToSize)
            phasey = torch.reshape(phasey, reshapeToSize)
            transy = torch.reshape(transy, reshapeToSize)

            return torch.squeeze(torch.stack([transx, transy]), 1), torch.squeeze(torch.stack([phasex, phasey]), 1)

        return torch.stack([transx, transy]), torch.stack([phasex, phasey])

    def write_param_to_gds(self, param_array, ms_dx_m, savepath, aperture=None, rotation_array=None, add_markers=False):
        # param_array will be passed in with shape D x Ny x Nx.
        # Need to reshape to convert param to shape

        cell_design_tuple = prepare_gds(param_array, ms_dx_m, self.cell_size, aperture, rotation_array)

        # Convert the param to shape
        param_array = cell_design_tuple[0]
        init_shape = param_array.shape
        param_array = np.reshape(np.transpose(param_array, [1, 2, 0]), [-1, init_shape[0]])
        shape_array = self.convert_param_to_shape(param_array)
        shape_array = np.transpose(np.reshape(shape_array, [*init_shape[1:], -1]), [2, 0, 1])
        cell_design_tuple[0] = shape_array

        # call generation fun
        assemble_nanofins_gds(cell_design_tuple, self.cell_size, savepath, add_markers=add_markers)

        return


class MLP_Nanocylinders_U180_H600(MLP_Object):
    def __init__(self, dtype=torch.float64):
        super().__init__()

        # Define model input normalization during training/inference
        # Units in m; These are private class variables and should not be altered unless
        # the corresponding library class was altered
        # NOTE: this is hardcoded here rather than loading directly from library because we
        # do not want the computational/memory cost of loading the library when model is
        # used for inference only!
        self.cell_size = {"x": 180e-9, "y": 180e-9}
        __param1Limits = [30e-9, 150e-9]  # corresponds to radius m of cylinder for data
        __param2Limits = [310e-9, 750e-9]  # corresponds to wavelength m for training data
        __paramLimit_labels = ["radius_m", "wavelength_m"]
        self.set_preprocessDataBounds([__param1Limits, __param2Limits], __paramLimit_labels)

        self.set_model_dtype(dtype)
        self.set_input_shape(2)
        self.set_output_shape(3)
        self.set_output_pol_state(1)
        self.set_wavelengthFlag(True)

        return

    def returnLibraryAsTrainingData(self):
        # FDTD generated data loaded from library class file
        useLibrary = library.Nanocylinders_U180nm_H600nm()
        params = useLibrary.params
        phase = useLibrary.phase
        transmittance = useLibrary.transmittance

        # Normalize inputs (always done based on self model normalize function)
        normalizedParams = self.normalizeInput(params)
        trainx = np.stack([param.flatten() for param in normalizedParams], -1)
        trainy = np.stack(
            [
                np.cos(phase[:, :]).flatten(),  # cos of phase x polarized light
                np.sin(phase[:, :]).flatten(),  # sin of phase x polarized light
                transmittance[:, :].flatten(),  # x transmission
            ],
            -1,
        )

        return trainx, trainy

    def get_trainingParam(self):
        useLibrary = library.Nanocylinders_U180nm_H600nm()
        return useLibrary.params

    def convert_output_complex(self, y_model, reshapeToSize=None):
        phasex = torch.atan2(y_model[:, 1], y_model[:, 0])
        transx = y_model[:, 2]

        # allow an option to reshape to a grid size
        if reshapeToSize is not None:
            phasex = torch.reshape(phasex, reshapeToSize)
            transx = torch.reshape(transx, reshapeToSize)

        return transx, phasex

    def write_param_to_gds(self, param_array, ms_dx_m, savepath, aperture=None, rotation_array=None, add_markers=False):
        # param_array will be passed in with shape D x Ny x Nx.
        # Need to reshape to convert param to shape
        cell_design_tuple = prepare_gds(param_array, ms_dx_m, self.cell_size, aperture, rotation_array)

        # Convert the param to shape
        param_array = cell_design_tuple[0]
        init_shape = param_array.shape
        param_array = np.reshape(np.transpose(param_array, [1, 2, 0]), [-1, init_shape[0]])
        shape_array = self.convert_param_to_shape(param_array)
        shape_array = np.transpose(np.reshape(shape_array, [*init_shape[1:], -1]), [2, 0, 1])
        cell_design_tuple[0] = shape_array

        # call generation fun
        assemble_nanocylinder_gds(cell_design_tuple, self.cell_size, savepath, add_markers=add_markers)

        return
