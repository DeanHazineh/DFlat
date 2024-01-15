from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

from torch.utils.data import DataLoader, Dataset, random_split
from dflat.plot_utilities import formatPlot


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / "data/"
    return resource_path.joinpath(file_name)


def outpath(savepath):
    if savepath is None:
        current_path = os.path.dirname(os.path.abspath(__file__))
        savepath = os.path.join(current_path, "out")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    return savepath


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Nanofins_TiO2_U350nm_H600nm:
    def __init__(self):
        self.datpath = get_path_to_data("Nanofins_TiO2_Unit350nm_Height600nm_FDTD.mat")
        self.loaded = False
        self.trans = None
        self.phase = None
        self.params = None

        self.param_names = ["leny", "lenx", "wavelength"]
        self.param_limits = [[60e-9, 300e-9], [60e-9, 300e-9], [310e-9, 750e-9]]

    def load_data(self):
        # Phase and transmittance has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        data = scipy.io.loadmat(self.datpath)
        self.phase = np.angle(np.exp(1j * data["phase"])) + np.pi
        self.trans = np.clip(np.sqrt(data["transmission"]), 0, np.finfo(np.float32).max)
        self.params = [data["leny"], data["lenx"], data["wavelength_m"].flatten()]
        self.loaded = True
        return

    def plot(self, savepath=None):
        if not self.loaded:
            self.load_data()

        lx = self.params[0] * 1e9
        ly = self.params[1] * 1e9
        lam = self.params[2] * 1e9

        phase = self.phase
        trans = self.trans
        sh = phase.shape
        fig, ax = plt.subplots(2, 4, figsize=(17, 9))
        for i in range(2):
            ax[i, 0].imshow(trans[i, sh[1] // 2, :, :], vmin=0, vmax=1)
            formatPlot(
                fig,
                ax[i, 0],
                yvec=lam,
                xvec=lx,
                setAspect="auto",
                ylabel="x-pol \n width x (nm)"
                if i == 0
                else "y-pol \n wavelength (nm)",
                title="transmittance",
                xlabel="wavelength (nm) ",
            )

            ax[i, 1].imshow(trans[i, :, sh[2] // 2, :], vmin=0, vmax=1)
            formatPlot(
                fig,
                ax[i, 1],
                yvec=ly,
                xvec=lam,
                setAspect="auto",
                addcolorbar=True,
                title="transmittance",
                ylabel="width y (nm)",
                xlabel="wavelength (nm) ",
            )

            ax[i, 2].imshow(
                phase[i, sh[1] // 2, :, :], vmin=0, vmax=2 * np.pi, cmap="hsv"
            )
            formatPlot(
                fig,
                ax[i, 2],
                yvec=lam,
                xvec=lx,
                setAspect="auto",
                title="phase",
                ylabel="width x (nm)",
                xlabel="wavelength (nm) ",
            )

            ax[i, 3].imshow(
                phase[i, :, sh[2] // 2, :], vmin=0, vmax=2 * np.pi, cmap="hsv"
            )
            formatPlot(
                fig,
                ax[i, 3],
                yvec=ly,
                xvec=lam,
                setAspect="auto",
                addcolorbar=True,
                title="phase",
                ylabel="width y (nm)",
                xlabel="wavelength (nm) ",
            )

        folder_path = outpath(savepath)
        plt.savefig(folder_path + "/" + self.__class__.__name__ + ".png")
        plt.close()

        return

    def dataloader(self, test_split=0.15, batch_size=256):
        assert test_split < 1.0
        if not self.loaded:
            self.load_data()
        trans = self.trans
        phase = self.phase / 2 / np.pi
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for i, (p, l) in enumerate(zip(self.params, self.param_limits))
            ],
            indexing="ij",
        )

        x = np.stack([p.flatten() for p in params], -1)
        y = np.stack(
            [
                trans[0].flatten(),
                np.cos(phase[0]).flatten(),
                np.sin(phase[0]).flatten(),
                trans[1].flatten(),
                np.cos(phase[1]).flatten(),
                np.sin(phase[1]).flatten(),
            ],
            -1,
        )

        # create a dataloader
        dataset = CustomDataset(x, y)
        total_size = len(dataset)
        train_size = int(total_size * (1 - test_split))
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    # def optical_response_to_param(
    #     self, trans_asList, phase_asList, wavelength_asList, reshape=True, fast=False
    # ):
    #     """Computes the shape vector (here, nanocylinder radius) that most closely matches a transmittance and phase profile input.
    #     Note that each transmittance and phase profile for a given wavelength
    #     (in wavelength_aslist) is assumed to be a seperate lens. This is a naive-table look-up function.

    #     Args:
    #         trans_asList (float): list of transmittance profiles
    #         phase_asList (float): list of phase profiles
    #         wavelength_asList (float): list of wavelengths corresponding to the target transmittance and phase profiles
    #         reshape (float): Boolean if returned shape vectors are to be given in the same shape as the input or if just return as a flattened list
    #         fast (bool, optional): Whether to do exhaustive min-search or a less accurate but fast dictionary look-up. The dictionary look-up assumed the target transmittance is unity and finds the best phase match. Defaults to False.

    #     Returns:
    #         list: List containing the shape vector for each trans and phase pair passed in (elements of the input list)
    #     """

    #     ### Run input assertions
    #     # List assertion
    #     if not all(
    #         type(input) is list
    #         for input in [trans_asList, phase_asList, wavelength_asList]
    #     ):
    #         raise TypeError(
    #             "optical_response_to_param: trans, phase, and wavelength must all be passed in as lists"
    #         )

    #     # List length assertion
    #     length = len(wavelength_asList)
    #     if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
    #         raise ValueError(
    #             "optical_response_to_param: All lists must be the same length"
    #         )

    #     # Assert polarization basis dimension is two
    #     if not all([trans.shape[0] == 2 for trans in trans_asList]) or not all(
    #         [phase.shape[0] == 2 for phase in phase_asList]
    #     ):
    #         raise ValueError(
    #             "optical_response_to_param: All transmission/phase profiles in the list must be a stack of two profiles, (2, Ny, Nx)"
    #         )

    #     ### Assemble metasurfaces
    #     shape_Vector = []
    #     shape_Vector_norm = []
    #     for i in range(length):
    #         use_wavelength = wavelength_asList[i]
    #         ms_trans = trans_asList[i]
    #         ms_phase = phase_asList[i]
    #         initial_shape = [1, *ms_trans.shape[-2:]]

    #         if fast:
    #             design_lx, design_ly = lookup_D2_pol2(
    #                 self.name + ".pickle", use_wavelength, ms_trans, ms_phase
    #             )
    #         else:
    #             design_lx, design_ly = minsearch_D2_pol2(
    #                 self.phase,
    #                 self.transmittance,
    #                 self.params[0][:, :, 0].flatten(),
    #                 self.params[1][:, :, 0].flatten(),
    #                 self.params[2][0, 0, :],
    #                 use_wavelength,
    #                 ms_trans,
    #                 ms_phase,
    #             )

    #         # Define a normalized shape vector for convenience
    #         norm_design_lx = np.clip(
    #             (design_lx - self.__param1Limits[0])
    #             / (self.__param1Limits[1] - self.__param1Limits[0]),
    #             0,
    #             1,
    #         )
    #         norm_design_ly = np.clip(
    #             (design_ly - self.__param2Limits[0])
    #             / (self.__param2Limits[1] - self.__param2Limits[0]),
    #             0,
    #             1,
    #         )

    #         if reshape:
    #             shape_Vector.append(
    #                 np.vstack(
    #                     (
    #                         np.reshape(design_lx, initial_shape),
    #                         np.reshape(design_ly, initial_shape),
    #                     )
    #                 )
    #             )
    #             shape_Vector_norm.append(
    #                 np.vstack(
    #                     (
    #                         np.reshape(norm_design_lx, initial_shape),
    #                         np.reshape(norm_design_ly, initial_shape),
    #                     )
    #                 )
    #             )
    #         else:
    #             shape_Vector.append(
    #                 np.hstack(
    #                     (np.expand_dims(design_lx, -1), np.expand_dims(design_ly, -1))
    #                 )
    #             )
    #             shape_Vector_norm.append(
    #                 np.hstack(
    #                     (
    #                         np.expand_dims(norm_design_lx, -1),
    #                         np.expand_dims(norm_design_ly, -1),
    #                     )
    #                 )
    #             )

    #     return shape_Vector, shape_Vector_norm


class Nanocylinders_TiO2_U180nm_H600nm:
    def __init__(self):
        rawPath = get_path_to_data("Nanocylinders_TiO2_Unit180nm_Height600nm_FDTD.mat")
        data = scipy.io.loadmat(rawPath)

        # Phase and transmission has shape [wavelength=441, lenr=191]
        self.phase = data["phase"]
        self.transmittance = np.sqrt(
            np.clip(data["transmission"], 0, np.finfo(np.float32).max)
        )

        self.param1 = data["radius_m"]
        self.param2 = data["wavelength_m"]

        self.params = np.meshgrid(self.param1, self.param2, indexing="xy")
        self.paramLimits = [[30e-9, 150e-9], [310e-9, 750e-9]]

    def plot(self, savepath=None):
        phase = self.phase
        trans = self.transmittance

        r = self.params[0][0, :] * 1e9
        lam = self.params[1][:, 0] * 1e9

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(trans, vmin=0, vmax=1)
        formatPlot(
            fig,
            ax[0],
            xvec=r,
            yvec=lam,
            xlabel="radius (nm)",
            ylabel="wavelength (nm)",
            title="transmittance",
            addcolorbar=True,
            setAspect="auto",
        )
        ax[1].imshow(phase, vmin=0, vmax=2 * np.pi, cmap="hsv")
        formatPlot(
            fig,
            ax[1],
            xvec=r,
            yvec=lam,
            xlabel="radius (nm)",
            ylabel="wavelength (nm)",
            title="transmittance",
            addcolorbar=True,
            setAspect="auto",
        )

        folder_path = outpath(savepath)
        plt.savefig(folder_path + "/" + self.__class__.__name__ + ".png")
        plt.close()


#     def optical_response_to_param(
#         self, trans_asList, phase_asList, wavelength_asList, reshape=True, fast=False
#     ):
#         """Computes the shape vector (here, nanocylinder radius) that most closely matches a transmittance and phase profile input. Note that each transmittance and phase profile for a given wavelength
#         (in wavelength_aslist) is assumed to be a seperate lens. This is a naive-table look-up function.

#         Args:
#             trans_asList (float): list of transmittance profiles
#             phase_asList (float): list of phase profiles
#             wavelength_asList (float): list of wavelengths corresponding to the target transmittance and phase profiles
#             reshape (float): Boolean if returned shape vectors are to be given in the same shape as the input or if just return as a flattened list
#             fast (bool, optional): Whether to do exhaustive min-search or a less accurate but fast dictionary look-up. The dictionary look-up assumed the target transmittance is unity and finds the best phase match. Defaults to False.

#         Returns:
#             list: List containing the shape vector for each trans and phase pair passed in (elements of the input list)
#         """

#         # list assertion
#         length = len(wavelength_asList)
#         if not all(
#             type(input) is list
#             for input in [trans_asList, phase_asList, wavelength_asList]
#         ):
#             raise TypeError(
#                 "optical_response_to_param: trans, phase, and wavelength must all be passed in as lists"
#             )

#         # list length assertion
#         if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
#             raise ValueError(
#                 "optical_response_to_param: All lists must be the same length"
#             )

#         # polarization dimensionality check
#         if not all([trans.shape[0] == 1 for trans in trans_asList]) or not all(
#             [phase.shape[0] == 1 for phase in phase_asList]
#         ):
#             raise ValueError(
#                 "optical_response_to_param: All transmission/phase profiles in the list must be a single transmission profile, (1, Ny, Nx)"
#             )

#         ### Assemble metasurfaces
#         shape_Vector = []
#         shape_Vector_norm = []
#         for i in range(length):
#             use_wavelength = wavelength_asList[i]
#             ms_trans = trans_asList[i]
#             ms_phase = phase_asList[i]
#             initial_shape = [1, *ms_trans.shape[-2:]]

#             if fast:
#                 design_radius = lookup_D1_pol1(
#                     self.name + ".pickle", use_wavelength, ms_trans, ms_phase
#                 )
#             else:
#                 design_radius = minsearch_D1_pol1(
#                     self.phase,
#                     self.transmittance,
#                     self.param1.flatten(),
#                     self.param2.flatten(),
#                     use_wavelength,
#                     ms_trans,
#                     ms_phase,
#                 )

#             norm_design_radius = np.clip(
#                 (design_radius - self.__param1Limits[0])
#                 / (self.__param1Limits[1] - self.__param1Limits[0]),
#                 0,
#                 1,
#             )

#             if reshape:
#                 shape_Vector.append(np.reshape(design_radius, initial_shape))
#                 shape_Vector_norm.append(np.reshape(norm_design_radius, initial_shape))
#             else:
#                 shape_Vector.append(np.expand_dims(design_radius, -1))
#                 shape_Vector_norm.append(np.expand_dims(norm_design_radius, -1))

#         return shape_Vector, shape_Vector_norm


class Nanoellipse_TiO2_U350nm_H600nm(Nanofins_TiO2_U350nm_H600nm):
    def __init__(self):
        super().__init__()
        rawPath = get_path_to_data("Nanoellipse_TiO2_Unit350nm_Height600nm_FDTD.mat")
        data = scipy.io.loadmat(rawPath)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.transmittance = np.sqrt(
            np.clip(data["transmission"], 0, np.finfo(np.float32).max)
        )

        self.param1 = data["lenx"]
        self.param2 = data["leny"]
        self.param3 = data["wavelength_m"].flatten()

        self.params = np.meshgrid(self.param1, self.param2, self.param3, indexing="xy")
        self.paramLimits = [[60e-9, 300e-9], [60e-9, 300e-9], [310e-9, 750e-9]]


if __name__ == "__main__":
    lib = Nanofins_TiO2_U350nm_H600nm()
    lib.plot()
    lib.dataloader()
