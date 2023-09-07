import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .core.util_library_lookup import *
import dflat.plot_utilities as df_plt

listLibraryNames = ["Nanofins_U350nm_H600nm", "Nanocylinders_U180nm_H600nm", "Nanoellipse_U350nm_H600nm"]


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / "core/raw_meta_libraries"
    return resource_path.joinpath(file_name)


class Nanofins_U350nm_H600nm:
    def __init__(self):
        __rawPath = get_path_to_data("data_Nanofins_Unit350nm_Height600nm_EngineFDTD.mat")
        data = scipy.io.loadmat(__rawPath)
        self.name = "Nanofins_U350nm_H600nm"

        # Phase and transmittance has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.transmittance = np.sqrt(np.clip(data["transmission"], 0, np.finfo(np.float32).max))

        # the input parameters (lenx, leny, wavelength) are loaded in here
        # all parameters must be converted to meshgrid format for compatibility with model class
        # param1, param2, and param3 all have units of m
        self.param1 = data["lenx"]
        self.param2 = data["leny"]
        self.param3 = data["wavelength_m"].flatten()
        self.params = np.meshgrid(self.param1, self.param2, self.param3)

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [60e-9, 300e-9]
        self.__param2Limits = [60e-9, 300e-9]
        self.__param3Limits = [310e-9, 750e-9]

    def plotLibrary(self, savepath=None):
        # This function is added so the user may get familiar with the data themselves
        # and poke around the visualization as a test.
        # It is also a test that we have flattened and packaged things in the way that is expected when later loading
        # the training data

        lx = self.params[0][0, :, 0] * 1e9
        ly = self.params[1][:, 0, 0] * 1e9
        wavelength = self.params[2][0, 0, :] * 1e9

        fig = plt.figure(figsize=(12, 6))
        ax = df_plt.addAxis(fig, 2, 4)
        wl_use = 532
        wl_idx = np.argmin(np.abs(wavelength - wl_use))
        ax[0].imshow(self.transmittance[0, :, :, wl_idx], vmin=0, vmax=1)
        ax[1].imshow(self.transmittance[1, :, :, wl_idx], vmin=0, vmax=1)
        ax[4].imshow(self.phase[0, :, :, wl_idx], vmin=0, vmax=2 * np.pi, cmap="hsv")
        ax[5].imshow(self.phase[1, :, :, wl_idx], vmin=0, vmax=2 * np.pi, cmap="hsv")

        yidx = 24
        ax[2].imshow(self.transmittance[0, 24, :, :].T, vmin=0, vmax=1)
        ax[3].imshow(self.transmittance[1, 24, :, :].T, vmin=0, vmax=1)
        ax[6].imshow(self.phase[0, 24, :, :].T, vmin=0, vmax=2 * np.pi, cmap="hsv")
        ax[7].imshow(self.phase[1, 24, :, :].T, vmin=0, vmax=2 * np.pi, cmap="hsv")

        for axidx in [0, 1, 4, 5]:
            thisax = ax[axidx]
            df_plt.formatPlots(fig, thisax, None, setAspect="auto", xgrid_vec=lx, ygrid_vec=ly)
        for axidx in [2, 3, 6, 7]:
            thisax = ax[axidx]
            df_plt.formatPlots(fig, thisax, None, setAspect="auto", xgrid_vec=lx, ygrid_vec=wavelength)

        ax1 = fig.add_axes([0.125, 0.85, 0.35, 0.05], frameon=False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("Wavelength Slice: 532 nm ", size=12)
        ax2 = fig.add_axes([0.55, 0.85, 0.35, 0.05], frameon=False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title(f"Fin width Ly Slice: {ly[yidx]:.0f} nm", size=12)

        if savepath is not None:
            plt.savefig(savepath + ".png")
            plt.savefig(savepath + ".pdf")
        else:
            plt.show()

        return

    def optical_response_to_param(self, trans_asList, phase_asList, wavelength_asList, reshape=True, fast=False):
        """Computes the shape vector (here, nanocylinder radius) that most closely matches a transmittance and phase profile input.
        Note that each transmittance and phase profile for a given wavelength
        (in wavelength_aslist) is assumed to be a seperate lens. This is a naive-table look-up function.

        Args:
            trans_asList (float): list of transmittance profiles
            phase_asList (float): list of phase profiles
            wavelength_asList (float): list of wavelengths corresponding to the target transmittance and phase profiles
            reshape (float): Boolean if returned shape vectors are to be given in the same shape as the input or if just return as a flattened list
            fast (bool, optional): Whether to do exhaustive min-search or a less accurate but fast dictionary look-up. The dictionary look-up assumed the target transmittance is unity and finds the best phase match. Defaults to False.

        Returns:
            list: List containing the shape vector for each trans and phase pair passed in (elements of the input list)
        """

        ### Run input assertions
        # List assertion
        if not all(type(input) is list for input in [trans_asList, phase_asList, wavelength_asList]):
            raise TypeError("optical_response_to_param: trans, phase, and wavelength must all be passed in as lists")

        # List length assertion
        length = len(wavelength_asList)
        if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
            raise ValueError("optical_response_to_param: All lists must be the same length")

        # Assert polarization basis dimension is two
        if not all([trans.shape[0] == 2 for trans in trans_asList]) or not all([phase.shape[0] == 2 for phase in phase_asList]):
            raise ValueError("optical_response_to_param: All transmission/phase profiles in the list must be a stack of two profiles, (2, Ny, Nx)")

        ### Assemble metasurfaces
        shape_Vector = []
        shape_Vector_norm = []
        for i in range(length):
            use_wavelength = wavelength_asList[i]
            ms_trans = trans_asList[i]
            ms_phase = phase_asList[i]
            initial_shape = [1, *ms_trans.shape[-2:]]

            if fast:
                design_lx, design_ly = lookup_D2_pol2(self.name + ".pickle", use_wavelength, ms_trans, ms_phase)
            else:
                design_lx, design_ly = minsearch_D2_pol2(
                    self.phase,
                    self.transmittance,
                    self.params[0][:, :, 0].flatten(),
                    self.params[1][:, :, 0].flatten(),
                    self.params[2][0, 0, :],
                    use_wavelength,
                    ms_trans,
                    ms_phase,
                )

            # Define a normalized shape vector for convenience
            norm_design_lx = np.clip(
                (design_lx - self.__param1Limits[0]) / (self.__param1Limits[1] - self.__param1Limits[0]),
                0,
                1,
            )
            norm_design_ly = np.clip(
                (design_ly - self.__param2Limits[0]) / (self.__param2Limits[1] - self.__param2Limits[0]),
                0,
                1,
            )

            if reshape:
                shape_Vector.append(
                    np.vstack(
                        (
                            np.reshape(design_lx, initial_shape),
                            np.reshape(design_ly, initial_shape),
                        )
                    )
                )
                shape_Vector_norm.append(
                    np.vstack(
                        (
                            np.reshape(norm_design_lx, initial_shape),
                            np.reshape(norm_design_ly, initial_shape),
                        )
                    )
                )
            else:
                shape_Vector.append(np.hstack((np.expand_dims(design_lx, -1), np.expand_dims(design_ly, -1))))
                shape_Vector_norm.append(
                    np.hstack(
                        (
                            np.expand_dims(norm_design_lx, -1),
                            np.expand_dims(norm_design_ly, -1),
                        )
                    )
                )

        return shape_Vector, shape_Vector_norm


class Nanocylinders_U180nm_H600nm:
    def __init__(self):
        __rawPath = get_path_to_data("data_Nanocylinders_Unit180nm_Height600nm_EngineFDTD.mat")
        data = scipy.io.loadmat(__rawPath)
        self.name = "Nanocylinders_U180nm_H600nm"

        # Phase and transmission has shape [wavelength=441, lenr=191]
        self.phase = data["phase"]
        self.transmittance = np.sqrt(np.clip(data["transmission"], 0, np.finfo(np.float32).max))

        # the input parameters (lenr, wavelength) are loaded in here
        # all parameters must be converted to meshgrid format for compatibility with model class
        # param1, param2, all have units of m
        self.param1 = data["radius_m"]
        self.param2 = data["wavelength_m"]
        self.params = np.meshgrid(self.param1, self.param2)

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [30e-9, 150e-9]
        self.__param2Limits = [310e-9, 750e-9]

    def plotLibrary(self, savepath=[]):
        # This function is added so the user may get familiar with the data themselves
        # and poke around the visualization as a test
        lr = self.params[0][0, :] * 1e9
        wl = self.params[1][:, 0] * 1e9

        fig = plt.figure(figsize=(12, 6))
        axisList = df_plt.addAxis(fig, 1, 2)
        tt = axisList[0].imshow(
            self.transmittance[:, :],
            extent=(min(lr), max(lr), max(wl), min(wl)),
            vmin=0,
            vmax=1,
        )
        phi = axisList[1].imshow(self.phase[:, :], extent=(min(lr), max(lr), max(wl), min(wl)), cmap="hsv")
        df_plt.formatPlots(
            fig,
            axisList[0],
            tt,
            "len r (nm)",
            "wavelength (nm)",
            "transmission",
            addcolorbar=True,
        )
        df_plt.formatPlots(
            fig,
            axisList[1],
            phi,
            "len r (nm)",
            "wavelength (nm)",
            "phase",
            addcolorbar=True,
        )

        if savepath:
            plt.savefig(savepath + ".png")
            plt.savefig(savepath + ".pdf")
        else:
            plt.show()

        return

    def optical_response_to_param(self, trans_asList, phase_asList, wavelength_asList, reshape=True, fast=False):
        """Computes the shape vector (here, nanocylinder radius) that most closely matches a transmittance and phase profile input. Note that each transmittance and phase profile for a given wavelength
        (in wavelength_aslist) is assumed to be a seperate lens. This is a naive-table look-up function.

        Args:
            trans_asList (float): list of transmittance profiles
            phase_asList (float): list of phase profiles
            wavelength_asList (float): list of wavelengths corresponding to the target transmittance and phase profiles
            reshape (float): Boolean if returned shape vectors are to be given in the same shape as the input or if just return as a flattened list
            fast (bool, optional): Whether to do exhaustive min-search or a less accurate but fast dictionary look-up. The dictionary look-up assumed the target transmittance is unity and finds the best phase match. Defaults to False.

        Returns:
            list: List containing the shape vector for each trans and phase pair passed in (elements of the input list)
        """

        # list assertion
        length = len(wavelength_asList)
        if not all(type(input) is list for input in [trans_asList, phase_asList, wavelength_asList]):
            raise TypeError("optical_response_to_param: trans, phase, and wavelength must all be passed in as lists")

        # list length assertion
        if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
            raise ValueError("optical_response_to_param: All lists must be the same length")

        # polarization dimensionality check
        if not all([trans.shape[0] == 1 for trans in trans_asList]) or not all([phase.shape[0] == 1 for phase in phase_asList]):
            raise ValueError("optical_response_to_param: All transmission/phase profiles in the list must be a single transmission profile, (1, Ny, Nx)")

        ### Assemble metasurfaces
        shape_Vector = []
        shape_Vector_norm = []
        for i in range(length):
            use_wavelength = wavelength_asList[i]
            ms_trans = trans_asList[i]
            ms_phase = phase_asList[i]
            initial_shape = [1, *ms_trans.shape[-2:]]

            if fast:
                design_radius = lookup_D1_pol1(self.name + ".pickle", use_wavelength, ms_trans, ms_phase)
            else:
                design_radius = minsearch_D1_pol1(
                    self.phase,
                    self.transmittance,
                    self.param1.flatten(),
                    self.param2.flatten(),
                    use_wavelength,
                    ms_trans,
                    ms_phase,
                )

            norm_design_radius = np.clip(
                (design_radius - self.__param1Limits[0]) / (self.__param1Limits[1] - self.__param1Limits[0]),
                0,
                1,
            )

            if reshape:
                shape_Vector.append(np.reshape(design_radius, initial_shape))
                shape_Vector_norm.append(np.reshape(norm_design_radius, initial_shape))
            else:
                shape_Vector.append(np.expand_dims(design_radius, -1))
                shape_Vector_norm.append(np.expand_dims(norm_design_radius, -1))

        return shape_Vector, shape_Vector_norm


class Nanoellipse_U350nm_H600nm(Nanofins_U350nm_H600nm):
    def __init__(self):
        super(Nanoellipse_U350nm_H600nm, self).__init__()
        __rawPath = get_path_to_data("data_NanoEllipse_Unit350nm_Height600nm_EngineFDTD.mat")
        data = scipy.io.loadmat(__rawPath)
        self.name = "Nanoellipse_U350nm_H600nm"

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.transmittance = np.sqrt(np.clip(data["transmission"], 0, np.finfo(np.float32).max))

        # the input parameters (lenx, leny, wavelength) are loaded here
        # all parameters must be converted to meshgrid format for compatibility with model class
        self.param1 = data["lenx"]
        self.param2 = data["leny"]
        self.param3 = data["wavelength_m"].flatten()
        self.params = np.meshgrid(self.param1, self.param2, self.param3)

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [60e-9, 300e-9]
        self.__param2Limits = [60e-9, 300e-9]
        self.__param3Limits = [310e-9, 750e-9]
