import scipy.io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from dflat.plot_utilities.mp_format import formatPlot


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / "data/"
    return resource_path.joinpath(file_name)


###
class Nanocylinder_base1(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def plot(self, savepath=None):
        phase = self.phase
        trans = self.trans
        r = self.params[0] * 1e9
        lam = self.params[1] * 1e9

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(trans)
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
        ax[1].imshow(phase, cmap="hsv")
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

        if savepath is not None:
            plt.savefig(savepath + self.__class__.__name__ + ".png")
            plt.close()


class Nanocylinders_TiO2_U200nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        datpath = get_path_to_data(
            "Nanocylinders_TiO2_Unit200nm_height600nm_FDTD.pickle"
        )
        with open(datpath, "rb") as file:
            data = pickle.load(file)
            rvec = data["rvec"]
            lam = data["lam"]
            phase = data["phase"]
            trans = data["trans"]

        # This data has a shape of [lenr, wavelength=441]
        self.phase = np.angle(np.exp(1j * phase))
        self.trans = np.sqrt(np.clip(data["trans"], 0, np.finfo(np.float32).max))
        self.params = [rvec, lam]
        # self.param_limits = [[rvec.min(), rvec.max()], [lam.min(), lam.max()]]
        self.param_limits = [[15e-9, 85e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for (p, l) in zip(self.params, self.param_limits)
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans.flatten(),
                (np.cos(phase.flatten()) + 1) / 2,
                (np.sin(phase.flatten()) + 1) / 2,
            ],
            -1,
        )


class Nanocylinders_TiO2_U250nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        datpath = get_path_to_data(
            "Nanocylinders_TiO2_Unit250nm_height600nm_FDTD.pickle"
        )
        with open(datpath, "rb") as file:
            data = pickle.load(file)
            rvec = data["rvec"]
            lam = data["lam"]
            phase = data["phase"]
            trans = data["trans"]

        # This data has a shape of [lenr, wavelength=441]
        self.phase = np.angle(np.exp(1j * phase))
        self.trans = np.sqrt(np.clip(data["trans"], 0, np.finfo(np.float32).max))
        self.params = [rvec, lam]
        # self.param_limits = [[rvec.min(), rvec.max()], [lam.min(), lam.max()]]
        self.param_limits = [[15e-9, 110e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for (p, l) in zip(self.params, self.param_limits)
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans.flatten(),
                (np.cos(phase.flatten()) + 1) / 2,
                (np.sin(phase.flatten()) + 1) / 2,
            ],
            -1,
        )


class Nanocylinders_TiO2_U300nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        datpath = get_path_to_data(
            "Nanocylinders_TiO2_Unit300nm_height600nm_FDTD.pickle"
        )

        with open(datpath, "rb") as file:
            # Load the data from the file
            data = pickle.load(file)
            rvec = data["rvec"]
            lam = data["lam"]
            phase = data["phase"]
            trans = data["trans"]

        # This data has a shape of [lenr=121, wavelength=441]
        self.phase = np.angle(np.exp(1j * phase))
        self.trans = np.sqrt(np.clip(data["trans"], 0, np.finfo(np.float32).max))
        self.params = [rvec, lam]
        # self.param_limits = [[rvec.min(), rvec.max()], [lam.min(), lam.max()]]
        self.param_limits = [[15e-9, 135e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for (p, l) in zip(self.params, self.param_limits)
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans.flatten(),
                (np.cos(phase.flatten()) + 1) / 2,
                (np.sin(phase.flatten()) + 1) / 2,
            ],
            -1,
        )


class Nanocylinders_TiO2_U350nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        datpath = get_path_to_data(
            "Nanocylinders_TiO2_Unit350nm_height600nm_FDTD.pickle"
        )

        with open(datpath, "rb") as file:
            # Load the data from the file
            data = pickle.load(file)
            rvec = data["rvec"]
            lam = data["lam"]
            phase = data["phase"]
            trans = data["trans"]

        # This data has a shape of [lenr=121, wavelength=441]
        self.phase = np.angle(np.exp(1j * phase))
        self.trans = np.sqrt(np.clip(data["trans"], 0, np.finfo(np.float32).max))
        self.params = [rvec, lam]
        # self.param_limits = [[rvec.min(), rvec.max()], [lam.min(), lam.max()]]
        self.param_limits = [[15e-9, 160e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for (p, l) in zip(self.params, self.param_limits)
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans.flatten(),
                (np.cos(phase.flatten()) + 1) / 2,
                (np.sin(phase.flatten()) + 1) / 2,
            ],
            -1,
        )


###
class Nanocylinders_Si3N4_U250nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        datpath = get_path_to_data(
            "Nanocylinders_Si3N4_Unit250nm_height600nm_FDTD.pickle"
        )
        with open(datpath, "rb") as file:
            data = pickle.load(file)
            rvec = data["rvec"]
            lam = data["lam"]
            phase = data["phase"]
            trans = data["trans"]

        # This data has a shape of [lenr, wavelength=441]
        self.phase = np.angle(np.exp(1j * phase))
        self.trans = np.sqrt(np.clip(data["trans"], 0, np.finfo(np.float32).max))
        self.params = [rvec, lam]
        self.param_limits = [[15e-9, 109e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for (p, l) in zip(self.params, self.param_limits)
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans.flatten(),
                (np.cos(phase.flatten()) + 1) / 2,
                (np.sin(phase.flatten()) + 1) / 2,
            ],
            -1,
        )


class Nanocylinders_Si3N4_U300nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        datpath = get_path_to_data(
            "Nanocylinders_Si3N4_Unit300nm_height600nm_FDTD.pickle"
        )
        with open(datpath, "rb") as file:
            data = pickle.load(file)
            rvec = data["rvec"]
            lam = data["lam"]
            phase = data["phase"]
            trans = data["trans"]

        # This data has a shape of [lenr, wavelength=441]
        self.phase = np.angle(np.exp(1j * phase))
        self.trans = np.sqrt(np.clip(data["trans"], 0, np.finfo(np.float32).max))
        self.params = [rvec, lam]
        self.param_limits = [[15e-9, 135e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for (p, l) in zip(self.params, self.param_limits)
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans.flatten(),
                (np.cos(phase.flatten()) + 1) / 2,
                (np.sin(phase.flatten()) + 1) / 2,
            ],
            -1,
        )


class Nanocylinders_Si3N4_U350nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        datpath = get_path_to_data(
            "Nanocylinders_Si3N4_Unit350nm_height600nm_FDTD.pickle"
        )
        with open(datpath, "rb") as file:
            data = pickle.load(file)
            rvec = data["rvec"]
            lam = data["lam"]
            phase = data["phase"]
            trans = data["trans"]

        # This data has a shape of [lenr, wavelength=441]
        self.phase = np.angle(np.exp(1j * phase))
        self.trans = np.sqrt(np.clip(data["trans"], 0, np.finfo(np.float32).max))
        self.params = [rvec, lam]
        self.param_limits = [[15e-9, 159e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for (p, l) in zip(self.params, self.param_limits)
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans.flatten(),
                (np.cos(phase.flatten()) + 1) / 2,
                (np.sin(phase.flatten()) + 1) / 2,
            ],
            -1,
        )


###
class Nanofins_TiO2_U350nm_H600nm(Dataset):
    def __init__(self):
        datpath = get_path_to_data("Nanofins_TiO2_Unit350nm_Height600nm_FDTD.mat")

        # Raw phase and transmittance data has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        data = scipy.io.loadmat(datpath)
        self.phase = data["phase"]
        self.trans = np.sqrt(np.clip(data["transmission"], 0, np.finfo(np.float32).max))
        self.params = [data["leny"], data["lenx"], data["wavelength_m"].flatten()]
        self.param_limits = [[60e-9, 300e-9], [60e-9, 300e-9], [310e-9, 750e-9]]

        # Transform the data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for _, (p, l) in enumerate(zip(self.params, self.param_limits))
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans[0].flatten(),
                (np.cos(phase[0].flatten()) + 1) / 2,
                (np.sin(phase[0].flatten()) + 1) / 2,
                trans[1].flatten(),
                (np.cos(phase[1].flatten()) + 1) / 2,
                (np.sin(phase[1].flatten()) + 1) / 2,
            ],
            -1,
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def plot(self, savepath=None):
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

        if savepath is not None:
            plt.savefig(savepath + self.__class__.__name__ + ".png")
            plt.close()

        return


class Nanoellipse_TiO2_U350nm_H600nm(Nanofins_TiO2_U350nm_H600nm):
    def __init__(self):
        super().__init__()
        datapath = get_path_to_data("Nanoellipse_TiO2_Unit350nm_Height600nm_FDTD.mat")
        data = scipy.io.loadmat(datapath)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.trans = np.sqrt(np.clip(data["transmission"], 0, np.finfo(np.float32).max))
        self.params = [data["lenx"], data["leny"], data["wavelength_m"].flatten()]
        self.param_limits = [[60e-9, 300e-9], [60e-9, 300e-9], [310e-9, 750e-9]]

        # Transform the data into a cell-level dataset ([0, 1])
        trans = self.trans
        phase = self.phase
        params = np.meshgrid(
            *[
                (p - l[0]) / (l[1] - l[0])
                for _, (p, l) in enumerate(zip(self.params, self.param_limits))
            ],
            indexing="ij",
        )
        self.x = np.stack([p.flatten() for p in params], -1)
        self.y = np.stack(
            [
                trans[0].flatten(),
                (np.cos(phase[0].flatten()) + 1) / 2,
                (np.sin(phase[0].flatten()) + 1) / 2,
                trans[1].flatten(),
                (np.cos(phase[1].flatten()) + 1) / 2,
                (np.sin(phase[1].flatten()) + 1) / 2,
            ],
            -1,
        )
