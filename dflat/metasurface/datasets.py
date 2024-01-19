import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset

from dflat.plot_utilities.mp_format import formatPlot


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / "data/"
    return resource_path.joinpath(file_name)


class Nanofins_TiO2_U350nm_H600nm(Dataset):
    def __init__(self):
        datpath = get_path_to_data("Nanofins_TiO2_Unit350nm_Height600nm_FDTD.mat")

        # Raw phase and transmittance data has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        data = scipy.io.loadmat(datpath)
        self.phase = np.angle(np.exp(1j * data["phase"])) + np.pi
        self.trans = np.sqrt(data["transmission"])
        self.params = [data["leny"], data["lenx"], data["wavelength_m"].flatten()]
        self.param_limits = [[60e-9, 300e-9], [60e-9, 300e-9], [310e-9, 750e-9]]

        # Transform the data into a cell-level dataset ([0, 1])
        trans = np.clip(self.trans, 0, np.finfo(np.float32).max)
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
                np.cos(phase[0].flatten()),
                np.sin(phase[0].flatten()),
                trans[1].flatten(),
                np.cos(phase[1].flatten()),
                np.sin(phase[1].flatten()),
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


class Nanocylinders_TiO2_U180nm_H600nm(Dataset):
    def __init__(self):
        datpath = get_path_to_data("Nanocylinders_TiO2_Unit180nm_Height600nm_FDTD.mat")

        # Phase and transmission has shape [wavelength=441, lenr=191]
        data = scipy.io.loadmat(datpath)
        self.phase = np.angle(np.exp(1j * data["phase"])) + np.pi
        self.trans = np.sqrt(data["transmission"])
        self.params = [data["radius_m"], data["wavelength_m"].flatten()]
        self.param_limits = [[30e-9, 150e-9], [310e-9, 750e-9]]

        # Transform data into a cell-level dataset ([0, 1])
        trans = np.clip(self.trans, 0, np.finfo(np.float32).max)
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
                trans.flatten(),
                np.cos(phase.flatten()),
                np.sin(phase.flatten()),
            ],
            -1,
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def plot(self, savepath=None):
        phase = self.phase
        trans = self.transmittance
        r = self.params[0] * 1e9
        lam = self.params[1] * 1e9

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

        if savepath is not None:
            plt.savefig(savepath + self.__class__.__name__ + ".png")
            plt.close()


class Nanoellipse_TiO2_U350nm_H600nm(Nanofins_TiO2_U350nm_H600nm):
    def __init__(self):
        super().__init__()
        datapath = get_path_to_data("Nanoellipse_TiO2_Unit350nm_Height600nm_FDTD.mat")
        data = scipy.io.loadmat(datapath)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = np.angle(np.exp(1j * data["phase"])) + np.pi
        self.transmittance = np.sqrt(data["transmission"])
        self.params = [data["lenx"], data["leny"], data["wavelength_m"].flatten()]
        self.param_limits = [[60e-9, 300e-9], [60e-9, 300e-9], [310e-9, 750e-9]]

        # Transform the data into a cell-level dataset ([0, 1])
        trans = np.clip(self.trans, 0, np.finfo(np.float32).max)
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
                np.cos(phase[0].flatten()),
                np.sin(phase[0].flatten()),
                trans[1].flatten(),
                np.cos(phase[1].flatten()),
                np.sin(phase[1].flatten()),
            ],
            -1,
        )
