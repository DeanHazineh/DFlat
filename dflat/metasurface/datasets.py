import scipy.io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
import os
#import subprocess
import requests

from dflat.plot_utilities.mp_format import format_plot

def download_data(dataset_name, storage_path):
    if not os.path.exists("DFlat"):
        os.mkdir("DFlat/")
    if not os.path.exists("DFlat/Datasets/"):
        os.mkdir("DFlat/Datasets/")

    file_path = os.path.join("DFlat/Datasets/", dataset_name)
    if not os.path.exists(file_path):
        print("Downloading dataset from online storage.")
        with requests.get(storage_path, stream=True) as response:
            response.raise_for_status()  # Raises a HTTPError if the response has an error status code
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192): 
                    file.write(chunk)
            
    return file_path

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
        format_plot(
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
        format_plot(
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

        print(savepath)
        if savepath is not None:
            plt.savefig(os.path.join(savepath, self.__class__.__name__ + ".png"))
            plt.close()


class Nanocylinders_TiO2_U200nm_H600nm(Nanocylinder_base1):
    def __init__(self):
        super().__init__()

        dataset_name = "Nanocylinders_TiO2_Unit200nm_height600nm_FDTD.pickle"
        online_path = "https://www.dropbox.com/scl/fi/48g33fzfmfrzlm6e3fwst/Nanocylinders_TiO2_Unit200nm_height600nm_FDTD.pickle?rlkey=la94xsaro6ulmvw3l6dgz5itb&dl=1"
        file_path = download_data(dataset_name, online_path)
        with open(file_path, "rb") as file:
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
        super().__init__()

        dataset_name = "Nanocylinders_TiO2_U250nm_H600nm.pickle"
        online_path = "https://www.dropbox.com/scl/fi/2yybb9lx1ge3zs5ep7pf2/Nanocylinders_TiO2_Unit250nm_height600nm_FDTD.pickle?rlkey=ioubnrarcnvnr4lwcdyqovtw4&dl=1"
        file_path = download_data(dataset_name, online_path)
        with open(file_path, "rb") as file:
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
        super().__init__()

        dataset_name = "Nanocylinders_TiO2_U300nm_H600nm.pickle"
        online_path = "https://www.dropbox.com/scl/fi/uyltiglfm5bbr8pqmvf2p/Nanocylinders_TiO2_Unit300nm_height600nm_FDTD.pickle?rlkey=jfekg6i0gcqzv0fxxybqomz52&dl=1"
        file_path = download_data(dataset_name, online_path)
        with open(file_path, "rb") as file:
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
        super().__init__()

        dataset_name = "Nanocylinders_TiO2_U350nm_H600nm.pickle"
        online_path = "https://www.dropbox.com/scl/fi/k7z6pg6d55ncaonjdkrkr/Nanocylinders_TiO2_Unit350nm_height600nm_FDTD.pickle?rlkey=h9xfg6w45756d8ds8818yqa1z&dl=1"
        file_path = download_data(dataset_name, online_path)
        with open(file_path, "rb") as file:
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
        super().__init__()

        dataset_name = "Nanocylinders_Si3N4_U250nm_H600nm.pickle"
        online_path = "https://www.dropbox.com/scl/fi/ur45okx30ssetohzf066i/Nanocylinders_Si3N4_Unit250nm_height600nm_FDTD.pickle?rlkey=hpeoe2we0wjsunac7vfo6k2yo&dl=1"
        datpath = download_data(dataset_name, online_path)
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
        super().__init__()

        dataset_name = "Nanocylinders_Si3N4_U300nm_H600nm.pickle"
        online_path = "https://www.dropbox.com/scl/fi/5m2uyqa9lj9h5fesxoy8o/Nanocylinders_Si3N4_Unit300nm_height600nm_FDTD.pickle?rlkey=opb7hoqomofiga9ne46wyw0uv&dl=1"
        datpath = download_data(dataset_name, online_path)
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
        super().__init__()

        dataset_name = "Nanocylinders_Si3N4_U350nm_H600nm.pickle"
        online_path = "https://www.dropbox.com/scl/fi/mfh11d3aowh3t8warp1xc/Nanocylinders_Si3N4_Unit350nm_height600nm_FDTD.pickle?rlkey=o66zjfj2poafyssqmi65bt7n0&dl=1"
        datpath = download_data(dataset_name, online_path)
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
        dataset_name = "Nanofins_TiO2_U350nm_H600nm.mat"
        online_path = "https://www.dropbox.com/scl/fi/l3bitqamxinumq101s49e/Nanofins_TiO2_Unit350nm_Height600nm_FDTD.mat?rlkey=71cas4mcokl44bldmu17xdy72&dl=1"
        datpath = download_data(dataset_name, online_path)

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
            format_plot(
                fig,
                ax[i, 0],
                yvec=lam,
                xvec=lx,
                setAspect="auto",
                ylabel=(
                    "x-pol \n width x (nm)" if i == 0 else "y-pol \n wavelength (nm)"
                ),
                title="transmittance",
                xlabel="wavelength (nm) ",
            )

            ax[i, 1].imshow(trans[i, :, sh[2] // 2, :], vmin=0, vmax=1)
            format_plot(
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
            format_plot(
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
            format_plot(
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
            plt.savefig(os.path.join(savepath, self.__class__.__name__ + ".png"))
            plt.close()

        return


class Nanoellipse_TiO2_U350nm_H600nm(Nanofins_TiO2_U350nm_H600nm):
    def __init__(self):
        super().__init__()

        dataset_name = "Nanoellipse_TiO2_U350nm_H600nm.mat"
        online_path = "https://www.dropbox.com/scl/fi/y3kn6nplwbr689p3up94p/Nanoellipse_TiO2_Unit350nm_Height600nm_FDTD.mat?rlkey=98krxmsfrr0ixiul1934u63qc&dl=1"
        datpath = download_data(dataset_name, online_path)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        data = scipy.io.loadmat(datpath)
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

