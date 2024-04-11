from scipy import interpolate
from pathlib import Path
import numpy as np
import torch


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / ""
    return resource_path.joinpath(file_name)


def get_xyz_bar_CIE1931(channels_nm):
    xyzpath = get_path_to_data("color_space/1931XYZFunctions.txt")
    xyzCIE = np.loadtxt(xyzpath, skiprows=1)

    xbar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 1], kind="linear")
    ybar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 2], kind="linear")
    zbar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 3], kind="linear")

    return np.transpose(
        np.stack([xbar_(channels_nm), ybar_(channels_nm), zbar_(channels_nm)], axis=0)
    )


def get_rgb_bar_CIE1931(channels_nm):
    channels_nm = np.squeeze(channels_nm)
    rgbpath = get_path_to_data("color_space/1931RGBFunctions.txt")
    rgbCIE = np.loadtxt(rgbpath, skiprows=1)

    rbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 1], kind="linear")
    gbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 2], kind="linear")
    bbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 3], kind="linear")

    return np.transpose(
        np.stack([rbar_(channels_nm), gbar_(channels_nm), bbar_(channels_nm)], axis=0)
    )


def get_illuminant_6500(channels_nm):
    channels_nm = channels_nm.flatten()
    illum_path = get_path_to_data("illuminants/D65.txt")
    illum = np.loadtxt(illum_path, skiprows=1)

    illum_ = interpolate.interp1d(illum[:, 0], illum[:, 1], kind="linear")

    return illum_(channels_nm)


def gamma_correction(sRGB):
    gamma_map = sRGB > 0.0031308
    corrected_high = 1.055 * torch.pow(sRGB, 1.0 / 2.4) - 0.055
    corrected_low = 12.92 * sRGB
    sRGB_corrected = torch.where(gamma_map, corrected_high, corrected_low)

    # Clipping values between 0 and 1
    sRGB_clamped = torch.clamp(sRGB_corrected, 0, 1)

    return sRGB_clamped
