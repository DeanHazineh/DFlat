from pathlib import Path
from scipy import interpolate
import numpy as np
import pandas as pd


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / ""
    return resource_path.joinpath(file_name)


def get_QE_SONY_Pregius(channels_nm):
    band_lims = [300, 900]
    if min(channels_nm) < band_lims[0] or max(channels_nm) > band_lims[1]:
        raise ValueError("Input channel range for sensor QE is out of data file range")

    csv = get_path_to_data("QE_Estimates/QE_CMOS_Pregius.csv")
    column_labels = [
        "Sony_CMOS_IMX_Pregius_Gen1",
        "Sony_CMOS_IMX_Pregius_Gen2",
        "Sony_CMOS_IMX_Pregius_Gen3",
        "Sony_CMOS_IMX_Pregius_Gen4",
    ]
    delimiter = " "

    df = pd.read_csv(csv, delimiter=delimiter)
    sensor_QE = df[column_labels].to_numpy()
    bands = df["wavelength"].to_numpy()
    holdQE = []
    for sensor in range(sensor_QE.shape[1]):
        interp_fun = interpolate.interp1d(bands, sensor_QE[:, sensor], kind="linear")
        holdQE.append(interp_fun(channels_nm))

    return np.transpose(np.stack(holdQE)), column_labels


def get_QETrans_Basler_Bayer(channels_nm):
    band_lims = [360, 750]
    if min(channels_nm) < band_lims[0] or max(channels_nm) > band_lims[1]:
        raise ValueError("Input channel range for sensor QE is out of data file range")

    csv = get_path_to_data("QE_Estimates/RGB_Basler_Ace2_QE.csv")
    column_labels = ["R", "G1", "G2", "B"]
    delimiter = ","

    df = pd.read_csv(csv, delimiter=delimiter)
    sensor_QE = df[column_labels].to_numpy()
    bands = df["Wavelength[nm]"].to_numpy()
    holdQE = []
    for sensor in range(sensor_QE.shape[1]):
        interp_fun = interpolate.interp1d(bands, sensor_QE[:, sensor], kind="linear")
        holdQE.append(interp_fun(channels_nm))

    return np.transpose(np.stack(holdQE)), column_labels
