from pathlib import Path
import h5py
import numpy as np
from scipy.interpolate import interp1d

MATERIAL_DICT = {
    "TiO2": "TiO2_Index.mat",
    "SiO2": "SiO2_Index.mat",
    "Vacuum": None,
    "Si": "Si_Index.mat",
    "Si3N4": "Si3N4_Index.mat",
}


def list_materials():
    return list(MATERIAL_DICT.keys())


def get_material_index(material_name, wavelength_list):
    # Scipy interp1d function allows for complex numbers

    if material_name == "Vacuum":
        return (1 + 1j * 0) * np.ones(shape=(len(wavelength_list)))
    else:
        resource_path = Path(__file__).parent / "material_index"
        index_path = resource_path.joinpath(MATERIAL_DICT[material_name])

        data = {}
        f = h5py.File(index_path)
        for k, v in f.items():
            data[k] = np.array(v)

        # Check sign of +/- imag
        index_dat = data["index"]
        index_dat = np.squeeze(index_dat["real"] - 1j * index_dat["imag"])
        wavelength_dat = np.squeeze(data["w"])

        if (np.min(wavelength_list) < np.min(wavelength_dat)) or (
            np.max(wavelength_list) > np.max(wavelength_dat)
        ):
            raise ValueError(
                "get_material_index: wavelength is outside the boundaries of the index dat file"
            )
        else:
            interp_func = interp1d(wavelength_dat, index_dat)

    return interp_func(wavelength_list)
