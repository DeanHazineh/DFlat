import numpy as np
import pickle
from pathlib import Path


def get_path_to_data(file_name: str):
    ## This is not the accepted solution but it should work for bootstrapping research with few users
    resource_path = Path(__file__).parent / "pregen_lookup_tables"
    return resource_path.joinpath(file_name)


def minsearch_D2_pol2(
    phaseTable,
    transTable,
    p1_vect,
    p2_vect,
    wavelength,
    use_wavelength,
    ms_trans,
    ms_phase,
):
    # Find the sub-table matching the wavelength requested
    _, w_idx = min((val, idx) for (idx, val) in enumerate(np.abs(use_wavelength - wavelength)))
    sublib_phase = phaseTable[:, :, :, w_idx]
    sublib_trans = transTable[:, :, :, w_idx]
    or_table = sublib_trans * np.exp(1j * sublib_phase)
    or_table = np.reshape(or_table, [2, -1])

    # Get the two polarization profiles for each metasurface
    target_profile = np.reshape(ms_trans, [2, -1]) * np.exp(1j * np.reshape(ms_phase, [2, -1]))

    # Brute-force look-up for the metasurface
    design_p1 = []
    design_p2 = []
    for cell in range(target_profile.shape[1]):
        _, param_idx = min(
            (val, idx)
            for (idx, val) in enumerate(0.5 * np.abs(target_profile[0, cell] - or_table[0, :]) + 0.5 * np.abs(target_profile[1, cell] - or_table[1, :]))
        )
        design_p1.append(p1_vect[param_idx])
        design_p2.append(p2_vect[param_idx])

    # Define a normalized shape vector for convenience
    design_p1 = np.array(design_p1)
    design_p2 = np.array(design_p2)

    return design_p1, design_p2


def minsearch_D1_pol1(phaseTable, transTable, p1_vect, wavelength, use_wavelength, ms_trans, ms_phase):
    # Find the sub-table matching the wavelength requested
    _, w_idx = min((val, idx) for (idx, val) in enumerate(np.abs(use_wavelength - wavelength)))
    sublib_phase = phaseTable[w_idx, :]
    sublib_trans = transTable[w_idx, :]
    or_table = sublib_trans * np.exp(1j * sublib_phase)

    # Get the target profile
    target_profile = ms_trans.flatten() * np.exp(1j * ms_phase.flatten())

    # minimize via look-up table
    design_p1 = []
    for cell in range(target_profile.shape[0]):
        _, param_idx = min((val, idx) for (idx, val) in enumerate(np.abs(target_profile[cell] - or_table[:])))
        design_p1.append(p1_vect[param_idx])

    return np.array(design_p1)


def lookup_D1_pol1(dict_name, use_wavelength, ms_trans, ms_phase):
    # Get the look-up table dictionary
    with open(get_path_to_data(dict_name), "rb") as fhandle:
        lookup_table = pickle.load(fhandle)

    # Wrap the phase to be consistent with the table
    ms_phase = np.angle(np.exp(1j * ms_phase))
    ms_phase = np.round(ms_phase.flatten(), 1)

    design_p1 = []
    use_wavelength = int(use_wavelength * 1e9)
    for cell in range(ms_phase.shape[0]):
        design_p1.append(lookup_table[(1.0, ms_phase[cell], use_wavelength)])

    return np.array(design_p1)


def lookup_D2_pol2(dict_name, use_wavelength, ms_trans, ms_phase):
    # Get the look-up table dictionary
    with open(get_path_to_data(dict_name), "rb") as fhandle:
        lookup_table = pickle.load(fhandle)

    # Wrap the phase to be consistent with the table
    init_shape = ms_phase.shape
    ms_phase = np.angle(np.exp(1j * ms_phase))
    ms_phase = np.round(np.reshape(ms_phase, [init_shape[0], -1]), 1)

    use_wavelength = int(use_wavelength * 1e9)
    design_p = []
    for cell in range(ms_phase.shape[1]):
        design_p.append(lookup_table[(1.0, 1.0, ms_phase[0, cell], ms_phase[1, cell], use_wavelength)])
    design_p = np.stack(design_p)

    return design_p[:, 0], design_p[:, 1]
