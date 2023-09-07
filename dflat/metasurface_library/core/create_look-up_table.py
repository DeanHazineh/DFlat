import numpy as np
import pickle as pickle
from pathlib import Path
from tqdm import tqdm

import dflat.metasurface_library as df_library
from util_library_lookup import *


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / "pregen_lookup_tables"
    return resource_path.joinpath(file_name)


def create_d1_pol1_lookup_table(library):
    # Get the library
    phase_table = library.phase
    trans_table = library.transmittance
    p1_table = library.param1
    wl_table = library.param2

    # Wrap Phase to be consistent with the other method
    or_table = trans_table * np.exp(1j * phase_table)
    phase_table = np.angle(or_table)

    # Create dictionary keys (unity transmittance, wl (nm) as int)
    wl_dx = 1
    phase_dx = 0.1
    phase_key = np.round(np.arange(-3.14, 3.14 + phase_dx, phase_dx), 1)
    trans_key = np.round([1.0], 1)
    wl_key = np.arange(np.min(wl_table) * 1e9, np.max(wl_table) * 1e9 + wl_dx, wl_dx).astype(int)
    t, p = np.meshgrid(trans_key, phase_key)

    # Fil dictionary
    lookup_table = {}
    for this_wl in wl_key:
        design_p1 = minsearch_D1_pol1(phase_table, trans_table, p1_table.flatten(), wl_table.flatten(), this_wl, t, p)
        for i in range(len(design_p1)):
            lookup_table[(t[i, 0], p[i, 0], this_wl)] = design_p1[i]

    # Save dictionary
    with open(get_path_to_data(library.name + ".pickle"), "wb") as fhandle:
        pickle.dump(lookup_table, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def create_d2_pol2_lookup_table(library):
    # Get the library
    phase_table = library.phase
    trans_table = library.transmittance

    # Wrap Phase to be consistent with the other method
    or_table = trans_table * np.exp(1j * phase_table)
    phase_table = np.angle(or_table)

    p1_vect = library.params[0][:, :, 0].flatten()
    p2_vect = library.params[1][:, :, 0].flatten()
    wl_vect = library.params[2][0, 0, :]

    # Create dictionary keys (unity transmittance, wl in nm units as int)
    wl_dx = 1
    phase_dx = 0.1
    phasex_key = np.round(np.arange(-3.14, 3.14 + phase_dx, phase_dx), 1)
    phasey_key = np.round(np.arange(-3.14, 3.14 + phase_dx, phase_dx), 1)
    transx_key = np.round([1.0], 1)
    transy_key = np.round([1.0], 1)
    wl_key = np.arange(np.min(wl_vect) * 1e9, np.max(wl_vect) * 1e9 + wl_dx, wl_dx).astype(int)

    tx, ty, px, py = np.meshgrid(transx_key, transy_key, phasex_key, phasey_key)
    target_trans = np.reshape(np.stack((tx, ty)), [2, -1])
    target_phase = np.reshape(np.stack((px, py)), [2, -1])

    # Fill dictionary
    lookup_table = {}
    for this_wl in tqdm(wl_key):
        design_p1, design_p2 = minsearch_D2_pol2(phase_table, trans_table, p1_vect, p2_vect, wl_vect, this_wl, target_trans, target_phase)
        for i in range(len(design_p1)):
            lookup_table[(target_trans[0, i], target_trans[1, i], target_phase[0, i], target_phase[1, i], this_wl)] = (
                design_p1[i],
                design_p2[i],
            )

    # Save dictionary
    with open(get_path_to_data(library.name + ".pickle"), "wb") as fhandle:
        pickle.dump(lookup_table, fhandle, protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":
    # create_d1_pol1_lookup_table(df_library.Nanocylinders_U180nm_H600nm())
    create_d2_pol2_lookup_table(df_library.Nanofins_U350nm_H600nm())
    # create_d2_pol2_lookup_table(df_library.Nanoellipse_U350nm_H600nm())
