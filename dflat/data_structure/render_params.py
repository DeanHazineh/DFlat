from copy import deepcopy
import torch
import numpy as np

ALL_MANDATORY_KEYS = [
    "QE",  # Detection Quantum Efficiency (Photons to electron conversion)
    "dark_offset",  # Dark offset electrons
    "dark_noise_e",  # dark noise in electrons
    "gain",  # Electron to ADU gain conversion
    "shot_noise",  # Boolean of whether to apply shot noise (Poisson) to the rendered signal
    "dark_noise",  # Boolean of wheter to apply dark noise (Gaussian) to the rendered signal
]


ALL_OPTIONAL_KEYS = {
    "ADC": 12,
    "well_depth_e": False,  # This is not used at the moment because we don't want to worry about well saturation
    "name": None,
}


class sensor_params(dict):
    def __init__(self, input_dict, verbose=False):
        self.__dict__ = deepcopy(input_dict)
        self.__check_mandatory_keys()
        self.__check_optional_keys()
        self.__check_unknown_keys()

    def __check_mandatory_keys(self):
        # Verify all mandatory keys are included
        for exclusive_key in ALL_MANDATORY_KEYS:
            if isinstance(exclusive_key, list):
                if not any(check_key in self.__dict__.keys() for check_key in exclusive_key):
                    print(exclusive_key)
                    raise KeyError("\n params: one of the above keys must be included")
            else:
                if not (exclusive_key in self.__dict__.keys()):
                    raise KeyError("params: Missing mandatory parameter option for simulation settings: " + exclusive_key)

        return

    def __check_optional_keys(self):
        for optional_key in ALL_OPTIONAL_KEYS.keys():
            # If an optional key was not provided, add and assign to default optional value
            if not (optional_key in self.__dict__.keys()):
                self.__dict__[optional_key] = ALL_OPTIONAL_KEYS[optional_key]

        return

    def __check_unknown_keys(self):
        # Check unknown keys against all possible keys
        for providedKey in self.__dict__.keys():
            if not providedKey in (ALL_MANDATORY_KEYS + list(ALL_OPTIONAL_KEYS.keys())):
                raise KeyError("params: unknown parameter key/setting inputed: " + providedKey)

        return

    def SNR_to_meanPhotons(self, SNR):
        if self.__dict__["shot_noise"] == False and self.self.__dict__["dark_noise"] == False:
            raise ValueError("You cannot request SNR scaling whan all noise parameters are set to False!")

        QE = self.__dict__["QE"]
        sig_dark = self.__dict__["dark_noise_e"]
        ADC = self.__dict__["ADC"]
        gain = self.__dict__["gain"]
        return SNR**2 / 2 / QE * (1 + np.sqrt(1 + 4 * (sig_dark**2 + 1 / ADC / gain**2) / SNR**2))

    ###
    def __setitem__(self, key, item):
        if key in self.__dict__.keys():
            # no change on the items after initialization shall be allowed
            raise "The params cannot be changed after initialization"
        else:
            # allow adding new keys
            self.__dict__[key] = item
        return item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def get_dict(self):
        return deepcopy(self.__dict__)

    def has_key(self, key_name):
        if key_name in self.__dict__.keys():
            return True
        else:
            return False
