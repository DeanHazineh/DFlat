from copy import deepcopy
import numpy as np
import math
import torch

from .core.fourier_param_utils import *

## Internal Notes: To customize the behavior of a dictionary, It is better to subclass collections.UserDict instead of the built-in dict.
ALL_MANDATORY_KEYS = [
    ["wavelength_m", "wavelength_set_m"],  # Simulation wavelengths
    "ms_samplesM",  # number of samples in the input field, as dictionary {"x": Mx, "y": My}
    "ms_dx_m",  # Initial Field/DOE discretization (m)
    "sensor_distance_m",  # Distance from first plane to output plane (normal to input field)
    "initial_sensor_dx_m",  # Requested output/sensor grid discretization for field calculations
    "sensor_pixel_size_m",  # Requested output/sensor "pixel size" to downsample field by area integration to (if different)
    "sensor_pixel_number",  # Number of output/sensor "pixels" along x and y
    "radial_symmetry",  # Flag -whether the input has radial symmetry and is given by a radial vector
    "diffractionEngine",  # Allow selection of Fresnel or ASM diffraction engine
]

diffractionEngines = [
    "fresnel_fourier",
    "ASM_fourier",
]

ALL_OPTIONAL_KEYS = {
    "manual_upsample_factor": 1,  # Factor to upsample the input fields when conducting calculations
    "automatic_upsample": False,  # Flag to indicate if DFlat should estimate and apply its own upsample factor
    "radius_m": None,  # Radius of a circular field aperture to consider against the square input field array
    "dtype": torch.float32,
    "mixed_dtype": True,  # To improve accuracy, we can mix float32 and float64 precision in various parts of the code
    "ASM_Pad_opt": 1,  # Development flag to control field padding for ASM Field transformations
}

HIDDEN_KEYS = [
    "_prop_params__verbose",
]

## These keys get automatically added to the dictionary after initialization and processing of the user-provided inputs
ADDED_KEYS = [
    "wavelength_um",
    "wavelength_set_um",
    "ms_dx_um",
    "calc_ms_dx_um",
    "sensor_distance_um",
    "initial_sensor_dx_um",
    "calc_sensor_dx_um",
    "sensor_pixel_size_um",
    "ms_length_um",
    "radius_um",
    "padms_half",
    "calc_samplesM",
    "calc_samplesN",
    "ratio_pixel_to_grid",
    "broadband_flag",
    "grid_shape",
]

ADDED_TENSOR_KEYS = [
    "_input_pixel_x",
    "_input_pixel_y",
    "_output_pixel_x",
    "_output_pixel_y",
    "_torch_zero",
    "_ASM_x",
    "_ASM_y",
    "_ang_fx",
    "_normterm",
]


def print_full_settings(parameters):
    if not "wavelength_set_m" in parameters:
        parameters._print_info()
    else:
        wavelength_set_m = parameters["wavelength_set_m"]
        for wavelength in wavelength_set_m:
            setting_dict = parameters.get_dict()
            del setting_dict["wavelength_set_m"]
            setting_dict["wavelength_m"] = wavelength
            prop_params(setting_dict, verbose=True)

    return


class prop_params(dict):
    """Parameters object (dictionary) used for the propagation in the Fourier layers. Defines the simulation settings."""

    def __init__(self, input_dict, verbose=False):
        """Parameters object (dictionary) used for the propagation in the Fourier layers. Defines the simulation settings.

        Args:
            `input_dict` (dict): Input dictionary with keys listed below.
            `verbose` (bool, optional): Boolean flag to print simulation settings info after initialization. Defaults
                to False.

            A description of keys required/used for the input_dict:
                `"wavelength_m"` or `"wavelength_set_m"`: Simulation wavelength(s) in units of m, as a float or a list
                    of floats in the case of the latter.
                `"ms_length_m"`: Dict containing the metasurface length along x and y, via {"x": float, "y": float}.
                `"ms_dx_m"`: Dict containing the metasurface grid discretization along x and y,
                    via {"x": float, "y": float}.
                `"sensor_distance_m"`: Float constant providing the distance between the metasurface and the sensor
                    plane (or input plane to output plane for propagation more generally).
                `"initial_sensor_dx_m"`: Dict containing the sensor plane grid discretization along x and y,
                     via {"x": float, "y":float}.
                `"sensor_pixel_size_m"`: Dict containing the pixel size of a detector placed at the sensor plane,
                    via {"x": float, "y":float}.
                `"sensor_pixel_number"`: Dict containing the number of detector pixels at the sensor plane,
                    via {"x": int, "y": int}.
                `"radial_symmetry": Boolean flag indicating if input data is of radial vector form or 2D cartesian.
                `"diffractionEngine": String selection for the diffraction engine to use in propagation.
                    See documentation for valid options.
                (optional) `"radius_m"`: float indicating the radius of a circular field aperture to be placed at the
                     metasurface/input plane. Defaults to None.
                (optional) `"dtype"`: tf.dtype to be used during all calculations. Defaults to tf.float64 and should
                     not be changed in the current version!
        """

        self.__dict__ = deepcopy(input_dict)
        self.__check_mandatory_keys()
        self.__check_wavelength_key()
        self.__check_optional_keys()
        self.__check_unknown_keys()
        self.__check_diffractionEngine_selection()
        self.__check_number_samples_int()
        self.__check_radial_symmetry()
        self.__check_detector_pixel_size()

        self.__unit_conversion()  # Fourier calculations are switched in newest version from m base units to micrometer
        self.__add_implied_keys()
        self.__verbose = verbose

        if self.__dict__["radial_symmetry"]:
            self.__enforce_odd_sensor_pixel_number()

        # If wavelength_m is given, this is a usable prop_instance keys
        # so generate the additional keys required for function calls.
        # If wavelength_set_m is given instead, this is a parent object and additional keys will be added
        # The reason is two-fold:
        # - First, we allow users to use the Fresnel FFT method and get a user-specified output grid that is the same for each wavelength!
        # - This output grid requires zero-padding the initial field by an amount dependent on wavelength.
        # - Second, we have implemented an optional approach to determine automatically an estimate of upsampling
        # - which is dependent on the simulation wavelength!
        if "wavelength_m" in self.__dict__.keys():
            self.__regularizeInputOutputSpace()

            # Add the tensors that we will use in the calculations to this parameters object.
            # That way when they tell us a device, we can move all the intermediate tensors to the device of choice
            # and pass them around without having to reinitialize the tensors in other places
            self.__add_tensors()

        return

    def __check_mandatory_keys(self):
        for exclusive_key in ALL_MANDATORY_KEYS:
            if isinstance(exclusive_key, list):
                if not any(check_key in self.__dict__.keys() for check_key in exclusive_key):
                    print(exclusive_key)
                    raise KeyError("\n params: one of the above keys must be included")
            else:
                if not (exclusive_key in self.__dict__.keys()):
                    raise KeyError("params: Missing mandatory parameter option for simulation settings: " + exclusive_key)

        return

    def __check_wavelength_key(self):
        if ("wavelength_set_m" in self.__dict__.keys()) and ("wavelength_m" in self.__dict__.keys()):
            raise KeyError("params: wavelength_m and wavelength_set_m cannot both be given. Choose one or the other.")

        if "wavelength_m" in self.__dict__.keys():
            self.__dict__["broadband_flag"] = False

        if "wavelength_set_m" in self.__dict__.keys():
            self.__dict__["broadband_flag"] = True

            if not isinstance(self.__dict__["wavelength_set_m"], (list, np.ndarray)):
                raise ValueError("params: wavelength_set_m must be either a list or a numpy array")

        return

    def __check_optional_keys(self):
        for optional_key in ALL_OPTIONAL_KEYS.keys():
            if not (optional_key in self.__dict__.keys()):
                self.__dict__[optional_key] = ALL_OPTIONAL_KEYS[optional_key]

        return

    def __check_unknown_keys(self):
        # Flatten the mandatory key list given nested optionals
        flattened_all_mandatory_keys = []
        for item in ALL_MANDATORY_KEYS:
            if isinstance(item, list):
                [flattened_all_mandatory_keys.append(subitem) for subitem in item]
            else:
                flattened_all_mandatory_keys.append(item)

        # Check unknown keys against all possible keys
        for providedKey in self.__dict__.keys():
            if not providedKey in (flattened_all_mandatory_keys + list(ALL_OPTIONAL_KEYS.keys()) + HIDDEN_KEYS + ADDED_KEYS + ADDED_TENSOR_KEYS):
                raise KeyError("params: unknown parameter key/setting inputed: " + providedKey)

        return

    def __check_diffractionEngine_selection(self):
        if not (self.__dict__["diffractionEngine"] in diffractionEngines):
            raise ValueError("diffractionEngine selection invalid: must be either 'fresnel_fourier' or 'ASM_fourier'.")

        return

    def __check_number_samples_int(self):
        # To be safe and not have to cast later, lets ensure that ms_samplesM is passed in as an integer dictionary
        ms_samplesM = self.__dict__["ms_samplesM"]
        if not (isinstance(ms_samplesM["x"], int) and isinstance(ms_samplesM["y"], int)):
            print("Warning on params initialization: ms_samplesM should be an int. Int casting used")
            self.__dict__["ms_samplesM"] = {"x": int(ms_samplesM["x"]), "y": int(ms_samplesM["y"])}

        return

    def __check_radial_symmetry(self):
        # If radial_symmetry flag is activated, ensure initial lens space is square and uniformly sampled
        # also, ms_samplesM must have an odd number of samples otherwise radial vector is poorly defined at zero
        if self.__dict__["radial_symmetry"]:
            ms_samplesM = self.__dict__["ms_samplesM"]
            ms_dx_m = self.__dict__["ms_dx_m"]

            if not (ms_samplesM["x"] == ms_samplesM["y"]):
                raise ValueError("params: initial_lens_length_m must be square when radial_symmetry flag is used")

            if not (ms_dx_m["x"] == ms_dx_m["y"]):
                raise ValueError("params: iniital_lens_dx_m must be same along x and y when radial_symmetry flag is used")

            if np.mod(ms_samplesM["x"], 2) == 0:
                raise ValueError("params: ms_samplesM must have an odd number of samples to use radial flag")

        return

    def __check_detector_pixel_size(self):
        # The initial sensor-side calculation grid should always be equal to or smaller than the sensor "pixel" size
        initial_sensor_dx_m = self.__dict__["initial_sensor_dx_m"]
        sensor_pixel_size_m = self.__dict__["sensor_pixel_size_m"]
        if (initial_sensor_dx_m["x"] > sensor_pixel_size_m["x"]) or (initial_sensor_dx_m["y"] > sensor_pixel_size_m["y"]):
            raise ValueError(
                "params: requested output/sensor plane field grid cannot be discretized larger than the requested, resampled output/detector 'pixel' size!"
            )

        return

    def __unit_conversion(self):
        # Internal Conversion of all units to micrometer instead of meters for numerical stability
        convert_keys = ["wavelength_m", "wavelength_set_m", "ms_dx_m", "sensor_distance_m", "initial_sensor_dx_m", "sensor_pixel_size_m", "radius_m"]
        for key in convert_keys:
            if key in self.__dict__.keys():
                dict_item = self.__dict__[key]
                new_name = key[:-1] + "um"

                # Add the rescaled version
                if isinstance(dict_item, list):
                    self.__dict__[new_name] = np.array(dict_item) * 10**6
                elif isinstance(dict_item, dict):
                    self.__dict__[new_name] = {"x": dict_item["x"] * 10**6, "y": dict_item["y"] * 10**6}
                elif isinstance(dict_item, float) or isinstance(dict_item, np.ndarray):
                    self.__dict__[new_name] = dict_item * 10**6
                elif dict_item is None:
                    self.__dict__[new_name] = None
                else:
                    raise ValueError("In key conversion, ran into uknown datatype")

        return

    ### Add useful Keys for downstream calculations
    def __add_implied_keys(self):
        # add key for the length of the user-passed metasurface
        ms_dx_um = self.__dict__["ms_dx_um"]
        ms_samplesM = self.__dict__["ms_samplesM"]
        ms_length_um = {"x": ms_dx_um["x"] * ms_samplesM["x"], "y": ms_dx_um["y"] * ms_samplesM["y"]}
        self.__dict__["ms_length_um"] = ms_length_um

        # Add a radial length into the ms_samplesM key
        # Note: Check_radial_symmetry call has already ensured that ms_samplesM is has odd integers
        ms_samplesM_r = int((ms_samplesM["x"] - 1) / 2 + 1)
        self.__dict__["ms_samplesM"] = {"x": ms_samplesM["x"], "y": ms_samplesM["y"], "r": ms_samplesM_r}

        # It is convenient to add the lens shapes to a grid_shape parameter
        grid_shape = [1, 1, ms_samplesM_r] if self.__dict__["radial_symmetry"] else [1, ms_samplesM["y"], ms_samplesM["x"]]
        self.__dict__["grid_shape"] = grid_shape

        return

    def __enforce_odd_sensor_pixel_number(self):
        # Note: Need to go back and see if this is actually needed
        # -- I think we want to keep this enforced for safety in usage and downstream assumptions
        sensor_pixel_number = self.__dict__["sensor_pixel_number"]
        sensor_num_x = sensor_pixel_number["x"]
        sensor_num_y = sensor_pixel_number["y"]
        if np.mod(sensor_num_x, 2) == 0:
            sensor_num_x += 1
        if np.mod(sensor_num_y, 2) == 0:
            sensor_num_y += 1

        # Add a radial key for convenience in the propagator functions
        sensor_num_r = int((sensor_num_x - 1) / 2 + 1)
        self.__dict__["sensor_pixel_number"] = {"x": sensor_num_x, "y": sensor_num_y, "r": sensor_num_r}

        return

    def __add_tensors(self):
        calc_samplesN = self.__dict__["calc_samplesN"]
        calc_ms_dx_um = self.__dict__["calc_ms_dx_um"]
        calc_sensor_dx_um = self.__dict__["calc_sensor_dx_um"]
        radial_symmetry = self.__dict__["radial_symmetry"]
        dtype = self.__dict__["dtype"]
        torch_zero = torch.tensor(0.0, dtype=dtype)

        # Add scalar torch zero (because why not)
        self.__dict__["_torch_zero"] = torch_zero

        # General lens coord calc_pixel
        _input_pixel_x, _input_pixel_y = torch_coordinate_grid(calc_samplesN, calc_ms_dx_um, radial_symmetry, dtype)
        self.__dict__["_input_pixel_x"] = _input_pixel_x
        self.__dict__["_input_pixel_y"] = _input_pixel_y

        # Angular Spectrum coordinate grid
        if self.__dict__["diffractionEngine"] == "ASM_fourier":
            padFactor = self.__dict__["ASM_Pad_opt"]
            padhalfx = int(calc_samplesN["x"] * padFactor)
            padhalfy = int(calc_samplesN["y"] * padFactor)
            if radial_symmetry:
                x, y = torch.meshgrid(
                    torch.arange(0, calc_samplesN["r"] + padhalfx, dtype=dtype),
                    torch.arange(1, dtype=dtype),
                    indexing="xy",
                )
            else:
                x, y = torch.meshgrid(
                    torch.arange(0, calc_samplesN["x"] + 2 * padhalfx, dtype=dtype),
                    torch.arange(0, calc_samplesN["y"] + 2 * padhalfy, dtype=dtype),
                    indexing="xy",
                )
                x = x - (x.shape[1] - 1) / 2
                y = y - (y.shape[0] - 1) / 2
            x = x * calc_ms_dx_um["x"]
            y = y * calc_ms_dx_um["y"]
            self.__dict__["_ASM_x"] = x
            self.__dict__["_ASM_y"] = y

        ### Fresnel Fourier Optics tensors
        elif self.__dict__["diffractionEngine"] == "fresnel_fourier":
            if radial_symmetry:
                norm_constant = torch.tensor(
                    calc_ms_dx_um["x"] * calc_ms_dx_um["y"] * calc_sensor_dx_um["x"] * calc_sensor_dx_um["y"] * calc_samplesN["x"] * calc_samplesN["y"],
                    dtype=dtype,
                )
                normterm = torch.complex(torch.sqrt(1.0 / norm_constant), torch_zero)
                ang_fx = torch.arange(0, 1 / 2 / calc_ms_dx_um["x"], 1 / calc_ms_dx_um["x"] / calc_samplesN["x"], dtype=dtype)
                self.__dict__["_ang_fx"] = ang_fx
            else:
                norm_constant = torch.tensor(
                    calc_ms_dx_um["x"] * calc_ms_dx_um["y"] / calc_sensor_dx_um["x"] / calc_sensor_dx_um["y"] / calc_samplesN["x"] / calc_samplesN["y"],
                    dtype=dtype,
                )
                normterm = torch.complex(torch.sqrt(norm_constant), torch_zero)
            self.__dict__["_normterm"] = normterm

            # Compute output grid for fourier coefficients
            output_pixel_x, output_pixel_y = torch_coordinate_grid(calc_samplesN, calc_sensor_dx_um, radial_symmetry, dtype)
            self.__dict__["_output_pixel_x"] = output_pixel_x
            self.__dict__["_output_pixel_y"] = output_pixel_y

        return

    def move_tensors_to_device(self, device):
        for key in ADDED_TENSOR_KEYS:
            try:
                self.__dict__[key] = self.__dict__[key].to(device)
            except:
                continue
        return

    ### Run regularization on input output space for proper calculations
    def print(self):
        return print_full_settings(self)

    def _print_info(self):
        wavelength_um = self.__dict__["wavelength_um"]
        ms_length_um = self.__dict__["ms_length_um"]
        ms_dx_um = self.__dict__["ms_dx_um"]
        calc_ms_dx_um = self.__dict__["calc_ms_dx_um"]
        ms_samplesM = self.__dict__["ms_samplesM"]
        calc_samplesM = self.__dict__["calc_samplesM"]
        calc_samplesN = self.__dict__["calc_samplesN"]
        initial_sensor_dx_um = self.__dict__["initial_sensor_dx_um"]
        calc_sensor_dx_um = self.__dict__["calc_sensor_dx_um"]
        sensor_pixel_size_um = self.__dict__["sensor_pixel_size_um"]
        sensor_pixel_number = self.__dict__["sensor_pixel_number"]
        sensor_distance_um = self.__dict__["sensor_distance_um"]
        radius_um = self.__dict__["radius_um"]

        print("\n OVERVIEW OF PARAMETERS \n")
        print("\n", "wavelength_um", wavelength_um)
        print("\n", "ms_length_um: ", ms_length_um)
        print("\n", "radius_um: ", radius_um)
        print("\n", "ms_dx_um: ", ms_dx_um)
        print("\n", "calc_ms_dx_um: ", calc_ms_dx_um)
        print("\n", "ms_samplesM: ", ms_samplesM)
        print("\n", "calc_samplesM: ", calc_samplesM)
        print("\n", "calc_samplesN: ", calc_samplesN)
        print("\n", "initial_sensor_dx_um: ", initial_sensor_dx_um)
        print("\n", "calc_sensor_dx_um: ", calc_sensor_dx_um)
        print("\n", "sensor_pixel_size_um: ", sensor_pixel_size_um)
        print("\n", "sensor_pixel_number: ", sensor_pixel_number)
        print("\n", "sensor_distance_um: ", sensor_distance_um)
        print("\n")
        return

    def __regularizeInputOutputSpace(self):
        # Call to update lens sampling rate and number of samples representing the upadded lens
        self.__regularizeLensPlaneSampling()

        # Call to compute the required padding to achieve an approximate target sampling for field before sensor plane
        self.__regularizeSensorPlaneSampling()

        # Print statement to show if samples have changed relative to what the user requested
        if self.__verbose:
            self._print_info()

        return

    def __regularizeLensPlaneSampling(self):
        ### Setup values for calc_ms_dx and calc_ms_dy which is the initial field grid size that will be used during calculations!
        ### which may be smaller than the user specified grid in order to try and get more accurate field caluclations

        # Obtain the max estimated bandwidth required to compute a metasurface sampling rate cutoff
        # For the fresnel case, just get estBandwidth from quadratic phase profile
        # For the exact transfer function, we should consider the estBandwidth vs the H bandwidth
        diffractionEngine = self.__dict__["diffractionEngine"]
        if diffractionEngine == "fresnel_fourier":
            estBandwidth = np.array(estimateBandwidth(self.__dict__))

        elif diffractionEngine == "ASM_fourier":
            wavelength_um = self.__dict__["wavelength_um"]
            tf_bandwidth = np.array([1 / wavelength_um, 1 / wavelength_um])
            quad_bandwidth = np.array(estimateBandwidth(self.__dict__))
            estBandwidth = np.maximum(tf_bandwidth, quad_bandwidth)

        samplingCutoff = 1 / 2 / estBandwidth

        # automatic_upsample determines if we use our estimate calculation for sampling requirement or just use the user
        # specified number of samples with some manual upsample factor
        ms_dx_um = self.__dict__["ms_dx_um"]
        automatic_upsample = self.__dict__["automatic_upsample"]
        manual_upsample_factor = int(self.__dict__["manual_upsample_factor"])
        if automatic_upsample:
            calc_ms_dx = samplingCutoff[0] if ms_dx_um["x"] > samplingCutoff[0] else ms_dx_um["x"]
            calc_ms_dy = samplingCutoff[1] if ms_dx_um["y"] > samplingCutoff[1] else ms_dx_um["y"]
        else:
            calc_ms_dx = ms_dx_um["x"] / manual_upsample_factor if manual_upsample_factor > 1.0 else ms_dx_um["x"]
            calc_ms_dy = ms_dx_um["y"] / manual_upsample_factor if manual_upsample_factor > 1.0 else ms_dx_um["y"]

        # If we are using the transferFunction engine, we should also verify that calc_ms_dx is
        # smaller than the initial_sensor_dx_m requested by the user. This is because the lens sampling will
        # directly equal the sensor plane sampling.
        # To give the user the right value, we would rather downsample the accuracte calculation vs upsample the coarse calculation
        if diffractionEngine == "ASM_fourier":
            initial_sensor_dx_um = self.__dict__["initial_sensor_dx_um"]
            calc_ms_dx = initial_sensor_dx_um["x"] if calc_ms_dx > initial_sensor_dx_um["x"] else calc_ms_dx
            calc_ms_dy = initial_sensor_dx_um["y"] if calc_ms_dy > initial_sensor_dx_um["y"] else calc_ms_dy

        ### Given the new calculation grid size, we should change the number of samples taken at the input grid
        ### (e.g. after upsampling) so that the originally data is correctly represented.
        # add a parameter to reperesent the corresponding number of samples to be used for unpadded, upsampled lens ("M")
        ms_length_um = self.__dict__["ms_length_um"]
        calc_samplesM_x = int(round(ms_length_um["x"] / calc_ms_dx))
        calc_samplesM_y = int(round(ms_length_um["y"] / calc_ms_dy))

        # Ensure number of samples for unpadded lens is odd
        calc_samplesM_x = calc_samplesM_x + 1 if np.mod(calc_samplesM_x, 2) == 0 else calc_samplesM_x
        calc_samplesM_y = calc_samplesM_y + 1 if np.mod(calc_samplesM_y, 2) == 0 else calc_samplesM_y

        # samples along r is added in for convenience
        calc_samplesM_r = int((calc_samplesM_x - 1) / 2 + 1)
        self.__dict__["calc_samplesM"] = {
            "x": calc_samplesM_x,
            "y": calc_samplesM_y,
            "r": calc_samplesM_r,
        }

        # update the calculated grid that results
        calc_ms_dx = ms_length_um["x"] / calc_samplesM_x
        calc_ms_dy = ms_length_um["y"] / calc_samplesM_y
        self.__dict__["calc_ms_dx_um"] = {"x": calc_ms_dx, "y": calc_ms_dy}

        return

    def __regularizeSensorPlaneSampling(self):
        diffractionEngine = self.__dict__["diffractionEngine"]
        initial_sensor_dx_um = self.__dict__["initial_sensor_dx_um"]
        calc_samplesM = self.__dict__["calc_samplesM"]
        calc_ms_dx_um = self.__dict__["calc_ms_dx_um"]
        padms_halfx = 0
        padms_halfy = 0

        ### For the Fresnel Engine, the output field grid size is tuned by zero-padding the input field
        # Based on new ms sampling rate used during calculations, define required padding for the fresnel fourier engine
        # to achieve a target sensor sampling.

        # Also note, we always allow the output grid to be finer than that requested but it should be padded if it is larger!
        # Reinterpolation and area-integrated downsampling is used later to put back to defined grid
        # samplesN corresponds to the length of the lens space after padding
        if diffractionEngine == "fresnel_fourier":
            wavelength_um = self.__dict__["wavelength_um"]
            sensor_distance_um = self.__dict__["sensor_distance_um"]

            estN_x = int(math.ceil(wavelength_um * sensor_distance_um / initial_sensor_dx_um["x"] / calc_ms_dx_um["x"]))
            estN_y = int(math.ceil(wavelength_um * sensor_distance_um / initial_sensor_dx_um["y"] / calc_ms_dx_um["y"]))

            padms_halfx = int(math.ceil((estN_x - calc_samplesM["x"]) / 2)) if estN_x > calc_samplesM["x"] else 0
            padms_halfy = int(math.ceil((estN_y - calc_samplesM["y"]) / 2)) if estN_y > calc_samplesM["y"] else 0

        ### For the ASM case, we already checked that the ms grid size is smaller or equal to the sensor grid size
        # Now we should pad to ensure that fields are calculated on the entire sensing space requested by user
        if diffractionEngine == "ASM_fourier":
            sensor_pixel_size_um = self.__dict__["sensor_pixel_size_um"]
            sensor_pixel_number = self.__dict__["sensor_pixel_number"]
            current_span_x = calc_ms_dx_um["x"] * calc_samplesM["x"]
            current_span_y = calc_ms_dx_um["y"] * calc_samplesM["y"]
            desired_span_x = sensor_pixel_size_um["x"] * sensor_pixel_number["x"]
            desired_span_y = sensor_pixel_size_um["y"] * sensor_pixel_number["y"]

            if current_span_x < desired_span_x:
                padms_halfx = int(round((desired_span_x - current_span_x) / 2 / calc_ms_dx_um["x"]))

            if current_span_y < desired_span_y:
                padms_halfy = int(round((desired_span_y - current_span_y) / 2 / calc_ms_dx_um["y"]))

        # Update the parameter settings based on new padding settings
        self.__dict__["padms_half"] = {"x": padms_halfx, "y": padms_halfy}
        calc_samplesN_x = padms_halfx * 2 + calc_samplesM["x"]
        calc_samplesN_y = padms_halfy * 2 + calc_samplesM["y"]
        calc_samplesN_r = int((calc_samplesN_x - 1) / 2 + 1)
        self.__dict__["calc_samplesN"] = {
            "x": calc_samplesN_x,
            "y": calc_samplesN_y,
            "r": calc_samplesN_r,
        }

        ### Redefine the exact sensor calculation grid size now after trying to get as close to or smaller than requested
        if diffractionEngine == "fresnel_fourier":
            calc_sensor_dx_um = wavelength_um * sensor_distance_um / calc_ms_dx_um["x"] / calc_samplesN_x
            calc_sensor_dy_um = wavelength_um * sensor_distance_um / calc_ms_dx_um["y"] / calc_samplesN_y

        elif diffractionEngine == "ASM_fourier":
            calc_sensor_dx_um = calc_ms_dx_um["x"]
            calc_sensor_dy_um = calc_ms_dx_um["y"]

        # Update the parameter setting with the new sensor plane calculation
        self.__dict__["calc_sensor_dx_um"] = {"x": calc_sensor_dx_um, "y": calc_sensor_dy_um}

        return

    def _set_wavelength_m_None(self):
        # This is a special routine that is used in an experimental implementation of the ASM (matrix instead of loops)
        # In that set of subroutines, we want to set the wavelength, regularize the grid, and then change wavelength to
        # None to avoid mistakes because we don't want the wavelength parameter to be used
        self.__dict__["wavelength_m"] = None
        return

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
