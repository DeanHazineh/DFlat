from .core.ops_field_aperture import sqrt_energy_illumination, gen_aperture_disk
import numpy as np


def focus_lens_init(parameters, wavelength_m_aslist, focal_distance_m_aslist, focus_offset_m_aslist):
    """Wrapper function for gen_focusing_profile. Generate a stack of theoretical transmittance and phase profiles for
    focusing light of particular depths and wavelengths, onto the sensor plane.

    This is sometimes more convenient to call than `gen_focusing_profiles` directly as it does not require explicitly
    unpacking the relevent function inputs in another script. Returns the profile(s) for focusing given the incident
    wavelength, object distance, of off-axis focal shift combinations passed in list form.

    Args:
        `parameters` (prop_param):  Settings object defining field propagation details. wavelength parameter key is not
            used. Only the geometry of the metasurface grid is pulled out.
        `wavelength_m_aslist` (list): N length list of floats defining incident wavelength_m in meters, to be designed
            for by the metasurface profile instantiation.
        `focal_distance_m_aslist` (list): N length list containing in-focus object-to-lens plane distances, in units of
            meters.
        `focus_offset_m_aslist` (list): N length list containing the off-axis focal-point shift at the sensor,
            via {"x": np.float, "y": np.float}.

    Raises:
        TypeError: focal_distance_m_aslist and focus_offset_m_aslist must be passed as list.
        ValueError: All inputted lists must have same number of items.
        ValueError: lens radius exceeds the minimum radius of the initialzed lens-grid.

    Returns:
        `np.float`: Transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or (n, 1, ms_samplesM["r"]).
        `np.float`: Phase profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or (n, 1, ms_samplesM["r"]).
        `np.float`: Aperture transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or (n, 1, ms_samplesM["r"]).
        `np.float`: Sqrt of total energy incident on the metasurface, post-aperture, of shape (n, 1, 1).

    """

    # Unpack necessary arguments from parameters
    sensor_distance_m = parameters["sensor_distance_m"]
    ms_dx_m = parameters["ms_dx_m"]
    ms_samplesM = parameters["ms_samplesM"]
    radial_symmetry = parameters["radial_symmetry"]
    radius_m = parameters["radius_m"]

    # Handle exception if focus_distance, wavelength, or offset are not passed as lists
    if (type(focal_distance_m_aslist) is not list) or (type(focus_offset_m_aslist) is not list) or (type(wavelength_m_aslist) is not list):
        raise TypeError("focus_lens_init: focal_distance_m_aslist and focus_offset_m_aslist must be passed as list")

    # Handle exception if list lengths are not all the same
    length = len(wavelength_m_aslist)
    if not all(len(lst) == length for lst in [focal_distance_m_aslist, focus_offset_m_aslist]):
        raise ValueError("focus_lens_init: All inputted lists must have the same length")

    # Loop over focusing lenses and generate metasurfaces profiles
    lens_transmittanceStack = []
    lens_phaseStack = []
    aperture_transmittanceStack = []
    for iter in range(length):
        (
            lens_transmittance,
            lens_phase,
            aperture_transmittance,
            _,
        ) = gen_focusing_profile(
            ms_samplesM,
            ms_dx_m,
            wavelength_m_aslist[iter],
            focal_distance_m_aslist[iter],
            focus_offset_m_aslist[iter],
            sensor_distance_m,
            radius_m,
            radial_symmetry,
        )

        lens_transmittanceStack.append(lens_transmittance)
        lens_phaseStack.append(lens_phase)
        aperture_transmittanceStack.append(aperture_transmittance)

    lens_transmittanceStack = np.stack(lens_transmittanceStack)
    lens_phaseStack = np.stack(lens_phaseStack)
    aperture_transmittanceStack = np.stack(aperture_transmittanceStack)
    sqrt_energy_illum = sqrt_energy_illumination(aperture_transmittanceStack, ms_dx_m, radial_symmetry)

    return (
        lens_transmittanceStack,
        lens_phaseStack,
        aperture_transmittanceStack,
        sqrt_energy_illum,
    )


def randomPhase_lens_init(parameters, numlenses):
    """Generate a random phase profile with unity transmittance.

    Args:
        `parameters` (prop_param):  Settings object defining field propagation details including metasurface to sensor
            distance.
        `numlenses` (int): number n of random transmission and phase profiles to generate

    Returns:
        `np.float`: Unity transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or
            (n, 1, ms_samplesM["r"]).
        `np.float`: Random phase profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or
            (n, 1, ms_samplesM["r"]).
        `np.float`: Aperture transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or
            (n, 1, ms_samplesM["r"]).
        `np.float`: Sqrt of total energy incident on the metasurface, post-aperture, of shape (n, 1, 1).
    """
    ms_samplesM = parameters["ms_samplesM"]
    radial_symmetry = parameters["radial_symmetry"]
    numpts_y = 1 if radial_symmetry else ms_samplesM["y"]
    numpts_x = ms_samplesM["r"] if radial_symmetry else ms_samplesM["x"]

    lens_trans = np.ones((numlenses, numpts_y, numpts_x))
    aperture, sqrt_energy_illum = gen_aperture_disk(parameters)
    lens_phase = 2 * np.pi * np.random.rand(lens_trans.shape)

    return lens_trans, lens_phase, aperture, sqrt_energy_illum


def gen_focusing_profile(
    ms_samplesM,
    ms_dx_m,
    wavelength_m,
    focal_distance_m,
    focus_offset_m,
    sensor_distance_m,
    radius_m,
    radial_symmetry,
):
    """Generate the metasurface phase and transmittance profile for ideal focusing of light. Focusing is designed for a
    single, input wavelength, object plane distance/depth, and sensor distance. The lens may also be specified to focus
    light with an off-axis focal shift.

    Args:
        `ms_samplesM` (dict): Dictionary specifying the number of points in the metasurface cartesian grid, of form
            {"x": float, "y": float, "r": float}.
        `ms_dx_m` (dict): Dictionary specifying the grid discretization/pitch along x and y, of form {"x": float, "y": float}.
        `wavelength_m` (float): Wavelength of incident light, in units of m.
        `focal_distance_m` (float): Object plane distance/depth that will be imaged in focus, in units of m.
        `focus_offset_m` (dict): Off-axis shift of the focused psf at the sensor plane, in units of m and of the form
            {"x": float, "y": float}.
        `sensor_distance_m` (float): Distance from the metasurface to the sensor plane, in units of m.
        `radius_m` (float): Radius of the metasurface via a circular field aperture placed before the metasurface. If
            None, no field aperture is considered and a square metasurface is returned.
        `radial_symmetry` (bool): Radial symmetry flag denoting if the profile to be returned is 2D or a 1D radial
            vector of phase and transmission.

    Returns:
        `float`: Metasurface transmittance, of shape (ms_samplesM["y"], ms_samplesM["x"]) or (1, ms_samplesM["r"])
        `float`: Metasurface phase profile, of shape (ms_samplesM["y"], ms_samplesM["x"]) or (1, ms_samplesM["r"])
        `float`: Pre-metasurface field aperture (None or circular disk), of shape (ms_samplesM["y"], ms_samplesM["x"])
            or (1, ms_samplesM["r"])
        `float`: Total energy passing through the aperture and incident on the metasurface, used for normalizing psfs downstream
    """

    # Generate coordinate grid on the lens plane
    xx, yy = np.meshgrid(np.arange(ms_samplesM["x"]), np.arange(ms_samplesM["y"]))
    xx = xx - (xx.shape[1] - 1) / 2
    yy = yy - (yy.shape[0] - 1) / 2
    xx = xx * ms_dx_m["x"]
    yy = yy * ms_dx_m["y"]

    # Define focusing profile phase and transmittance (wrapped to 2pi)
    lens_phase = (
        -2
        * np.pi
        / wavelength_m
        * (
            np.sqrt(focal_distance_m**2 + xx**2 + yy**2)
            + np.sqrt(sensor_distance_m**2 + (xx - focus_offset_m["x"]) ** 2 + (yy - focus_offset_m["y"]) ** 2)
        )
    )
    lens_transmittance = np.ones_like(lens_phase)
    lens_phase = np.angle(lens_transmittance * np.exp(1j * lens_phase))

    # Define the lens aperture
    # Add a small transmittance to the block portion -- it was once reported this fixed nan gradients
    aperture_transmittance = np.ones_like(lens_phase)
    if radius_m:
        aperture_transmittance = ((np.sqrt(xx**2 + yy**2) <= radius_m)).astype(np.float32) + 1e-6

    # Get measure of total radiance passed through the aperture onto the metasurface which is useful for
    # field normalizations
    sqrt_energy_illum = sqrt_energy_illumination(np.expand_dims(aperture_transmittance, 0), ms_dx_m, False)

    # Handle return case if radial_symmetry flag is active
    # If the radius is of length N, the total size of the full 2D lens is (2N-1)x(2N-1)
    if radial_symmetry:
        ms_samplesM_r = ms_samplesM["r"]
        lens_transmittance = lens_transmittance[ms_samplesM_r - 1 : ms_samplesM_r, ms_samplesM_r - 1 :]
        lens_phase = lens_phase[ms_samplesM_r - 1 : ms_samplesM_r, ms_samplesM_r - 1 :]
        aperture_transmittance = aperture_transmittance[ms_samplesM_r - 1 : ms_samplesM_r, ms_samplesM_r - 1 :]

    return lens_transmittance, lens_phase, aperture_transmittance, sqrt_energy_illum
