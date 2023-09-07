import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift, fft2, ifft2
from .ops_hankel import iqdht, qdht
from .ops_transform_util import tf_generalSpline_regular1DGrid, resize_with_crop_or_pad


def transfer_function_Broadband(wavefront_ampl, wavefront_phase, wavelength_set_um, distance_um, parameters):
    """Uses the angular spectrum method to propagate a broadband field

    Args:
        `wavefront_ampl` (tf.float): Amplitude of the field via tensor of shape (Nbatch, Nwl or 1, input_pixel_number['y'], input_pixel_number['x'])
            or of shape (Nbatch, Nwl or 1, 1, input_pixel_number['r'])
        `wavefront_phase` (tf.float): Phase of the field, the same shape as wavefront_ampl
        `wavelength_set_m` (tf.float): list of simulation wavelengths
        `distance_m` (tf.float): rank 0 tensor corresponding to the distance to propagate the input field
        'modified parameters'

    Returns:
        `float64`: Field amplitude at the output plane grid, the same shape as the input field
        `float64`: Field phase at the output plane grid, the same shape as the input field
    """

    # trans and phase input has shape (Nbatch, Nwl, Ny, Nx)
    torch_zero = parameters["_torch_zero"]
    radial_symmetry = parameters["radial_symmetry"]
    padFactor = parameters["ASM_Pad_opt"]
    calc_samplesN = parameters["calc_samplesN"]
    padhalfx = int(calc_samplesN["x"] * padFactor)
    padhalfy = int(calc_samplesN["y"] * padFactor)
    if radial_symmetry:
        paddings = (0, padhalfx, 0, 0, 0, 0, 0, 0)
    else:
        paddings = (padhalfx, padhalfx, padhalfy, padhalfy, 0, 0, 0, 0)
    padded_wavefront_ampl = F.pad(wavefront_ampl, paddings, mode="constant", value=0)
    padded_wavefront_phase = F.pad(wavefront_phase, paddings, mode="constant", value=0)

    # Get the coordinate grid (pre-initialized and on the device)
    x = parameters["_ASM_x"]
    y = parameters["_ASM_y"]

    ### Get the angular decomposition of the input field
    fourier_transform_term = torch.complex(padded_wavefront_ampl, torch_zero) * torch.exp(torch.complex(torch_zero, padded_wavefront_phase))
    if radial_symmetry:
        kr, angular_spectrum = qdht(torch.squeeze(x), fourier_transform_term)
    else:
        angular_spectrum = fftshift(fft2(ifftshift(fourier_transform_term)))

    #### Define the transfer function via FT of the Sommerfield solution
    # Define the output field grid
    rarray = torch.sqrt(distance_um**2 + x**2 + y**2)
    rarray = rarray[None, None, :, :]
    angular_wavenumber = 2 * np.pi / wavelength_set_um
    angular_wavenumber = angular_wavenumber[None, :, None, None]
    h = (
        torch.complex(1 / 2 / np.pi * distance_um / rarray**2, torch_zero)
        * torch.complex(1 / rarray, -1 * angular_wavenumber)
        * torch.exp(torch.complex(torch_zero, angular_wavenumber * rarray))
    )

    # Compute Fourier Space Transfer Function
    if radial_symmetry:
        kr, H = qdht(torch.squeeze(x), h)
    else:
        H = fftshift(fft2(ifftshift(h)))

    # note: we have decided to ingore the physical, depth dependent energy scaling here (and in the fresnel method)
    # This change makes it easier to play with normalized PSF (Energy under the IPSF less than or equal to energy incident on aperture)
    H = torch.exp(torch.complex(torch_zero, torch.angle(H)))

    ### Propagation by multiplying angular decomposition with H then taking the inverse transform
    fourier_transform_term = angular_spectrum * H
    if radial_symmetry:
        r2, outputwavefront = iqdht(kr, fourier_transform_term)
        outputwavefront = tf_generalSpline_regular1DGrid(r2, torch.squeeze(x), outputwavefront)
    else:
        outputwavefront = fftshift(ifft2(ifftshift(fourier_transform_term)))

    ### Crop to remove the padding used in the calculation
    # Radial symmetry needs asymmetric cropping not central crop
    target_width = calc_samplesN["r"] if radial_symmetry else calc_samplesN["x"]
    target_height = 1 if radial_symmetry else calc_samplesN["y"]
    outputwavefront = resize_with_crop_or_pad(outputwavefront, target_height, target_width, radial_symmetry)

    return torch.abs(outputwavefront), torch.angle(outputwavefront)


def transfer_function_diffraction(wavefront_ampl, wavefront_phase, wavelength_um, distance_um, parameters):
    """Uses the angular spectrum method to propagate an input complex field to the output plane.

    Args:
        `wavefront_ampl` (tf.float): Starting field amplitude, of shape (batch, input_pixel_number['y'], input_pixel_number['x'])
            or (batch, 1, input_pixel_number['r'])
        `wavefront_phase` (tf.float): Starting field phase, the same shape as wavefront_ampl.
        `wavelength_m` (tf.float): Tf constant defining the wavelength of light for the calculation, in units of m
        `distance_m` (tf.float64): Tf constant defining the distance between the starting plane and the propagated plane, in units of m
        'parameters'
    Returns:
        `tf.float64`: Field amplitude at the output plane grid, of shape
            (batch, input_pixel_number['y'], input_pixel_number['x']) or (batch, 1, input_pixel_number['r'])
        `tf.float64`: Field phase at the output plane grid, of shape
            (batch, input_pixel_number['y'], input_pixel_number['x']) or (batch, 1, input_pixel_number['r'])

    """
    # trans and phase input has shape (Nbatch, Nwl, Ny, Nx)
    torch_zero = parameters["_torch_zero"]
    padFactor = parameters["ASM_Pad_opt"]
    calc_samplesN = parameters["calc_samplesN"]
    padhalfx = int(calc_samplesN["x"] * padFactor)
    padhalfy = int(calc_samplesN["y"] * padFactor)
    radial_symmetry = parameters["radial_symmetry"]
    if radial_symmetry:
        paddings = (0, padhalfx, 0, 0, 0, 0)
    else:
        paddings = (padhalfx, padhalfx, padhalfy, padhalfy, 0, 0)
    padded_wavefront_ampl = F.pad(wavefront_ampl, paddings, mode="constant", value=0)
    padded_wavefront_phase = F.pad(wavefront_phase, paddings, mode="constant", value=0)

    # Get the coordinate grid (pre-initialized and on the device)
    x = parameters["_ASM_x"]
    y = parameters["_ASM_y"]

    ### Get the angular decomposition of the input field
    fourier_transform_term = torch.complex(padded_wavefront_ampl, torch_zero) * torch.exp(torch.complex(torch_zero, padded_wavefront_phase))
    if radial_symmetry:
        kr, angular_spectrum = qdht(x, fourier_transform_term)
    else:
        angular_spectrum = fftshift(fft2(ifftshift(fourier_transform_term)))

    #### Define the transfer function via FT of the Sommerfield solution
    # Define the output field grid
    rarray = torch.sqrt(distance_um**2 + x**2 + y**2)
    angular_wavenumber = torch.tensor(2 * np.pi / wavelength_um).type_as(rarray)
    h = (
        torch.complex(1 / 2 / np.pi * distance_um / rarray**2, torch_zero)
        * torch.complex(1 / rarray, -1 * angular_wavenumber)
        * torch.exp(torch.complex(torch_zero, angular_wavenumber * rarray))
    ).unsqueeze(0)

    # Compute Fourier Space Transfer Function
    if radial_symmetry:
        kr, H = qdht(torch.squeeze(x), h)
    else:
        H = fftshift(fft2(ifftshift(h)))

    # note: we have decided to ingore the physical, depth dependent energy scaling here (and in the fresnel method)
    # This change makes it easier to play with normalized PSF (Energy under the IPSF less than or equal to energy incident on aperture)
    H = torch.exp(torch.complex(torch_zero, torch.angle(H)))

    ### Propagation by multiplying angular decomposition with H then taking the inverse transform
    fourier_transform_term = angular_spectrum * H
    if radial_symmetry:
        r2, outputwavefront = iqdht(kr, fourier_transform_term)
        outputwavefront = tf_generalSpline_regular1DGrid(r2, torch.squeeze(x), outputwavefront)
    else:
        outputwavefront = fftshift(ifft2(ifftshift(fourier_transform_term)))

    ### Crop to remove the padding used in the calculation
    # Radial symmetry needs asymmetric cropping not central crop
    target_width = calc_samplesN["r"] if radial_symmetry else calc_samplesN["x"]
    target_height = 1 if radial_symmetry else calc_samplesN["y"]
    outputwavefront = resize_with_crop_or_pad(outputwavefront, target_height, target_width, radial_symmetry)

    return torch.abs(outputwavefront), torch.angle(outputwavefront)
