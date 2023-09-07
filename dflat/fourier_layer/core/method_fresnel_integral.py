import torch
from torch.fft import fftshift, ifftshift, fft2
import numpy as np
from .ops_hankel import qdht
from .ops_transform_util import tf_generalSpline_regular1DGrid


def fresnel_diffraction_fft(wavefront_ampl, wavefront_phase, wavelength_um, distance_um, parameters):
    """Uses the single-Fourier transform implementation of the Fresnel diffraction equation to propagate fields.

    The complex coefficients in the formulation are excluded here but can be added by appropriately calling
    fresnel_diffraction_coeffs().

    Args:
        `wavefront_ampl` (tf.float): Starting field amplitude, of shape (batch_size, input_pixel_number['y'], input_pixel_number['x'])
            or (batch_size, 1, input_pixel_number['r']).
        `wavefront_phase` (tf.float): Starting field phase, the same shape as wavefront_ampl.
        `wavelength_m` (tf.float): Tf constant defining the wavelength of light for the calculation, in units of m
        `distance_m` (tf.float): Tf constant defining the distance between the starting plane and the propagated plane, in units of m
        'parameters'
    Returns:
        `float64`: Field amplitude at the output plane grid, of shape (batch_size, input_pixel_number['y'], input_pixel_number['x'])
            or (batch_size, 1, input_pixel_number['r']).
        `float64`: Field phase at the output plane grid (batch_size, input_pixel_number['y'], input_pixel_number['x'])
            or (batch_size, 1, input_pixel_number['r'])
    """

    # create the coordinate grid at the input
    input_pixel_x, input_pixel_y = parameters["_input_pixel_x"], parameters["_input_pixel_y"]
    torch_zero = parameters["_torch_zero"]
    radial_symmetry = parameters["radial_symmetry"]

    # fourier transform approximation of fresnel diffraction
    angular_wave_number = 2 * np.pi / wavelength_um
    quadratic_term = angular_wave_number / 2 / distance_um * (input_pixel_x**2 + input_pixel_y**2)
    fourier_transform_trans = wavefront_ampl
    fourier_transform_phase = wavefront_phase + quadratic_term[None]
    fourier_transform_term = torch.complex(fourier_transform_trans, torch_zero) * torch.exp(torch.complex(torch_zero, fourier_transform_phase))

    # If radialy symmetric input, then use the hankel transform otherwise use 2D DFT
    normterm = parameters["_normterm"]
    if radial_symmetry:
        kr, wavefront_outPlane = qdht(input_pixel_x, fourier_transform_term)
        ang_fx = parameters["_ang_fx"]
        wavefront_outPlane = tf_generalSpline_regular1DGrid(kr / 2 / np.pi, ang_fx, wavefront_outPlane) * normterm

    else:
        wavefront_outPlane = fftshift(fft2(ifftshift(fourier_transform_term))) * normterm

    return torch.abs(wavefront_outPlane), torch.angle(wavefront_outPlane)


def fresnel_diffraction_coeffs(out_wavefront_ampl, out_wavefront_phase, wavelength_um, distance_um, parameters):
    """Adds the complex coefficient terms in the Fresnel diffraction integral formulation to the out-plane wavefront,
    computed by fresnel_diffraction_fft().

    Args:
        `out_wavefront_ampl` (tf.float): Field amplitude at the output plane, excluding complex coeffs, after propagating
            by the fresnel method, of shape (batch_size, output_pixel_number["x"], output_pixel_number["y"]) or
            (batch_size, 1, output_pixel_number["r"]).
        `out_wavefront_phase` (tf.float): Field phase at the output plane, excluding complex coeffs, after propagating
            by the fresnel method, of shape (batch_size, output_pixel_number["x"], output_pixel_number["y"]) or
            (batch_size, 1, output_pixel_number["r"]).
        `wavelength_m` (tf.float): tf constant corresponding to the wavelength of the field
            (should match that used in the fresnel_diffraction call)
        `distance_m` (tf.float64): tf constant corresponding to the distance propagated
            (should match that used in the fresnel_diffraction call)
        `output_pixel_size_m` (dict): Output field grid discretization/pitch in units of m, via dictionary {"x": float, "y": float}.
        `output_pixel_number` (dict): Output field grid length in terms of number of pixels, via dictionary {"x": float, "y": float}.
        `dtype` (tf.dtype): Datatype to be used in the tensorflow calculation; Only tf.float64 is currently supported.
        `radial_symmetry` (bool): Flag indicating if radial symmetry is used.

    Returns:
        `float`: Field amplitude at the output plane with the complex coefficients added in. same shape as input.
        `float`: Field phase at the output plane with the compelx coefficients added in. same shape as input.
    """
    # create the output plane coordinate grid
    output_pixel_x, output_pixel_y = parameters["_output_pixel_x"], parameters["_output_pixel_y"]
    output_pixel_x = output_pixel_x[None]
    output_pixel_y = output_pixel_y[None]

    # add the final terms
    torch_zero = parameters["_torch_zero"]
    angular_wave_number = 2 * np.pi / wavelength_um
    quadterm = distance_um + (output_pixel_x**2 + output_pixel_y**2) / 2 / distance_um
    ### Neglect the power term, 1/i/lambda/z since we want psf normalized by energy at the end
    # wavefront = (
    #     tf.complex(wavefront_trans, TF_ZERO)
    #     * tf.complex(TF_ZERO, tf.cast(1.0 / wavelength_m / distance_m, dtype))
    #     * tf.exp(tf.complex(TF_ZERO, wavefront_phase + angular_wave_number * quadterm))
    # )
    wavefront = torch.complex(out_wavefront_ampl, torch_zero) * torch.exp(torch.complex(torch_zero, out_wavefront_phase + angular_wave_number * quadterm))

    return torch.abs(wavefront), torch.angle(wavefront)
