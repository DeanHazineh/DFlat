import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.fft import fftshift, ifftshift, fft2, ifft2
from einops import rearrange

from .util import cart_grid
from dflat.radial_tranforms import (
    qdht,
    iqdht,
    general_interp_regular_1d_grid,
    radial_2d_transform,
    radial_2d_transform_wrapped_phase,
    resize_with_crop_or_pad,
)


class BaseFrequencySpace(nn.Module):
    """Base class for frequency space propagation methods.

    This class provides common initialization and validation for frequency space
    propagation methods such as Fresnel and Angular Spectrum Method (ASM).

    Attributes:
        in_size (np.ndarray): Input field size [height, width].
        in_dx_m (np.ndarray): Input pixel size in meters [dy, dx].
        out_distance_m (float): Propagation distance in meters.
        out_size (np.ndarray): Output field size [height, width].
        out_dx_m (np.ndarray): Output pixel size in meters [dy, dx].
        wavelength_set_m (np.ndarray): Set of wavelengths in meters.
        out_resample_dx_m (np.ndarray): Output resampling pixel size in meters [dy, dx].
        manual_upsample_factor (float): Manual upsampling factor for input field.
        radial_symmetry (bool): If True, assume radial symmetry in the input field.

    Args:
        in_size (List[int]): Input field size [height, width].
        in_dx_m (List[float]): Input pixel size in meters [dy, dx].
        out_distance_m (float): Propagation distance in meters.
        out_size (List[int]): Output field size [height, width].
        out_dx_m (List[float]): Output pixel size in meters [dy, dx].
        wavelength_set_m (List[float]): Set of wavelengths in meters.
        out_resample_dx_m (List[float], optional): Output resampling pixel size in meters [dy, dx].
        manual_upsample_factor (float, optional): Manual upsampling factor for input field. Defaults to 1.
        radial_symmetry (bool, optional): If True, assume radial symmetry in the input field. Defaults to False.
    """

    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        wavelength_set_m,
        out_resample_dx_m,
        manual_upsample_factor,
        radial_symmetry,
    ):
        super().__init__()
        # Quick checks on the inputs
        out_resample_dx_m = out_dx_m if out_resample_dx_m is None else out_resample_dx_m
        assert isinstance(radial_symmetry, bool), "radial symmetry must be boolean."
        assert manual_upsample_factor >= 1, "manual_upsample factor must >= 1."
        assert isinstance(out_distance_m, float), "out_distance_m must be float."
        assert isinstance(
            wavelength_set_m, (list, np.ndarray)
        ), "wavelength_set_m must be a list or numpy array."
        for obj in [in_size, in_dx_m, out_size, out_dx_m, out_resample_dx_m]:
            assert len(obj) == 2, "Expected len 2 list for inputs."

        # Move to attributes and convert lists to numpy arrays
        self.in_size = np.array(in_size).astype(int)
        self.in_dx_m = np.array(in_dx_m)
        self.out_distance_m = out_distance_m
        self.out_size = np.array(out_size).astype(int)
        self.out_dx_m = np.array(out_dx_m)
        self.wavelength_set_m = np.array(wavelength_set_m)
        self.out_resample_dx_m = np.array(out_resample_dx_m)
        self.manual_upsample_factor = manual_upsample_factor
        self.radial_symmetry = radial_symmetry

        # Run additional assertions required for calculations
        self._validate_inputs()

        # Apply unit conversion from meters to different base
        self.rescale = 1e6  # m to micrometer as our default units
        self._unit_conversion()

    def _validate_inputs(self):
        assert np.all(
            self.in_size % 2 != 0
        ), "Input grid must be odd for Fourier-based methods"

        assert all(
            x <= y for x, y in zip(self.out_dx_m, self.out_resample_dx_m)
        ), "out_resample_dx_m must be geq to out_dx."

        if self.radial_symmetry:
            in_size = self.in_size
            in_dx_m = self.in_dx_m
            assert (
                in_size[0] == in_size[1]
            ), "params: in_size must be square (H=W) when radial_symmetry flag is used"
            assert (
                in_dx_m[0] == in_dx_m[1]
            ), "params: in_dx_m must be same along x and y when radial_symmetry flag is used"

    def _unit_conversion(self):
        convert_keys = [
            "in_dx_m",
            "out_distance_m",
            "out_dx_m",
            "out_resample_dx_m",
            "wavelength_set_m",
        ]
        for key in convert_keys:
            obj = self.__dict__[key]
            new_name = key[:-2]

            if isinstance(obj, float) or isinstance(obj, np.ndarray):
                setattr(self, new_name, obj * self.rescale)
            elif obj is None:
                setattr(self, new_name, None)
            else:
                raise ValueError("In key conversion, ran into unknown datatype")


class FresnelPropagation(BaseFrequencySpace):
    """Fresnel propagation method for optical field propagation.

    This class implements the Fresnel propagation method for propagating optical fields
    from an input plane to an output plane.

    Note:
        The Fresnel propagation method is suitable for propagation distances where the
        paraxial approximation holds. To generate fields on a specified, uniform output grid,
        the input grid is sometimes upsampled and/or zero-padded. The memory/output has a non-linear relationship
        with requested output grid size.
    """

    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        wavelength_set_m,
        out_resample_dx_m=None,
        manual_upsample_factor=1,
        radial_symmetry=False,
        verbose=False,
        *args,
        **kwargs,
    ):
        """Initializes the propagation class.

        Args:
            in_size (list): input grid shape as [H, W].
            in_dx_m (list): input grid discretization (in meters) as [dy, dx]
            out_distance_m (float): output plane distance
            out_size (list): output grid shape as [H, W]
            out_dx_m (list): output grid discretization (in meters) as [dy, dx]
            wavelength_set_m (list): List of wavelengths (in meters) corresponding to the wavelength dimension stack in forward.
            out_resample_dx_m (list, optional): List of output grid discretizations to resample by area sum (area averaging for phase). This can be used to compute at a sub-pixel scale then return the integrated field on each pixel. Defaults to None.
            manual_upsample_factor (int, optional): Force factor to manually upsample (nearest neighbor) the input lens. This can improve fourier space calculation accuracy. Defaults to 1.
            radial_symmetry (bool, optional): Flag to use radial symmetry during calculations. Note that we expect radial field profiles to be passed in if True. Defaults to False.
            verbose (bool, optional): If True, prints information about the actual grid sizes etc that will be used in the back-end calculation. This may often be larger than user defined sizes due to fourier space rules. Defaults to False.
        """
        super().__init__(
            in_size,
            in_dx_m,
            out_distance_m,
            out_size,
            out_dx_m,
            wavelength_set_m,
            out_resample_dx_m,
            manual_upsample_factor,
            radial_symmetry,
        )
        self.verbose = verbose
        self._init_calc_params()
        self._init_fresnel_constants()

    def _init_calc_params(self):
        # Determine a manual_upsample_factor as a forced use to avoid field cropping
        factor = self.out_dx * self.out_size * self.in_dx / self.out_distance
        upsample_factors = factor[:, None] / self.wavelength_set[None, :]
        manual_upsample = np.array(
            [self.manual_upsample_factor, self.manual_upsample_factor]
        )
        upsample_factors = np.maximum(upsample_factors, manual_upsample[:, None]).T

        # Compute the input grid and upsampled size when we upsample the input field
        calc_in_dx = self.in_dx / upsample_factors
        in_length = self.in_dx * self.in_size
        calc_samplesM = np.rint(in_length / calc_in_dx)
        calc_samplesM = np.where(
            np.mod(calc_samplesM, 2) == 0, calc_samplesM + 1, calc_samplesM
        )

        self.calc_samplesM = calc_samplesM
        self.calc_samplesM_r = calc_samplesM[-1] // 2
        self.manual_upsample_factor = upsample_factors

        # Update the calculation grid corresponding to the upsampled integer number of samples
        self.calc_in_dx = in_length / calc_samplesM

        # Determine zero-padding required to hit an output target discretization
        estN = np.ceil(
            self.wavelength_set[:, None]
            * self.out_distance
            / self.out_dx[None, :]
            / self.calc_in_dx
        )

        estN = np.where(np.mod(estN, 2) == 0, estN + 1, estN)
        estN = np.where(estN < self.calc_samplesM, self.calc_samplesM, estN)

        pad_in = (estN - self.calc_samplesM) / 2
        self.pad_in = pad_in.astype(int)

        # Define the resulting output grid. For all wavelengths, the output value should be very close or much smaller
        self.calc_samplesN = estN
        self.calc_out_dx = (
            self.wavelength_set[:, None] * self.out_distance / self.calc_in_dx / estN
        )

        if self.verbose == True:
            print("Initializing the Fresnel Method: ")
            print(f"   - in_size: {self.in_size}, calc_in_size:")
            for i in range(len(self.wavelength_set)):
                print(f"      {calc_samplesM[i,:]},")

            print(f"   - padded_calc_in_size: ")
            for i in range(len(self.wavelength_set)):
                print(f"      {self.calc_samplesN[i,:]},")

            print(f"   - in_dx: {self.in_dx}, calc_in_dx: ")
            for i in range(len(self.wavelength_set)):
                print(f"      {calc_in_dx[i,:]},")

            print(f"   - out_dx: {self.out_dx}, calc_out_dx: ")
            for i in range(len(self.wavelength_set)):
                print(f"      {self.calc_out_dx[i,:]},")

    def _init_fresnel_constants(self):
        # Save compute time be pre-initializing tensors for the fresnel calculation
        self.quad_term_in = []
        self.ang_fx = []
        self.x = []
        self.out_phase = []
        self.norm = []
        for i, lam in enumerate(self.wavelength_set):
            x, y = cart_grid(
                self.calc_samplesN[i], self.calc_in_dx[i], self.radial_symmetry
            )
            self.x.append(x)

            quadratic_term = np.pi / lam / self.out_distance * (x**2 + y**2)
            self.quad_term_in.append(quadratic_term)

            if self.radial_symmetry:
                fx = torch.arange(
                    0,
                    1 / 2 / self.calc_in_dx[i, -1],
                    1 / self.calc_in_dx[i, -1] / self.calc_samplesN[i, -1],
                )
                self.ang_fx.append(fx)
                norm = torch.tensor(
                    1.0
                    / np.sqrt(
                        np.prod(self.calc_in_dx[i])
                        * np.prod(self.calc_out_dx[i])
                        * np.prod(self.calc_samplesN[i])
                    )
                )
            else:
                norm = torch.tensor(
                    np.sqrt(
                        np.prod(self.calc_in_dx[i])
                        / np.prod(self.calc_out_dx[i])
                        / np.prod(self.calc_samplesN[i])
                    )
                )
            self.norm.append(norm)

            xo, yo = cart_grid(
                self.calc_samplesN[i], self.calc_out_dx[i], self.radial_symmetry
            )
            out_phase = (
                2
                * np.pi
                / lam
                * (self.out_distance + (xo**2 + yo**2) / 2 / self.out_distance)
            )
            self.out_phase.append(out_phase)

        self.calc_in_dx = torch.tensor(self.calc_in_dx)
        self.calc_out_dx = torch.tensor(self.calc_out_dx)
        self.calc_samplesN = torch.tensor(self.calc_samplesN)
        self.torch_zero = torch.tensor(0.0)
        return

    def forward(self, amplitude, phase, **kwargs):
        """Propagates a complex field from an input plane to a planar output plane a distance out_distance_m.

        Args:
            amplitude (float): Field amplitude of shape (Batch, Lambda, H W) or (Batch, Lambda, 1, R).
            phase (float): Field phase of shape (Batch, Lambda, H W) or (Batch, Lambda, 1, R).

        Returns:
            list: amplitude and phase with the same shape.
        """
        if "wavelength_set_m" in kwargs:
            raise ValueError(
                "The 'wavelength_set_m' is not expected in this model. "
                "Please use the propagator from dflat.propagation.propagators_legacy class that accepts wavelength as a forward input."
            )

        assert amplitude.shape == phase.shape, "amplitude and phase must be same shape."
        assert (
            len(self.wavelength_set) == amplitude.shape[-3] or amplitude.shape[-3] == 1
        ), "Wavelength dimensions don't match"
        if self.radial_symmetry:
            assert amplitude.shape[-2] == 1, "Radial flag requires 1D input not 2D."
            assert amplitude.shape[-1] == self.in_size[-1] // 2 + 1
        else:
            assert all(
                amplitude.shape[-2:] == self.in_size
            ), f"Input field size does not match init in_size {self.in_size}."

        device = "cuda" if torch.cuda.is_available() else "cpu"
        amplitude = (
            torch.tensor(amplitude, dtype=torch.float32).to(device)
            if not torch.is_tensor(amplitude)
            else amplitude.to(dtype=torch.float32)
        )
        phase = (
            torch.tensor(phase, dtype=torch.float32).to(device)
            if not torch.is_tensor(phase)
            else phase.to(dtype=torch.float32)
        )

        return checkpoint(self._forward, amplitude, phase, **kwargs)

    def _forward(self, amplitude, phase, **kwargs):
        # Upsample and pad the field prior to fourier-based propagation transformation
        amplitude, phase = self._regularize_field(amplitude, phase)

        # propagate by the fresnel method
        amplitude, phase = self.fresnel_transform(amplitude, phase)

        # Transform field back to the specified output grid
        amplitude, phase = self._resample_field(amplitude, phase)

        # Convert to 2D and return to final sensor size
        for i, (amp, ph) in enumerate(zip(amplitude, phase)):
            if self.radial_symmetry:
                amp = radial_2d_transform(amp.squeeze(-2))
                ph = radial_2d_transform_wrapped_phase(ph.squeeze(-2))

            phase[i] = resize_with_crop_or_pad(ph, *self.out_size, False)
            amplitude[i] = resize_with_crop_or_pad(amp, *self.out_size, False)

        amplitude = torch.cat(amplitude, axis=-3)
        phase = torch.cat(phase, axis=-3)

        return amplitude, phase

    def fresnel_transform(self, amplitude, phase):
        radial_symmetry = self.radial_symmetry
        dtype = amplitude[0].dtype
        device = amplitude[0].device
        torch_zero = self.torch_zero.to(dtype=dtype, device=device)

        for i, lam in enumerate(self.wavelength_set):
            in_quad_term = self.quad_term_in[i].to(dtype=dtype, device=device)
            transform_term = torch.complex(amplitude[i], torch_zero) * torch.exp(
                torch.complex(torch_zero, phase[i] + in_quad_term)
            )

            if radial_symmetry:
                fx = self.ang_fx[i].to(dtype=dtype, device=device)
                x = torch.squeeze(self.x[i].to(dtype=dtype, device=device))
                norm = self.norm[i].to(dtype=dtype, device=device)
                norm = torch.complex(norm, torch_zero)
                kr, wavefront = qdht(x, transform_term)
                wavefront = (
                    general_interp_regular_1d_grid(kr / 2 / np.pi, fx, wavefront) * norm
                )
            else:
                norm = self.norm[i].to(dtype=dtype, device=device)
                norm = torch.complex(norm, torch_zero)
                wavefront = fftshift(fft2(ifftshift(transform_term))) * norm

            # add the complex phase term on the output
            phase_add = self.out_phase[i].to(dtype=dtype, device=device)
            amplitude[i] = torch.abs(wavefront)
            phase[i] = torch.angle(
                wavefront * torch.exp(torch.complex(torch_zero, phase_add))
            )

        return amplitude, phase

    def _regularize_field(self, amplitude, phase):
        # Natural broadcasting of the wavelength dimension cannot be done for Fresnel case
        if amplitude.shape[-3] == 1:
            amplitude = amplitude.repeat(1, len(self.wavelength_set), 1, 1)
            phase = phase.repeat(1, len(self.wavelength_set), 1, 1)

        method = "nearest-exact"
        samplesM = self.calc_samplesM
        radial_symmetry = self.radial_symmetry

        ampl_list = []
        phase_list = []
        for i, _ in enumerate(self.wavelength_set):
            # Resample the input field via nearest neighbors interpolation
            # Nearest matches openCV's implementation but torch recommends using nearest-exact
            resize_to = (
                np.array([1, samplesM[i, 1] // 2 + 1])
                if radial_symmetry
                else samplesM[i]
            )
            resize_to = tuple([int(_) for _ in resize_to])
            ampl = F.interpolate(amplitude[:, i : i + 1], size=resize_to, mode=method)
            ph = F.interpolate(phase[:, i : i + 1], size=resize_to, mode=method)

            # Add padding -- this changes the output grid dx so we need to pad per wavelength
            # Thus what follows will be a jagged tensor (for now a list)
            padi = self.pad_in[i]
            paddings = (
                [0, padi[1], 0, 0, 0, 0, 0, 0]
                if radial_symmetry
                else [padi[1], padi[1], padi[0], padi[0], 0, 0, 0, 0]
            )
            ampl_list.append(F.pad(ampl, paddings, mode="constant", value=0))
            phase_list.append(F.pad(ph, paddings, mode="constant", value=0))

        return ampl_list, phase_list

    def _resample_field(self, amplitude, phase):
        nl = len(amplitude)
        for i in range(nl):
            scale = tuple(self.calc_out_dx[i] / self.out_resample_dx)
            if self.radial_symmetry:
                scale = (1, scale[-1])

            phase[i] = torch.atan2(
                F.interpolate(torch.sin(phase[i]), scale_factor=scale, mode="area"),
                F.interpolate(torch.cos(phase[i]), scale_factor=scale, mode="area"),
            )
            amplitude[i] = F.interpolate(amplitude[i], scale_factor=scale, mode="area")

        return amplitude, phase


class ASMPropagation(BaseFrequencySpace):
    """Angular Spectrum Method (ASM) for optical field propagation.

    This class implements the Angular Spectrum Method for propagating optical fields
    from an input plane to an output plane.

    Note:
        The ASM is suitable for a wide range of propagation distances and can handle
        non-paraxial cases more accurately than the Fresnel method. The output grid for ASM methods will always be forced to match the input grid.
        Consequently, in the back-end, we upsample and pad the input profile to match your target output grid. This affects memory and computation costs
        in a sometimes non-intuitive way for users.
    """

    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        wavelength_set_m,
        out_resample_dx_m=None,
        manual_upsample_factor=1,
        radial_symmetry=False,
        FFTPadFactor=1.0,
        verbose=False,
    ):
        """Initializes the propagation class.

        Args:
            in_size (list): input grid shape as [H, W].
            in_dx_m (list): input grid discretization (in meters) as [dy, dx]
            out_distance_m (float): output plane distance
            out_size (list): output grid shape as [H, W]
            out_dx_m (list): output grid discretization (in meters) as [dy, dx]
            wavelength_set_m (list): List of wavelengths (in meters) corresponding to the wavelength dimension stack in forward.
            out_resample_dx_m (list, optional): List of output grid discretizations to resample by area sum (area averaging for phase). This can be used to compute at a sub-pixel scale then return the integrated field on each pixel. Defaults to None.
            manual_upsample_factor (int, optional): Force factor to manually upsample (nearest neighbor) the input lens. This can improve fourier space calculation accuracy. Defaults to 1.
            FFTPadFactor (float, optional): Force a larger zero-pad factor during FFT used for frequency-space convolution. This is for developer debug/testing.
            radial_symmetry (bool, optional): Flag to use radial symmetry during calculations. Note that we expect radial field profiles to be passed in if True. Defaults to False.
            verbose (bool, optional): If True, prints information about the actual grid sizes etc that will be used in the back-end calculation. This may often be larger than user defined sizes due to fourier space rules. Defaults to False.
        """

        super().__init__(
            in_size,
            in_dx_m,
            out_distance_m,
            out_size,
            out_dx_m,
            wavelength_set_m,
            out_resample_dx_m,
            manual_upsample_factor,
            radial_symmetry,
        )
        self.verbose = verbose
        self._init_calc_params()
        self.FFTPadFactor = FFTPadFactor

    def _init_calc_params(self):
        # calc_in_dx should be smaller than the out_dx requested by the user.
        # This is because the output sampling grid will match input
        in_dx = self.in_dx
        out_dx = self.out_dx
        calc_in_dx = in_dx / self.manual_upsample_factor
        calc_in_dx = np.where(calc_in_dx < out_dx, calc_in_dx, out_dx)

        # Given the new calculation grid size, we should change the number of samples (odd) taken
        # at the input grid so that the originally data is correctly represented.
        in_length = in_dx * self.in_size
        calc_samplesM = np.rint(in_length / calc_in_dx)
        calc_samplesM = np.where(
            np.mod(calc_samplesM, 2) == 0, calc_samplesM + 1, calc_samplesM
        )
        calc_samplesM_r = calc_samplesM[-1] // 2
        self.calc_samplesM = calc_samplesM
        self.calc_samplesM_r = calc_samplesM_r

        # Update the calculation grid corresponding to the upsampled integer number of samples
        # For ASM, the output grid size is the same as the input so update this quantity
        self.calc_in_dx = in_length / calc_samplesM
        self.calc_out_dx = self.calc_in_dx

        # For the ASM case, we already checked that the in grid dx is smaller or equal to the sensor grid dx
        # Now we should pad to ensure that out fields are calculated on the entire sensing space requested by user
        desired_span = self.out_size * self.out_resample_dx
        current_span = self.calc_out_dx * calc_samplesM
        pad_in = np.rint(
            np.where(
                current_span < desired_span,
                (desired_span - current_span) / 2 / calc_in_dx,
                np.zeros_like(current_span),
            )
        ).astype(int)

        calc_samplesN = 2 * pad_in + calc_samplesM
        calc_samplesN_r = calc_samplesN[-1] // 2 + 1
        self.calc_samplesN = calc_samplesN
        self.calc_samplesN_r = calc_samplesN_r
        self.pad_in = pad_in

        if self.verbose:
            print("Initializing the Angular Spectrum Method")
            print(
                f"   - in_size: {self.in_size}, calc_in_size: {self.calc_samplesM}, padded_calc_in_size: {self.calc_samplesN}"
            )
            print(f"   - in_dx: {self.in_dx}, calc_in_dx: {self.calc_in_dx}")
            print(f"   - out_dx: {self.out_dx}, calc_out_dx: {self.calc_out_dx}")
            print(f"   - Resampling to grid size: {self.out_resample_dx}")

    def forward(self, amplitude, phase, **kwargs):
        """Propagates a complex field from an input plane to a planar output plane a distance out_distance_m.

        Args:
            amplitude (float): Field amplitude of shape (Batch, Lambda, H W) or (Batch, Lambda, 1, R).
            phase (float): Field phase of shape (Batch, Lambda, H W) or (Batch, Lambda, 1, R).

        Returns:
            list: amplitude and phase on the output grid. Shape of tensors same as passed in.
        """
        if "wavelength_set_m" in kwargs:
            raise ValueError(
                "The 'wavelength_set_m' is not expected in this model. "
                "Please use the propagator from dflat.propagation.propagators_legacy class that accepts wavelength as a forward input."
            )

        assert amplitude.shape == phase.shape, "amplitude and phase must be same shape."
        assert (
            len(self.wavelength_set) == amplitude.shape[-3] or amplitude.shape[-3] == 1
        ), "Wavelength dimensions don't match"
        if self.radial_symmetry:
            assert amplitude.shape[-2] == 1, "Radial flag requires 1D input not 2D."
            assert amplitude.shape[-1] == self.in_size[-1] // 2 + 1
        else:
            assert all(
                amplitude.shape[-2:] == self.in_size
            ), f"Input field size does not match init in_size {self.in_size}."

        device = "cuda" if torch.cuda.is_available() else "cpu"
        amplitude = (
            torch.tensor(amplitude, dtype=torch.float32).to(device)
            if not torch.is_tensor(amplitude)
            else amplitude.to(dtype=torch.float32)
        )
        phase = (
            torch.tensor(phase, dtype=torch.float32).to(device)
            if not torch.is_tensor(phase)
            else phase.to(dtype=torch.float32)
        )

        return checkpoint(self._forward, amplitude, phase, **kwargs)

    def _forward(self, amplitude, phase, **kwargs):
        # Upsample and pad the field prior to fft-based propagation transformation
        amplitude, phase = self._regularize_field(amplitude, phase)

        # propagate by the asm method
        amplitude, phase = self._ASM_transform(amplitude, phase)

        # Transform field back to the specified output grid and convert to 2D
        amplitude, phase = self._resample_field(amplitude, phase)
        if self.radial_symmetry:
            amplitude = radial_2d_transform(amplitude.squeeze(-2))
            phase = radial_2d_transform_wrapped_phase(phase.squeeze(-2))

        # Crop or pad with zeros to the final sensor size
        phase = resize_with_crop_or_pad(phase, *self.out_size, False)
        amplitude = resize_with_crop_or_pad(amplitude, *self.out_size, False)

        return amplitude, phase

    def _ASM_transform(self, amplitude, phase):
        init_shape = amplitude.shape
        dtype = amplitude.dtype
        device = amplitude.device
        torch_zero = torch.tensor([0.0], dtype=dtype).to(device)
        FFTPadFactor = self.FFTPadFactor
        wavelength_set = torch.tensor(self.wavelength_set, dtype=dtype, device=device)

        # Optionally zero pad the input before conducting a fourier transform
        padhalf = [int(n * self.FFTPadFactor) for n in init_shape[-2:]]
        paddings = (
            (0, padhalf[1], 0, 0, 0, 0, 0, 0)
            if self.radial_symmetry
            else (padhalf[1], padhalf[1], padhalf[0], padhalf[0], 0, 0, 0, 0)
        )
        amplitude = F.pad(amplitude, paddings, mode="constant", value=0)
        phase = F.pad(phase, paddings, mode="constant", value=0)
        transform_term = torch.complex(amplitude, torch_zero) * torch.exp(
            torch.complex(torch_zero, phase)
        )

        new_gs = [self.calc_samplesN[i] + 2 * padhalf[i] for i in range(2)]
        x, y = cart_grid(new_gs, self.calc_in_dx, self.radial_symmetry, dtype, device)

        # Define the transfer function via FT of the Sommerfield solution
        # note: we have decided to ingore the physical, depth dependent energy scaling here (and in the fresnel method)
        rarray = torch.sqrt(self.out_distance**2 + x**2 + y**2)[None, None, :, :]
        angular_wavenumber = 2 * np.pi / wavelength_set
        angular_wavenumber = angular_wavenumber[None, :, None, None]
        h = (
            torch.complex(1 / 2 / np.pi * self.out_distance / rarray**2, torch_zero)
            * torch.complex(1 / rarray, -1 * angular_wavenumber)
            * torch.exp(torch.complex(torch_zero, angular_wavenumber * rarray))
        )

        # Compute the angular decomposition and frequency space transfer function
        if self.radial_symmetry:
            kr, angular_spectrum = qdht(torch.squeeze(x), transform_term)
            kr, H = qdht(torch.squeeze(x), h)
        else:
            angular_spectrum = fft2(ifftshift(transform_term))
            H = fft2(ifftshift(h))

        H = torch.exp(torch.complex(torch_zero, torch.angle(H)))
        transform_term = angular_spectrum * H

        # Propagation by multiplying angular decomposition with H then taking the inverse transform
        if self.radial_symmetry:
            r2, outputwavefront = iqdht(kr, transform_term)
            outputwavefront = general_interp_regular_1d_grid(
                r2, torch.squeeze(x), outputwavefront
            )
        else:
            outputwavefront = fftshift(ifft2(transform_term))

        # Crop to remove the padding used in the calculation
        # Radial symmetry needs asymmetric cropping not central crop
        outputwavefront = resize_with_crop_or_pad(
            outputwavefront, *init_shape[-2:], self.radial_symmetry
        )

        return torch.abs(outputwavefront), torch.angle(outputwavefront)

    def _resample_field(self, amplitude, phase):
        scale = tuple(self.calc_out_dx / self.out_resample_dx)
        if self.radial_symmetry:
            scale = (1, scale[-1])

        phase = torch.atan2(
            F.interpolate(torch.sin(phase), scale_factor=scale, mode="area"),
            F.interpolate(torch.cos(phase), scale_factor=scale, mode="area"),
        )
        amplitude = F.interpolate(amplitude, scale_factor=scale, mode="area")
        return amplitude, phase

    def _regularize_field(self, amplitude, phase):
        # Resample the input field via nearest neighbors interpolation
        # Nearest matches openCV's implementation but torch recommends using nearest-exact
        method = "nearest-exact"
        samplesM = self.calc_samplesM
        radial_symmetry = self.radial_symmetry
        pad_in = self.pad_in

        resize_to = np.array([1, samplesM[1] // 2 + 1]) if radial_symmetry else samplesM
        resize_to = tuple([int(_) for _ in resize_to])
        amplitude = F.interpolate(amplitude, size=resize_to, mode=method)
        phase = F.interpolate(phase, size=resize_to, mode=method)

        # Add padding
        paddings = (
            [0, pad_in[1], 0, 0, 0, 0, 0, 0]
            if radial_symmetry
            else [pad_in[1], pad_in[1], pad_in[0], pad_in[0], 0, 0, 0, 0]
        )
        amplitude = F.pad(amplitude, paddings, mode="constant", value=0)
        phase = F.pad(phase, paddings, mode="constant", value=0)

        return amplitude, phase


class PointSpreadFunction(nn.Module):
    """Calculates the Point Spread Function (PSF) for an optical system.

    This class uses either the Angular Spectrum Method (ASM) or Fresnel propagation
    to calculate the PSF of an optical system for given input fields and point source locations.

    Note: Normalize_to_aperture in the forward argument enables re-normalization of the output PSF relative to the total energy incident at the input plane.
    """

    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        wavelength_set_m,
        out_resample_dx_m=None,
        manual_upsample_factor=1,
        radial_symmetry=False,
        diffraction_engine="ASM",
        verbose=False,
    ):
        """Initializes the point-spread function class.

        Args:
            in_size (list): input grid shape as [H, W].
            in_dx_m (list): input grid discretization (in meters) as [dy, dx]
            out_distance_m (float): output plane distance
            out_size (list): output grid shape as [H, W]
            out_dx_m (list): output grid discretization (in meters) as [dy, dx]
            wavelength_set_m (list): List of wavelengths (in meters) corresponding to the wavelength dimension stack in forward.
            out_resample_dx_m (list, optional): List of output grid discretizations to resample by area sum (area averaging for phase). This can be used to compute at a sub-pixel scale then return the integrated field on each pixel. Defaults to None.
            manual_upsample_factor (int, optional): Force factor to manually upsample (nearest neighbor) the input lens. This can improve fourier space calculation accuracy. Defaults to 1.
            radial_symmetry (bool, optional): Flag to use radial symmetry during calculations. Note that we expect radial field profiles to be passed in if True. Defaults to False.
            verbose (bool, optional): If True, prints information about the actual grid sizes etc that will be used in the back-end calculation. This may often be larger than user defined sizes due to fourier space rules. Defaults to False.
        """
        super().__init__()

        assert isinstance(
            diffraction_engine, str
        ), "diffraction engine must be a string"
        assert isinstance(
            wavelength_set_m, (list, np.ndarray)
        ), "wavelengths must be passed as a list."

        diffraction_engine = diffraction_engine.lower()
        assert diffraction_engine in [
            "asm",
            "fresnel",
        ], "Diffraction engine must be either 'asm' or 'fresnel'"

        propagator = (
            ASMPropagation if diffraction_engine == "asm" else FresnelPropagation
        )
        self.propagator = propagator(
            in_size,
            in_dx_m,
            out_distance_m,
            out_size,
            out_dx_m,
            wavelength_set_m,
            out_resample_dx_m,
            manual_upsample_factor,
            radial_symmetry,
            verbose=verbose,
        )

        self.rescale = 1e6  # Convert m to um
        self.wavelength_set = torch.tensor(wavelength_set_m) * self.rescale
        self.in_size = in_size
        self.in_dx_m = in_dx_m
        self.out_resample_dx = (
            out_dx_m if out_resample_dx_m is None else out_resample_dx_m
        )
        self.radial_symmetry = radial_symmetry

    def forward(
        self,
        amplitude,
        phase,
        ps_locs_m,
        aperture=None,
        normalize_to_aperture=True,
        **kwargs,
    ):
        """Computes the pont-spread function for teh amplitude and phase profile given a list of point-source locations.

        Args:
            amplitude (tensor): Lens amplitude of shape [Batch L H W], where L may equal 1 for broadcasting.
            phase (tensor): Lens phase of shape [Batch L H W], where L may equal 1 for broadcasting.
            ps_locs_m (tensor): Array point-source locations of shape [N x 3] where each column corresponds to Y, X, Depth
            aperture (Tensor, optional): Field aperture applied on the lens the same rank as amplitude
                and with the same H W dimensions. Defaults to None.
            normalize_to_aperture (bool, optional): If true the energy in the PSF will be normalized to the total energy
                incident on the optic/aperture. Defaults to True.

        Returns:
            List: Returns point-spread function intensity and phase of shape [Batch Num_point_sources Lambda H W].
        """
        if "wavelength_set_m" in kwargs:
            raise ValueError(
                "The 'wavelength_set_m' is not expected in this model. "
                "Please use the propagator from dflat.propagation.propagators_legacy class that accepts wavelength as a forward input."
            )
        assert (
            amplitude.shape == phase.shape
        ), "ampl and phase should be the same shape."
        assert len(amplitude.shape) >= 4, "field profile must be at least rank 4"
        assert (
            len(self.wavelength_set) == amplitude.shape[-3] or amplitude.shape[-3] == 1
        ), "Mismatch in number of simulation wavelengths and wavelength dimension of profile."
        ps_locs_m = np.array(ps_locs_m) if not torch.is_tensor(ps_locs_m) else ps_locs_m
        assert len(ps_locs_m.shape) == 2
        assert ps_locs_m.shape[-1] == 3
        if self.radial_symmetry:
            assert amplitude.shape[-2] == 1, "Radial flag requires 1D input not 2D."
            assert amplitude.shape[-1] == self.in_size[-1] // 2 + 1
        else:
            assert (
                list(amplitude.shape[-2:]) == self.in_size
            ), f"Input field size does not match init in_size {self.in_size}."
        if aperture is not None:
            assert aperture.shape[-2:] == amplitude.shape[-2:]
            assert len(aperture.shape) == len(amplitude.shape)

        amplitude = (
            torch.tensor(amplitude, dtype=torch.float32)
            if not torch.is_tensor(amplitude)
            else amplitude.to(dtype=torch.float32)
        )
        phase = (
            torch.tensor(phase, dtype=torch.float32)
            if not torch.is_tensor(phase)
            else phase.to(dtype=torch.float32)
        )
        if aperture is not None:
            aperture = (
                torch.tensor(aperture, dtype=torch.float32, device=amplitude.device)
                if not torch.is_tensor(aperture)
                else aperture.to(dtype=torch.float32)
            )

        # Reshape B P L H  W to B L H W
        init_shape = amplitude.shape
        amplitude = amplitude.view(-1, *init_shape[-3:])
        phase = phase.view(-1, *init_shape[-3:])

        # Apply incident wavefront
        N = amplitude.shape[0]
        Z = len(ps_locs_m)
        amplitude, phase = self._incident_wavefront(amplitude, phase, ps_locs_m)
        if aperture is not None:
            amplitude = amplitude * aperture

        # Propagate field to sensor
        amplitude = rearrange(amplitude, "Z N L H W -> (N Z) L H W")
        phase = rearrange(phase, "Z N L H W -> (N Z) L H W")
        amplitude, phase = self.propagator(amplitude, phase)
        amplitude = rearrange(amplitude, "(N Z) L H W -> N Z L H W", N=N, Z=Z)
        phase = rearrange(phase, "(N Z) L H W -> N Z L H W", N=N, Z=Z)

        # Return to the original shape before returning
        out_shape = amplitude.shape
        amplitude = amplitude.view(*init_shape[:-3], *out_shape[-4:])
        phase = phase.view(*init_shape[:-3], *out_shape[-4:])

        amplitude = amplitude**2
        normalization = (
            np.prod(self.out_resample_dx) / self._aperture_energy(aperture)
        ).to(dtype=amplitude.dtype, device=amplitude.device)
        if normalize_to_aperture:
            return amplitude * normalization, phase
        else:
            return amplitude, phase

    def _incident_wavefront(self, amplitude, phase, ps_locs_m):
        # Z N L H W
        # Expand dimension to hold point_sources
        device = amplitude.device
        amplitude = amplitude[None].to(dtype=torch.float64)
        phase = phase[None].to(dtype=torch.float64)
        torch_zero = torch.tensor([0.0], dtype=torch.float64, device=device)
        wavelength_set = self.wavelength_set.to(dtype=torch.float64, device=device)

        k = 2 * np.pi / wavelength_set
        k = k[None, None, :, None, None]

        ps_locs = (
            torch.tensor(
                np.array(ps_locs_m) * self.rescale, dtype=torch.float64, device=device
            )
            if not torch.is_tensor(ps_locs_m)
            else ps_locs_m.to(dtype=torch.float64, device=device) * self.rescale
        )
        psx = ps_locs[:, 0][:, None, None, None, None]
        psy = ps_locs[:, 1][:, None, None, None, None]
        psz = ps_locs[:, 2][:, None, None, None, None]

        x, y = cart_grid(
            self.in_size,
            list(np.array(self.in_dx_m) * self.rescale),
            self.radial_symmetry,
            torch.float64,
            device,
        )
        x = x[None, None, None]
        y = y[None, None, None]
        distance = torch.sqrt((x - psx) ** 2 + (y - psy) ** 2 + psz**2)  # Z x H x W

        wavefront = torch.complex(amplitude, torch_zero) * torch.exp(
            torch.complex(torch_zero, phase + k * distance)
        )

        return torch.abs(wavefront).to(dtype=torch.float32), torch.angle(wavefront).to(
            dtype=torch.float32
        )

    @torch.no_grad()
    def _aperture_energy(self, aperture):
        in_size = self.in_size
        sz = [
            1,
            1,
            1 if self.radial_symmetry else in_size[-2],
            in_size[-1] // 2 + 1 if self.radial_symmetry else in_size[-1],
        ]

        fieldblock2d = aperture if aperture is not None else torch.ones(size=sz)
        if self.radial_symmetry:
            fieldblock2d = radial_2d_transform(fieldblock2d.squeeze(-2))

        in_energy = torch.sum(
            fieldblock2d**2 * np.prod(self.in_dx_m),
            dim=(-1, -2),
            keepdim=True,
        )[None]

        return in_energy
