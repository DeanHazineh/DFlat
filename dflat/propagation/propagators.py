import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fftshift, ifftshift, fft2, ifft2
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
    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        out_resample_dx_m,
        manual_upsample_factor,
        radial_symmetry,
    ):
        super().__init__()
        # Quick checks on the inputs
        out_resample_dx_m = out_dx_m if out_resample_dx_m is None else out_resample_dx_m
        assert isinstance(radial_symmetry, bool)
        assert manual_upsample_factor >= 1
        assert isinstance(out_distance_m, float)
        for obj in [in_size, in_dx_m, out_size, out_dx_m, out_resample_dx_m]:
            assert len(obj) == 2

        # Move to attributes and convert lists to numpy arrays
        self.in_size = np.array(in_size).astype(int)
        self.in_dx_m = np.array(in_dx_m)
        self.out_distance_m = out_distance_m
        self.out_size = np.array(out_size).astype(int)
        self.out_dx_m = np.array(out_dx_m)
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


class ASMPropagation(BaseFrequencySpace):
    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        out_resample_dx_m=None,
        manual_upsample_factor=1,
        radial_symmetry=False,
        FFTPadFactor=1.0,
    ):
        super().__init__(
            in_size,
            in_dx_m,
            out_distance_m,
            out_size,
            out_dx_m,
            out_resample_dx_m,
            manual_upsample_factor,
            radial_symmetry,
        )
        self._init_calc_params()
        self.FFTPadFactor = FFTPadFactor

    def _init_calc_params(self):
        in_dx = self.in_dx
        out_dx = self.out_dx

        # calc_in_dx should be smaller than the out_dx requested by the user.
        # This is because the output sampling grid will match input
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
        self.calc_in_dx = in_length / calc_samplesM

        # For ASM, the output grid size is the same as the input so update this quantity
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

    def forward(self, amplitude, phase, wavelength_set_m):
        """Propagates a complex field from an input plane to a planar output plane a distance out_distance_m.

        Args:
            amplitude (float): Field amplitude of shape (Batch, Lambda, *in_size) or (Batch, Lambda, 1, in_size_r).
            phase (float): Field phase of shape (Batch, Lambda, *in_size) or (Batch, Lambda, 1, in_size_r).
        """

        assert amplitude.shape == phase.shape, "amplitude and phase must be same shape."
        assert (
            len(wavelength_set_m) == amplitude.shape[-3] or amplitude.shape[-3] == 1
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
        wavelength_set = (
            torch.tensor(
                np.array(wavelength_set_m) * self.rescale, dtype=amplitude.dtype
            ).to(amplitude.device)
            if not torch.is_tensor(wavelength_set_m)
            else wavelength_set_m.to(dtype=amplitude.dtype)
        )

        # Upsample and pad the field prior to fft-based propagation transformation
        amplitude, phase = self._regularize_field(amplitude, phase)

        # propagate by the asm method
        amplitude, phase = self.ASM_transform(amplitude, phase, wavelength_set)

        # Transform field back to the specified output grid and convert to 2D
        amplitude, phase = self._resample_field(amplitude, phase)
        if self.radial_symmetry:
            amplitude = radial_2d_transform(amplitude.squeeze(-2))
            phase = radial_2d_transform_wrapped_phase(phase.squeeze(-2))

        # Crop or pad with zeros to the final sensor size
        phase = resize_with_crop_or_pad(phase, *self.out_size, False)
        amplitude = resize_with_crop_or_pad(amplitude, *self.out_size, False)

        return amplitude, phase

    def ASM_transform(self, amplitude, phase, wavelength_set):
        init_shape = amplitude.shape
        dtype = amplitude.dtype
        device = amplitude.device
        torch_zero = torch.tensor([0.0], dtype=dtype).to(device)
        FFTPadFactor = self.FFTPadFactor

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
        # Reinterpolate the phase with area averaging and the amplitude with area sum
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


# Note that the best way to do this would be to compute an upsample factor for each wavelength but
# this substantially increases the codes complexity and in most use cases it is not needed. We will for
# now just let users manually define a suitable upsample amount for all cases if they are in the regime
# where this matters.
class FresnelPropagation(BaseFrequencySpace):
    def __init__(
        self,
        in_size,
        in_dx_m,
        out_distance_m,
        out_size,
        out_dx_m,
        out_resample_dx_m=None,
        manual_upsample_factor=1,
        radial_symmetry=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_size,
            in_dx_m,
            out_distance_m,
            out_size,
            out_dx_m,
            out_resample_dx_m,
            manual_upsample_factor,
            radial_symmetry,
        )
        self._init_calc_params()

    def forward(self, amplitude, phase, wavelength_set_m):
        """Propagates a complex field from an input plane to a planar output plane a distance out_distance_m.

        Args:
            amplitude (float): Field amplitude of shape (Batch, Lambda, *in_size) or (Batch, Lambda, 1, in_size_r).
            phase (float): Field phase of shape (Batch, Lambda, *in_size) or (Batch, Lambda, 1, in_size_r).
        """

        assert amplitude.shape == phase.shape, "amplitude and phase must be same shape."
        assert (
            len(wavelength_set_m) == amplitude.shape[-3] or amplitude.shape[-3] == 1
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
        wavelength_set = (
            torch.tensor(
                np.array(wavelength_set_m) * self.rescale, dtype=amplitude.dtype
            ).to(amplitude.device)
            if not torch.is_tensor(wavelength_set_m)
            else wavelength_set_m.to(dtype=amplitude.dtype) * self.rescale
        )

        # Upsample and pad the field prior to fourier-based propagation transformation
        amplitude, phase = self._regularize_field(amplitude, phase, wavelength_set)

        # propagate by the fresnel method
        amplitude, phase = self.fresnel_transform(amplitude, phase, wavelength_set)

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

    def fresnel_transform(self, amplitude, phase, wavelength_set):
        radial_symmetry = self.radial_symmetry
        dtype = amplitude[0].dtype
        device = amplitude[0].device
        torch_zero = torch.tensor(0.0, dtype=dtype, device=device)

        for i, lam in enumerate(wavelength_set):
            # Define grid and compute quadratic term
            insize = amplitude[i].shape[-2:]
            x, y = torch.meshgrid(
                torch.arange(0, insize[-1], dtype=dtype, device=device),
                torch.arange(0, insize[-2], dtype=dtype, device=device),
                indexing="xy",
            )
            if not radial_symmetry:
                x = x - (x.shape[-1] // 2)
                y = y - (y.shape[-2] // 2)

            quadratic_term = (
                np.pi
                / lam
                / self.out_distance
                * ((x * self.calc_in_dx[-1]) ** 2 + (y * self.calc_in_dx[-2]) ** 2)
            )

            transform_term = torch.complex(amplitude[i], torch_zero) * torch.exp(
                torch.complex(torch_zero, phase[i] + quadratic_term)
            )

            # propagate with qdht or fft2
            if radial_symmetry:
                ang_fx = torch.arange(
                    0,
                    1 / 2 / self.calc_in_dx[-1],
                    1 / self.calc_in_dx[-1] / self.calc_samplesN[i, -1],
                    dtype=dtype,
                    device=device,
                )
                norm = torch.tensor(
                    1
                    / np.sqrt(
                        (
                            np.prod(self.calc_in_dx)
                            * np.prod(self.calc_out_dx[i])
                            * np.prod(self.calc_samplesN[i])
                        )
                    ),
                    dtype=dtype,
                    device=device,
                )
                kr, wavefront = qdht(
                    torch.squeeze(x * self.calc_in_dx[-1]), transform_term
                )
                wavefront = general_interp_regular_1d_grid(
                    kr / 2 / np.pi, ang_fx, wavefront
                ) * torch.complex(norm, torch_zero)
            else:
                norm = torch.tensor(
                    np.sqrt(
                        np.prod(self.calc_in_dx)
                        / np.prod(self.calc_out_dx[i])
                        / np.prod(self.calc_samplesN[i])
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                wavefront = fftshift(fft2(ifftshift(transform_term))) * torch.complex(
                    norm, torch_zero
                )

            # add the complex phase term on the output
            phase_add = (
                2
                * np.pi
                / lam
                * (
                    self.out_distance
                    + (
                        (x * self.calc_out_dx[i, -1]) ** 2
                        + (y * self.calc_out_dx[i, -2]) ** 2
                    )
                    / 2
                    / self.out_distance
                )
            )

            amplitude[i] = torch.abs(wavefront)
            phase[i] = torch.angle(
                wavefront * torch.exp(torch.complex(torch_zero, phase_add))
            )

        return amplitude, phase

    def _resample_field(self, amplitude, phase):
        # Reinterpolate the phase with area averaging and the amplitude with area sum
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

    def _regularize_field(self, amplitude, phase, wavelength_set):
        # Natural broadcasting of the wavelength dimension cannot be done for Fresnel case
        if amplitude.shape[-3] == 1:
            amplitude = amplitude.repeat(1, len(wavelength_set), 1, 1)
            phase = phase.repeat(1, len(wavelength_set), 1, 1)

        # Resample the input field via nearest neighbors interpolation
        # Nearest matches openCV's implementation but torch recommends using nearest-exact
        method = "nearest-exact"
        samplesM = self.calc_samplesM
        radial_symmetry = self.radial_symmetry
        resize_to = np.array([1, samplesM[1] // 2 + 1]) if radial_symmetry else samplesM
        resize_to = tuple([int(_) for _ in resize_to])
        amplitude = F.interpolate(amplitude, size=resize_to, mode=method)
        phase = F.interpolate(phase, size=resize_to, mode=method)

        # Add padding -- this changes the output grid dx so we need to pad per wavelength
        # Thus what follows will be a jagged tensor (for now a list)
        estN = np.ceil(
            wavelength_set[:, None].cpu().numpy()
            * self.out_distance
            / self.out_dx[None, :]
            / self.calc_in_dx
        )
        estN = np.where(np.mod(estN, 2) == 0, estN + 1, estN)
        estN = np.where(estN < self.calc_samplesM, self.calc_samplesM, estN)
        estN = np.where(estN < self.calc_samplesN, self.calc_samplesN, estN)

        pad_in = (estN - self.calc_samplesM) / 2
        pad_in = pad_in.astype(int)

        # Now redefine the exact output calculation grid
        self.calc_samplesN = estN
        self.calc_out_dx = (
            wavelength_set[:, None].cpu().numpy()
            * self.out_distance
            / self.calc_in_dx
            / estN
        )

        ampl_list = []
        phase_list = []
        for i, _ in enumerate(wavelength_set):
            padi = pad_in[i]
            paddings = (
                [0, padi[1], 0, 0, 0, 0, 0, 0]
                if radial_symmetry
                else [padi[1], padi[1], padi[0], padi[0], 0, 0, 0, 0]
            )
            ampl_list.append(
                F.pad(amplitude[:, i : i + 1], paddings, mode="constant", value=0)
            )
            phase_list.append(
                F.pad(phase[:, i : i + 1], paddings, mode="constant", value=0)
            )

        return ampl_list, phase_list

    def _init_calc_params(self):
        # Compute the input grid and upsampled size when we upsample the input field
        in_dx = self.in_dx

        calc_in_dx = in_dx / self.manual_upsample_factor
        in_length = in_dx * self.in_size
        calc_samplesM = np.rint(in_length / calc_in_dx)
        calc_samplesM = np.where(
            np.mod(calc_samplesM, 2) == 0, calc_samplesM + 1, calc_samplesM
        )
        calc_samplesM_r = calc_samplesM[-1] // 2
        self.calc_samplesM = calc_samplesM
        self.calc_samplesM_r = calc_samplesM_r

        # Update the calculation grid corresponding to the upsampled integer number of samples
        self.calc_in_dx = in_length / calc_samplesM

        # The fourier transform implies that the number of samples in the output grid will match the
        # input grid. To compute on the full output_plane, we should zero pad if needed in input (keep odd)
        calc_samplesN = np.where(
            self.out_size > self.in_size, self.out_size, self.in_size
        )
        self.calc_samplesN = np.where(
            np.mod(calc_samplesN, 2) == 0, calc_samplesN + 1, calc_samplesN
        )

        # For the Fresnel Engine, the output field grid size is tuned by zero-padding the input field
        # We should then zero-pad more, dependent on wavelength, to get the output pixel size. This will
        # be computed on the fly in forward call


# Initially started playing around with calculating the correct upsample factor to avoid cropped
# field calculations. It utlimately is not a trivial tweak given the current class structure
# class FresnelPropagation(BaseFrequencySpace):
#     def __init__(
#         self,
#         in_size,
#         in_dx_m,
#         out_distance_m,
#         out_size,
#         out_dx_m,
#         out_resample_dx_m=None,
#         manual_upsample_factor=1,
#         radial_symmetry=False,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(
#             in_size,
#             in_dx_m,
#             out_distance_m,
#             out_size,
#             out_dx_m,
#             out_resample_dx_m,
#             manual_upsample_factor,
#             radial_symmetry,
#         )
#         self._init_calc_params()

#     def forward(self, amplitude, phase, wavelength_set_m):
#         """Propagates a complex field from an input plane to a planar output plane a distance out_distance_m.

#         Args:
#             amplitude (float): Field amplitude of shape (Batch, Lambda, *in_size) or (Batch, Lambda, 1, in_size_r).
#             phase (float): Field phase of shape (Batch, Lambda, *in_size) or (Batch, Lambda, 1, in_size_r).
#         """

#         assert amplitude.shape == phase.shape, "amplitude and phase must be same shape."
#         assert (
#             len(wavelength_set_m) == amplitude.shape[-3] or amplitude.shape[-3] == 1
#         ), "Wavelength dimensions don't match"
#         if self.radial_symmetry:
#             assert amplitude.shape[-2] == 1, "Radial flag requires 1D input not 2D."
#             assert amplitude.shape[-1] == self.in_size[-1] // 2 + 1
#         else:
#             assert all(
#                 amplitude.shape[-2:] == self.in_size
#             ), f"Input field size does not match init in_size {self.in_size}."

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         amplitude = (
#             torch.tensor(amplitude, dtype=torch.float32).to(device)
#             if not torch.is_tensor(amplitude)
#             else amplitude.to(dtype=torch.float32)
#         )
#         phase = (
#             torch.tensor(phase, dtype=torch.float32).to(device)
#             if not torch.is_tensor(phase)
#             else phase.to(dtype=torch.float32)
#         )
#         wavelength_set = (
#             torch.tensor(
#                 np.array(wavelength_set_m) * self.rescale, dtype=amplitude.dtype
#             ).to(amplitude.device)
#             if not torch.is_tensor(wavelength_set_m)
#             else wavelength_set_m.to(dtype=amplitude.dtype)
#         )

#         # Upsample and pad the field prior to fourier-based propagation transformation
#         amplitude, phase = self._regularize_field(amplitude, phase, wavelength_set)

#         # propagate by the fresnel method
#         amplitue, phase = self.fresnel_transform(amplitude, phase, wavelength_set)

#         # Transform field back to the specified output grid
#         amplitude, phase = self._resample_field(amplitude, phase)

#         # Convert to 2D and return to final sensor size
#         for i, (amp, ph) in enumerate(zip(amplitude, phase)):
#             if self.radial_symmetry:
#                 amp = radial_2d_transform(amp.squeeze(-2))
#                 ph = radial_2d_transform_wrapped_phase(ph.squeeze(-2))

#             phase[i] = resize_with_crop_or_pad(ph, *self.out_size, False)
#             amplitude[i] = resize_with_crop_or_pad(amp, *self.out_size, False)

#         amplitude = torch.cat(amplitude, axis=-3)
#         phase = torch.cat(phase, axis=-3)

#         return amplitude, phase

#     def fresnel_transform(self, amplitude, phase, wavelength_set):
#         radial_symmetry = self.radial_symmetry
#         dtype = amplitude[0].dtype
#         device = amplitude[0].device
#         torch_zero = torch.tensor(0.0, dtype=dtype, device=device)

#         for i, lam in enumerate(wavelength_set):
#             # Define grid and compute quadratic term
#             insize = amplitude[i].shape[-2:]
#             x, y = torch.meshgrid(
#                 torch.arange(0, insize[-1], dtype=dtype, device=device),
#                 torch.arange(0, insize[-2], dtype=dtype, device=device),
#                 indexing="xy",
#             )
#             if not radial_symmetry:
#                 x = x - (x.shape[-1] // 2)
#                 y = y - (y.shape[-2] // 2)

#             quadratic_term = (
#                 np.pi
#                 / lam
#                 / self.out_distance
#                 * ((x * self.calc_in_dx[-1]) ** 2 + (y * self.calc_in_dx[-2]) ** 2)
#             )

#             transform_term = torch.complex(amplitude[i], torch_zero) * torch.exp(
#                 torch.complex(torch_zero, phase[i] + quadratic_term)
#             )

#             # propagate with qdht or fft2
#             if radial_symmetry:
#                 ang_fx = torch.arange(
#                     0,
#                     1 / 2 / self.calc_in_dx[-1],
#                     1 / self.calc_in_dx[-1] / self.calc_samplesN[i, -1],
#                     dtype=dtype,
#                     device=device,
#                 )
#                 norm = torch.tensor(
#                     1
#                     / np.sqrt(
#                         (
#                             np.prod(self.calc_in_dx)
#                             * np.prod(self.calc_out_dx[i])
#                             * np.prod(self.calc_samplesN[i])
#                         )
#                     ),
#                     dtype=dtype,
#                     device=device,
#                 )
#                 kr, wavefront = qdht(
#                     torch.squeeze(x * self.calc_in_dx[-1]), transform_term
#                 )
#                 wavefront = general_interp_regular_1d_grid(
#                     kr / 2 / np.pi, ang_fx, wavefront
#                 ) * torch.complex(norm, torch_zero)
#             else:
#                 norm = torch.tensor(
#                     np.sqrt(
#                         np.prod(self.calc_in_dx)
#                         / np.prod(self.calc_out_dx[i])
#                         / np.prod(self.calc_samplesN[i])
#                     ),
#                     dtype=torch.float32,
#                     device=device,
#                 )
#                 wavefront = fftshift(fft2(ifftshift(transform_term))) * torch.complex(
#                     norm, torch_zero
#                 )

#             # add the complex phase term on the output
#             phase_add = (
#                 2
#                 * np.pi
#                 / lam
#                 * (
#                     self.out_distance
#                     + (
#                         (x * self.calc_out_dx[i, -1]) ** 2
#                         + (y * self.calc_out_dx[i, -2]) ** 2
#                     )
#                     / 2
#                     / self.out_distance
#                 )
#             )

#             amplitude[i] = torch.abs(wavefront)
#             phase[i] = torch.angle(
#                 wavefront * torch.exp(torch.complex(torch_zero, phase_add))
#             )

#         return amplitude, phase

#     def _resample_field(self, amplitude, phase):
#         # Reinterpolate the phase with area averaging and the amplitude with area sum
#         nl = len(amplitude)
#         for i in range(nl):
#             scale = tuple(self.calc_out_dx[i] / self.out_resample_dx)
#             if self.radial_symmetry:
#                 scale = (1, scale[-1])

#             phase[i] = torch.atan2(
#                 F.interpolate(torch.sin(phase[i]), scale_factor=scale, mode="area"),
#                 F.interpolate(torch.cos(phase[i]), scale_factor=scale, mode="area"),
#             )
#             amplitude[i] = F.interpolate(amplitude[i], scale_factor=scale, mode="area")

#         return amplitude, phase

#     def _regularize_field(self, amplitude, phase, wavelength_set):
#         # Natural broadcasting of the wavelength dimension cannot be done for Fresnel case
#         if amplitude.shape[-3] == 1:
#             amplitude = amplitude.repeat(1, len(wavelength_set), 1, 1)
#             phase = phase.repeat(1, len(wavelength_set), 1, 1)

#         # Resample the input field via nearest neighbors interpolation
#         # Nearest matches openCV's implementation but torch recommends using nearest-exact
#         method = "nearest-exact"

#         # We actually should enforce a new manual upsampling to ensure full field calculations
#         # If you remove this, you can get cropped outputs
#         force_factor = np.round(
#             self.out_dx_m[None]
#             * self.out_size[None]
#             * self.in_dx_m[None]
#             / wavelength_set[:, None, None].cpu().numpy()
#             * self.rescale
#             / self.out_distance_m
#         )
#         manual_upsample = [self.manual_upsample_factor, self.manual_upsample_factor]
#         in_length = self.in_dx * self.in_size
#         ampl_list = []
#         phase_list = []
#         for i in range(len(wavelength_set)):
#             # For simplicity, just use one upsample factor
#             use_upsample = np.maximum(force_factor[i], manual_upsample).max()
#             calc_in_dx = self.in_dx / use_upsample
#             calc_samplesM = np.rint(in_length / calc_in_dx)

#             # Update the calculation grid corresponding to the upsampled integer number of samples
#             calc_in_dx = in_length / calc_samplesM
#             calc_samplesN = np.where(
#                 self.out_size > self.in_size, self.out_size, self.in_size
#             )

#             radial_symmetry = self.radial_symmetry
#             resize_to = (
#                 np.array([1, calc_samplesM[1] // 2 + 1])
#                 if radial_symmetry
#                 else calc_samplesM
#             )
#             resize_to = tuple([int(_) for _ in resize_to])
#             amplitude = F.interpolate(amplitude, size=resize_to, mode=method)
#             phase = F.interpolate(phase, size=resize_to, mode=method)

#             # Add padding -- this changes the output grid dx so we need to pad per wavelength
#             # Thus what follows will be a jagged tensor (for now a list)
#             estN = np.ceil(
#                 wavelength_set[:, None].cpu().numpy()
#                 * self.out_distance
#                 / self.out_dx[None, :]
#                 / calc_in_dx
#             )
#             estN = np.where(np.mod(estN, 2) == 0, estN + 1, estN)

#             estN = np.where(estN < calc_samplesM, calc_samplesM, estN)
#             estN = np.where(estN < calc_samplesN, calc_samplesN, estN)
#             pad_in = (estN - self.calc_samplesM) / 2
#             pad_in = pad_in.astype(int)

#             # Now redefine the exact output calculation grid
#             self.calc_out_dx = (
#                 wavelength_set[:, None].cpu().numpy()
#                 * self.out_distance
#                 / calc_in_dx
#                 / estN
#             )

#             padi = pad_in[i]
#             paddings = (
#                 [0, padi[1], 0, 0, 0, 0, 0, 0]
#                 if radial_symmetry
#                 else [padi[1], padi[1], padi[0], padi[0], 0, 0, 0, 0]
#             )
#             ampl_list.append(
#                 F.pad(amplitude[:, i : i + 1], paddings, mode="constant", value=0)
#             )
#             phase_list.append(
#                 F.pad(phase[:, i : i + 1], paddings, mode="constant", value=0)
#             )

#         return ampl_list, phase_list

#     def _init_calc_params(self):
#         # # Compute the input grid and upsampled size when we upsample the input field
#         # in_dx = self.in_dx
#         # in_length = in_dx * self.in_size

#         # calc_in_dx = in_dx / self.manual_upsample_factor
#         # calc_samplesM = np.rint(in_length / calc_in_dx)
#         # calc_samplesM = np.where(
#         #     np.mod(calc_samplesM, 2) == 0, calc_samplesM + 1, calc_samplesM
#         # )
#         # calc_samplesM_r = calc_samplesM[-1] // 2
#         # self.calc_samplesM = calc_samplesM
#         # self.calc_samplesM_r = calc_samplesM_r

#         # # Update the calculation grid corresponding to the upsampled integer number of samples
#         # self.calc_in_dx = in_length / calc_samplesM

#         # # The fourier transform implies that the number of samples in the output grid will match the
#         # # input grid. To compute on the full output_plane, we should zero pad if needed in input (keep odd)

#         # calc_samplesN = np.where(
#         #     self.out_size > self.in_size, self.out_size, self.in_size
#         # )
#         # self.calc_samplesN = np.where(
#         #     np.mod(calc_samplesN, 2) == 0, calc_samplesN + 1, calc_samplesN
#         # )

#         # # For the Fresnel Engine, the output field grid size is tuned by zero-padding the input field
#         # # We should then zero-pad more, dependent on wavelength, to get the output pixel size. This will
#         # # be computed on the fly in forward call
#         return
#
#
# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt

#     from dflat.initialize import focusing_lens
#     from dflat.propagation import ASMPropagation, FresnelPropagation

#     in_size = [701, 701]
#     in_dx_m = [500e-9, 500e-9]
#     wavelength_set_m = [532e-9]
#     depth_set_m = [1.0]
#     fshift_set_m = [[0.0, 0.0]]
#     out_distance_m = 500e-6

#     lens_t, lens_phi, aperture = focusing_lens(
#         in_size,
#         in_dx_m,
#         wavelength_set_m,
#         depth_set_m,
#         fshift_set_m,
#         out_distance_m,
#         aperture_radius_m=150e-6,
#     )

#     print(lens_t.shape, aperture.shape)
#     lens_t = lens_t * aperture
#     lens_phi = lens_phi * aperture

#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(lens_t[0], vmin=0, vmax=1)
#     ax[1].imshow(lens_phi[0])

#     out_size = in_size
#     out_dx_m = in_dx_m
#     out_distance_m = 100e-6
#     propagator = FresnelPropagation(
#         in_size, in_dx_m, out_distance_m, out_size, out_dx_m, manual_upsample_factor=1
#     )

#     out_t, out_phi = propagator(lens_t[None], lens_phi[None], [500e-9, 600e-9, 700e-9])
