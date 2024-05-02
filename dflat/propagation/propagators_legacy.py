# This version of the propagators takes wavelength as an input to the forward call instead of the class initialization
# After some thought, doing this seemed to present no significant advantage for real end-to-end optimization tasks
# but it did present some downsides for the Fresnel theory. For this reason, that approach is moved as legacy and
# we tweaked another set of propagation functions to have wavelengths defined on initializaiton as the default.

# One advantage of moving it to initialization is that there is more flexibility to pre-initialize tensors for a little bit more efficiency.
# We can also now, in advance, evaluate the memory and computational complexity of the fresnel vs asm at initialization time allowing
# for dynamic selection because the choice is of one over the other is extremely problem dependent

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

        return checkpoint(self._forward, amplitude, phase, wavelength_set)

    def _forward(self, amplitude, phase, wavelength_set):
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

        return checkpoint(self._forward, amplitude, phase, wavelength_set)

    def _forward(self, amplitude, phase, wavelength_set):
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
        # This is defined by the manual upsample factor and in the future, we might want to return to
        # a different upsample factor for each wavelength channel
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


class PointSpreadFunction(nn.Module):
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
        diffraction_engine="ASM",
    ):
        super().__init__()

        assert isinstance(
            diffraction_engine, str
        ), "diffraction engine must be a string"
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
            out_resample_dx_m,
            manual_upsample_factor,
            radial_symmetry,
        )

        self.rescale = 1e6  # Convert m to um
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
        wavelength_set_m,
        ps_locs_m,
        aperture=None,
        normalize_to_aperture=True,
    ):
        """_summary_

        Args:
            amplitude (tensor): Lens amplitude of shape [... L H W].
            phase (tensor): Lens phase of shape [... L H W]
            wavelength_set_m (list): List of wavelengths corresponding to the L dimension. If L=1 in the passed in profiles,
                broadcasting will be used to propagate the same field at different wavelengths.
            ps_locs_m (tensor): Array point-source locations of shape [N x 3] where each column corresponds to Y, X, Depth
            aperture (Tensor, optional): Field aperture applied on the lens the same rank as amplitude
                and with the same H W dimensions. Defaults to None.
            normalize_to_aperture (bool, optional): If true the energy in the PSF will be normalized to the total energy
                incident on the optic/aperture. Defaults to True.

        Returns:
            List: Returns point-spread function intensity and phase of shape [B P Z L H W].
        """
        assert amplitude.shape == phase.shape
        assert len(amplitude.shape) >= 4
        assert len(wavelength_set_m) == amplitude.shape[-3] or amplitude.shape[-3] == 1
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

        N = amplitude.shape[0]
        Z = len(ps_locs_m)
        amplitude, phase = self._incident_wavefront(
            amplitude, phase, wavelength_set_m, ps_locs_m
        )
        if aperture is not None:
            amplitude = amplitude * aperture

        amplitude = rearrange(amplitude, "Z N L H W -> (N Z) L H W")
        phase = rearrange(phase, "Z N L H W -> (N Z) L H W")
        amplitude, phase = self.propagator(amplitude, phase, wavelength_set_m)
        amplitude = rearrange(amplitude, "(N Z) L H W -> N Z L H W", N=N, Z=Z)
        phase = rearrange(phase, "(N Z) L H W -> N Z L H W", N=N, Z=Z)

        # Return to the original shape before returning
        out_shape = amplitude.shape
        amplitude = amplitude.view(*init_shape[:-3], *out_shape[-4:])
        phase = phase.view(*init_shape[:-3], *out_shape[-4:])

        amplitude = amplitude**2
        normalization = (
            np.prod(self.out_resample_dx)
            * self.rescale
            / self.aperture_energy(aperture)
        ).to(dtype=amplitude.dtype, device=amplitude.device)
        if normalize_to_aperture:
            return amplitude * normalization, phase
        else:
            return amplitude, phase

    def _incident_wavefront(self, amplitude, phase, wavelength_set_m, ps_locs_m):
        # Z N L H W
        # Expand dimension to hold point_sources
        device = amplitude.device
        amplitude = amplitude[None].to(dtype=torch.float64)
        phase = phase[None].to(dtype=torch.float64)
        torch_zero = torch.tensor([0.0], dtype=torch.float64, device=device)

        wavelength_set = (
            torch.tensor(
                np.array(wavelength_set_m) * self.rescale,
                dtype=torch.float64,
                device=device,
            )
            if not torch.is_tensor(wavelength_set_m)
            else wavelength_set_m.to(dtype=torch.float64, device=device) * self.rescale
        )
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
    def aperture_energy(self, aperture):
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
            fieldblock2d**2 * np.prod(self.in_dx_m) * self.rescale,
            dim=(-1, -2),
            keepdim=True,
        )[None]

        return in_energy
