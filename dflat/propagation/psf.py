import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .propagators import ASMPropagation, FresnelPropagation
from .util import cart_grid
from dflat.radial_tranforms import radial_2d_transform


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

    @torch.no_grad()
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
