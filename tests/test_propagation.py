import pytest
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial

from dflat.propagation import PointSpreadFunction, ASMPropagation, FresnelPropagation
from dflat.initialize import focusing_lens


@pytest.fixture
def shared_init():
    sd = {
        "in_size": [201, 201],
        "in_dx_m": [2e-6, 2e-6],
        "out_distance_m": 10e-3,
        "out_size": [200, 200],
        "out_dx_m": [1e-6, 1e-6],
        "out_resample_dx_m": [2e-6, 2e-6],
        "manual_upsample_factor": 1.0,
        "radial_symmetry": False,
        "diffraction_engine": "asm",
    }

    get_lens = partial(
        focusing_lens,
        in_size=sd["in_size"],
        in_dx_m=sd["in_dx_m"],
        wavelength_set_m=[600e-9],
        depth_set_m=[20e-3],
        fshift_set_m=[[0.0, 0.0]],
        out_distance_m=sd["out_distance_m"],
        aperture_radius_m=200e-6,
    )

    def rewrap_phase(x):
        x = x.cpu().numpy() if torch.is_tensor(x) else x
        cidx = x.shape[-1] // 2
        return np.angle(np.exp(1j * (x - x[..., cidx : cidx + 1, cidx : cidx + 1])))

    return sd, get_lens, rewrap_phase


class Test_PointSpreadFunction:
    @pytest.fixture(autouse=True)
    def initialize(self, shared_init):
        self.init_dict, self.get_lens, self.rewrap_phase = shared_init

    def test_init(self):
        for engine in ["asm", "fresnel"]:
            sd = deepcopy(self.init_dict)
            sd["diffraction_engine"] = engine
            PointSpreadFunction(**sd)

        with pytest.raises(AssertionError):
            sd["diffraction_engine"] = "invalid"
            PointSpreadFunction(**sd)

    def test_forward(self):
        def reshape_dat(x):
            return x.view(-1, *self.init_dict["out_size"]).cpu().numpy()

        # Repeat claculate for radial symmetry and 2D
        device_list = ["cuda"] if torch.cuda.is_available else []
        device_list.append("cpu")

        for radial_flag in [True, False]:
            amp, phase, aperture = self.get_lens(radial_symmetry=radial_flag)
            amp = torch.tensor(amp, dtype=torch.float32)
            phase = torch.tensor(phase, dtype=torch.float32)

            wavelength_set_m = [400e-9, 600e-9]
            ps_locs_m = [[0, 0, 10e-3], [0, 0, 20e-3]]
            ps_locs_m = torch.tensor(ps_locs_m, dtype=torch.float32)

            sd = deepcopy(self.init_dict)
            sd["radial_symmetry"] = radial_flag

            # Repeat calculation on cuda and cpu
            for device in device_list:
                amp = amp.to(device=device)
                phase = phase.to(device=device)
                ps_locs_m = ps_locs_m.to(device=device)

                sd["diffraction_engine"] = "fresnel"
                fresnel = PointSpreadFunction(**sd)
                fres_int, fres_phase = fresnel(
                    amp,
                    phase,
                    wavelength_set_m,
                    ps_locs_m,
                    aperture,
                    normalize_to_aperture=True,
                )

                sd["diffraction_engine"] = "asm"
                asm = PointSpreadFunction(**sd)
                asm_int, asm_phase = asm(
                    amp,
                    phase,
                    wavelength_set_m,
                    ps_locs_m,
                    aperture,
                    normalize_to_aperture=True,
                )

                assert (
                    fres_int.shape
                    == fres_phase.shape
                    == asm_int.shape
                    == asm_phase.shape
                )
                assert list(fres_int.shape[-2:]) == self.init_dict["out_size"]
                assert fres_int.shape[0] == amp.shape[0]
                assert fres_int.shape[1] == len(ps_locs_m)
                assert fres_int.shape[2] == len(wavelength_set_m)

                fres_int = reshape_dat(fres_int)
                asm_int = reshape_dat(asm_int)
                fres_phase = self.rewrap_phase(reshape_dat(fres_phase))
                asm_phase = self.rewrap_phase(reshape_dat(asm_phase))

                mse_int = np.mean((fres_int - asm_int) ** 2)
                mse_phase = np.mean((fres_phase - asm_phase) ** 2)
                assert mse_int < 1e-8

                fig, ax = plt.subplots(4, 4)
                for c in range(4):
                    ax[0, c].imshow(fres_int[c])
                    ax[1, c].imshow(asm_int[c])
                    ax[2, c].imshow(fres_phase[c], cmap="hsv")
                    ax[3, c].imshow(asm_phase[c], cmap="hsv")
                for axi in ax.flatten():
                    axi.axis("off")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                out_dir = os.path.join(script_dir, "out")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                plot_path = os.path.join(
                    out_dir, f"psf_radial_{radial_flag}_{device}.png"
                )
                plt.savefig(plot_path)
                plt.close()


class Test_Propagation:
    @pytest.fixture(autouse=True)
    def initialize(self, shared_init):
        self.init_dict, self.get_lens, self.rewrap_phase = shared_init
        del self.init_dict["diffraction_engine"]

    def test_forward(self):
        device_list = ["cuda"] if torch.cuda.is_available else []
        device_list.append("cpu")

        for radial_flag in [True, False]:
            amp, phase, aperture = self.get_lens(radial_symmetry=radial_flag)
            amp = torch.tensor(amp, dtype=torch.float32)
            phase = torch.tensor(phase, dtype=torch.float32)
            wavelength_set_m = [400e-9, 600e-9]

            sd = deepcopy(self.init_dict)
            sd["radial_symmetry"] = radial_flag

            # Repeat calculation on cuda and cpu
            for device in device_list:
                amp = amp.to(device=device)
                phase = phase.to(device=device)

                asm = ASMPropagation(**sd)
                asm_amp, asm_phase = asm(amp, phase, wavelength_set_m)

                fresnel = FresnelPropagation(**sd)
                fres_amp, fres_phase = fresnel(amp, phase, wavelength_set_m)

                assert (
                    asm_amp.shape
                    == asm_phase.shape
                    == fres_amp.shape
                    == fres_phase.shape
                )
                assert list(fres_amp.shape[-2:]) == self.init_dict["out_size"]
                assert fres_amp.shape[0] == amp.shape[0]
                assert fres_amp.shape[1] == len(wavelength_set_m)

                fres_amp = fres_amp.cpu().numpy()
                asm_amp = asm_amp.cpu().numpy()
                fres_phase = self.rewrap_phase(fres_phase)
                asm_phase = self.rewrap_phase(asm_phase)

                fig, ax = plt.subplots(4, 2)
                for c in range(2):
                    ax[0, c].imshow(fres_amp[0, c])
                    ax[1, c].imshow(asm_amp[0, c])
                    ax[2, c].imshow(fres_phase[0, c], cmap="hsv")
                    ax[3, c].imshow(asm_phase[0, c], cmap="hsv")
                for axi in ax.flatten():
                    axi.axis("off")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                out_dir = os.path.join(script_dir, "out")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                plot_path = os.path.join(
                    out_dir, f"prop_radial_{radial_flag}_{device}.png"
                )
                plt.savefig(plot_path)
                plt.close()
