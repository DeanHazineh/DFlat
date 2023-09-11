import numpy as np
import torch
import matplotlib.pyplot as plt

import dflat.optimization_helper as df_opt
import dflat.tools as df_tools
import dflat.fourier_layer as df_fourier
import dflat.neural_optical_layer as df_neural
import dflat.plot_utilities as df_plt
import dflat.data_structure as df_struct


class opt_pipeline(df_opt.dflat_pipeline):
    def __init__(self, prop_params, point_source_locs, neural_model, device, savepath, saveAtEpochs):
        super().__init__(savepath, saveAtEpochs)

        # Store to attributes
        self.prop_params = prop_params
        self.point_source_locs = point_source_locs
        self.wavelength_set_m = prop_params["wavelength_set_m"]
        self.sensor_pixel_number = prop_params["sensor_pixel_number"]
        airy_profile = df_tools.airy_disk(prop_params)
        self.airy_profile = torch.tensor(airy_profile[:, None, None, :, :], dtype=prop_params["dtype"], device=device)

        # Initialize layers
        self.psf_layer = df_fourier.PSF_Layer(prop_params, device)
        self.neural_model = df_neural.MLP_Latent_Layer(neural_model, dtype=prop_params["dtype"], device=device)
        init_latent = self.neural_model.initialize_input_tensor("uniform", prop_params["grid_shape"]).to(device)

        # Trainable Variable
        self.init_latent = torch.nn.Parameter(init_latent)

    def forward(self):
        x = self.neural_model(self.init_latent, self.wavelength_set_m)
        x, _ = self.psf_layer(x, self.point_source_locs, batch_loop=True)

        return torch.sum(torch.abs(x - self.airy_profile))

    def visualizeTrainingCheckpoint(self, epoch):
        # Get the PSF at current state
        self.eval()
        with torch.no_grad():
            out = self.neural_model(self.init_latent, self.wavelength_set_m)
            out, _ = self.psf_layer(out, self.point_source_locs)
            out = torch.sum(out, 1).cpu().numpy()

        # make a figure
        use_wl = self.wavelength_set_m * 1e9
        num_wl = len(use_wl)
        x, y = df_plt.get_detector_pixel_coordinates(self.prop_params)
        x = x * 1e6
        y = y * 1e6

        fig = plt.figure(figsize=(3 * num_wl, 6))
        ax = df_plt.addAxis(fig, 2, num_wl)
        for i in range(num_wl):
            ax[i].imshow(out[i, 0, :, :])
            df_plt.formatPlots(fig, ax[i], None, title=f"wl {use_wl[i]:.0f}", setAspect="equal", xgrid_vec=x, ygrid_vec=y, addcolorbar=True)
            ax[i + num_wl].imshow(out[i, 0, :, :])
            df_plt.formatPlots(
                fig,
                ax[i + num_wl],
                None,
                title=f"wl {use_wl[i]:.0f}",
                setAspect="equal",
                xgrid_vec=x,
                ygrid_vec=y,
                setxLim=[-150, 150],
                setyLim=[-150, 150],
                addcolorbar=True,
            )
        plt.tight_layout()
        plt.savefig(self.savepath + f"png_images/vis{epoch}.png")
        plt.close()

        return

    def save_to_gds(self):
        # Make sure we have loaded the last training checkpoint
        self.customLoad()

        ms_dx_m = self.prop_params["ms_dx_m"]
        ms_dx_m = {"x": ms_dx_m["x"], "y": ms_dx_m["y"]}
        aperture = self.psf_layer.aperture_trans
        self.neural_model.save_to_gds(self.init_latent, ms_dx_m, self.savepath, aperture)

        return


def call_nanofin_opt():
    # Initialize calculation
    device = torch.device("cuda:0")
    wavelength_set_m = np.linspace(450e-9, 750e-9, 6)
    point_source_locs = None
    savepath = f"test_code/output/"
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_set_m": wavelength_set_m,
            "ms_samplesM": {"x": 357, "y": 357},
            "ms_dx_m": {"x": 4 * 350e-9, "y": 4 * 350e-9},
            "radius_m": 0.25e-3,
            "sensor_distance_m": 5e-3,
            "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_number": {"x": 501, "y": 501},
            "radial_symmetry": False,
            "diffractionEngine": "ASM_fourier",
            "dtype": torch.float32,
        }
    )
    df_struct.print_full_settings(propagation_parameters)

    pipeline = opt_pipeline(propagation_parameters, point_source_locs, "MLP_Nanofins_Dense1024_U350_H600_SIREN100", device, savepath, saveAtEpochs=5)
    df_opt.run_pipeline_optimization(pipeline, num_epochs=5, optimizer_type="Adam", lr=1e-1, loss_fn=None, load_previous_ckpt=True)
    pipeline.save_to_gds()

    return


if __name__ == "__main__":
    call_nanofin_opt()
