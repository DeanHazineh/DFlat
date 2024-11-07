import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from .load_utils import instantiate_from_config
from .load_utils import get_obj_from_str


class NeuralCells(nn.Module):
    def __init__(self, nn_config, param_bounds, use_checkpoint=True, **kwargs):
        """Initializes a neural cell model from a config dictionary.

        Args:
            nn_config (dict): config dictionary
            param_bounds (list): List of min and max bounds for each design parameter (in units of m). The last nested list must correspond to wavelength.
            trainable_model (bool, optional): Flag to set model parameters to trainable (requires grad). Defaults to False.
        """
        super().__init__()

        if "trainable_model" in kwargs:
            print(
                "Note: trainable_model key in NeuralCells is deprecated. Model will be set to requires_grad."
            )
        self.dim_in = nn_config.params.dim_in
        self.dim_out = nn_config.params.dim_out
        self.model = instantiate_from_config(
            nn_config, ckpt_path=nn_config["ckpt_path"], strict=False
        )
        self.param_bounds = param_bounds
        self.loss = get_obj_from_str(nn_config.loss)()
        self.use_checkpoint = use_checkpoint

    def training_step(self, x, y):
        pred = self.model(x)
        return self.loss(pred, y)

    def forward(self, params, wavelength, pre_normalized=True, batch_size=None):
        """Predict the cell optical response from the passed in design parameters and wavelength

        Args:
            params (tensor): Metasurface design parameters of shape [B, H, W, D] where D is the shape dimensionality.
            wavelength (tensor): List of wavelengths to compute the optical response for.
            pre_normalized (bool, optional): Flag to indicate if the passed in params and wavelength are already normalized
                to the range [0,1]. If False, the passed in tensors will be automatically normalized based on the config
                settings.  Defaults to True.
            batch_size (int, optional): Number of cells to evaluate at once via model batching.

        Returns:
            list: Amplitude and Phase of shape [B, pol, Lam, H, W] where pol is 1 or 2.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = (
            torch.tensor(params, dtype=torch.float32).to(device)
            if not torch.is_tensor(params)
            else params.to(dtype=torch.float32)
        )
        lam = (
            torch.tensor(wavelength, dtype=torch.float32).to(device)
            if not torch.is_tensor(wavelength)
            else wavelength.to(dtype=torch.float32)
        )
        torch_zero = torch.tensor(0.0, dtype=x.dtype).to(device=x.device)

        num_ch = x.shape[-1]
        b, h, w, c = x.shape
        assert num_ch == (
            len(self.param_bounds) - 1
        ), "Channel dimension is inconsistent with loaded model"
        assert len(x.shape) == 4
        assert len(lam.shape) == 1
        if batch_size is not None:
            assert isinstance(batch_size, int), "batch size must be an integer"
        else:
            batch_size = b * h * w

        if not pre_normalized:
            x = self.normalize(x)
            lam = self.normalize_wavelength(lam)

        x = rearrange(x, "b h w c -> (b h w) c")
        out = []
        for li in lam:
            li_repeated = li.repeat(x.shape[0], 1)  # Repeat `li` once for all rows in x
            chout = [
                self.model(torch.cat((x[start:end], li_repeated[start:end]), dim=1))
                for start in range(0, x.shape[0], batch_size)
                for end in [min(start + batch_size, x.shape[0])]
            ]
            out.append(torch.cat(chout, dim=0))  # Concatenate results for each `li`
        out = torch.stack(out)

        g = int(out.shape[-1] / 3)
        out = rearrange(
            out, "l (b h w) (g c3) -> b g l h w c3", g=g, c3=3, b=b, h=h, w=w
        )

        out = torch.complex(out[..., 0], torch_zero) * torch.exp(
            torch.complex(
                torch_zero, torch.atan2((out[..., 2] * 2) - 1, (out[..., 1] * 2) - 1)
            )
        )
        return torch.abs(out), torch.angle(out)

    def normalize_wavelength(self, wavelength_set_m):
        bw = self.param_bounds[-1]
        return (wavelength_set_m - bw[0]) / (bw[1] - bw[0])

    def normalize(self, params):
        bs = self.param_bounds[:-1]
        norm_p = []
        for i, bounds in enumerate(bs):
            norm_p.append((params[:, :, :, i] - bounds[0]) / (bounds[1] - bounds[0]))

        norm_p = (
            torch.stack(norm_p, dim=-1)
            if torch.is_tensor(params)
            else np.stack(norm_p, axis=-1)
        )
        return norm_p

    def denormalize(self, params):
        bs = self.param_bounds[:-1]
        out = []
        for i, bounds in enumerate(bs):
            out.append(params[:, :, :, i] * (bounds[1] - bounds[0]) + bounds[0])

        out = (
            torch.stack(out, dim=-1)
            if torch.is_tensor(params)
            else np.stack(out, axis=-1)
        )
        return out


class NeuralFields(nn.Module):
    def __init__(self, nn_config, trainable_model=False):
        pass
