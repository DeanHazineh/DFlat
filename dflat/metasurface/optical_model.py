import torch
import torch.nn as nn
from einops import rearrange
from .load_utils import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class NeuralCells(nn.Module):
    def __init__(self, nn_config, param_bounds, trainable_model=False):
        super().__init__()
        self.model = self._initialize_model(nn_config, trainable_model)
        self.param_bounds = param_bounds
        self.loss = nn.L1Loss()

    def training_step(self, x, y):
        pred = self.model(x)
        return self.loss(pred, y)

    def forward(self, params, wavelength, pre_normalized=True):
        # params will be in a lens shape Batch, H, W, Channels
        # returns shape batch, g, lam, h, w,
        num_ch = params.shape[-1]
        assert num_ch == (
            len(self.param_bounds) - 1
        ), "Channel dimension is inconsistent with loaded model"
        assert len(params.shape) == 4
        assert len(wavelength.shape) == 1
        b, h, w, c = params.shape

        if not pre_normalized:
            bs, bw = self.param_bounds[:-1], self.param_bounds[-1]
            for i in range(num_ch):
                bounds = bs[i]
                params[:, :, :, i] = (params[:, :, :, i] - bounds[0]) / (
                    bounds[1] - bounds[0]
                )
            wavelength = (wavelength - bw[0]) / (bw[1] - bw[0])

        x = (
            torch.tensor(params, dtype=torch.float32).to("cuda")
            if not torch.is_tensor(params)
            else params.to(dtype=torch.float32, device="cuda")
        )
        lam = (
            torch.tensor(wavelength, dtype=torch.float32).to("cuda")
            if not torch.is_tensor(wavelength)
            else wavelength.to(dtype=torch.float32, device="cuda")
        )
        torch_zero = torch.tensor(0.0, dtype=x.dtype).to(device=x.device)

        x = rearrange(x, "b h w c -> (b h w) c")
        out = []
        for li in lam:
            out.append(self.model(torch.cat((x, li.repeat(x.shape[0], 1)), dim=1)))
        out = torch.stack(out)

        g = int(out.shape[-1] / 3)
        out = rearrange(
            out, "l (b h w) (g c3) -> l b h w g c3", g=g, c3=3, b=b, h=h, w=w
        )

        out = torch.complex(out[..., 0], torch_zero) * torch.exp(
            torch.complex(torch_zero, torch.atan2(out[..., 2], out[..., 1]))
        )

        return torch.abs(out), torch.angle(out)

    def _initialize_model(self, config, trainable_model):
        model = instantiate_from_config(
            config, ckpt_path=config["ckpt_path"], strict=False
        )

        if not trainable_model:
            model = model.eval()
            model.train = disabled_train
            for param in model.parameters():
                param.requires_grad = False

        return model


class NeuralFields(nn.Module):
    def __init__(self, nn_config, trainable_model=False):
        pass
