import torch
import torch.optim as optim

from .load_utils import load_optical_model
from .latent import latent_to_param


def reverse_lookup_optimize(
    amp,
    phase,
    wavelength_set_m,
    model_config_path,
    lr=1e-1,
    err_thresh=0.1,
    max_iter=1000,
):
    """Given a stack of wavelength dependent amplitude and phase profiles, runs a reverse optimization to identify the nanostructures that
    implements the desired profile across wavelength by minimizing the mean absolute errors of complex fields.

    Args:
        amp (float): Target amplitude of shape [B, Pol, Lam, H, W].
        phase (float): Target phase of shape [B, Pol, Lam, H, W].
        wavelength_set_m (list): List of wavelengths corresponding to the Lam dimension of the target profiles.
        model_config_path (str): Relative path for the model config file like "metasurface/ckpt/..."
        lr (float, optional): Optimization learning rate. Defaults to 1e-1.
        err_thresh (float, optional): Early termination threshold. Defaults to 0.1.
        max_iter (int, optional): Maximum number of steps. Defaults to 1000.

    Returns:
        list: Returns normalized and unnormalized metasurface design parameters of shape [B, H, W, D] where D is the number of shape parameters.
    """
    B, P, L, H, W = amp.shape
    assert amp.shape == phase.shape
    assert (
        len(wavelength_set_m) == L
    ), "Wavelength list should match amp,phase wavelength dim (dim3)."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_optical_model(model_config_path).to(device)
    pg = model.dim_out // 3
    assert pg == P, f"Polarization dimension of amp, phase (dim1) expected to be {pg}."

    z = torch.zeros(
        (B, H, W, model.dim_in - 1),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    wavelength = (
        torch.tensor(wavelength_set_m)
        if not torch.is_tensor(wavelength_set_m)
        else wavelength_set_m
    )
    wavelength = wavelength.to(dtype=torch.float32, device=device)
    wavelength = model.normalize_wavelength(wavelength)

    # optimize
    optimizer = optim.Adam([z], lr=lr)
    torch_zero = torch.tensor(0.0, dtype=z.dtype, device=device)
    amp = torch.tensor(amp, dtype=torch.float32, device=device)
    phase = torch.tensor(phase, dtype=torch.float32, device=device)
    target_field = torch.complex(amp, torch_zero) * torch.exp(
        torch.complex(torch_zero, phase)
    )

    err = 1e3
    steps = 0
    while err > err_thresh:
        if steps >= max_iter:
            break

        optimizer.zero_grad()
        pred_amp, pred_phase = model(
            latent_to_param(z), wavelength, pre_normalized=True
        )
        pred_field = torch.complex(pred_amp, torch_zero) * torch.exp(
            torch.complex(torch_zero, pred_phase)
        )

        loss = torch.mean(torch.abs(pred_field - target_field))
        loss.backward()
        optimizer.step()
        err = loss.item()
        steps += 1

    op_param = latent_to_param(z).detach().cpu().numpy()
    return op_param, model.denormalize(op_param)
