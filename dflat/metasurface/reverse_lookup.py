import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .load_utils import load_optical_model
from .latent import latent_to_param


def reverse_lookup_optimize(
    amp,
    phase,
    wavelength_set_m,
    model_name,
    lr=1e-1,
    err_thresh=1e-2,
    max_iter=1000,
    opt_phase_only=False,
    force_cpu=False,
    batch_size=None,
):
    """Given a stack of wavelength dependent amplitude and phase profiles, runs a reverse optimization to identify the nanostructures that
    implements the desired profile across wavelength by minimizing the mean absolute errors of complex fields.

    Args:
        amp (float): Target amplitude of shape [B, Pol, Lam, H, W].
        phase (float): Target phase of shape [B, Pol, Lam, H, W].
        wavelength_set_m (list): List of wavelengths corresponding to the Lam dimension of the target profiles.
        model_name (str): Model name. Either in the local path "DFlat/Models/NAME/" or to be retrieved from online.
        lr (float, optional): Optimization learning rate. Defaults to 1e-1.
        err_thresh (float, optional): Early termination threshold. Defaults to 0.1.
        max_iter (int, optional): Maximum number of steps. Defaults to 1000.
        batch_size (int, optional): Number of cells to evaluate at once via model batching.

    Returns:
        list: Returns normalized and unnormalized metasurface design parameters of shape [B, H, W, D] where D is the number of shape parameters. Last item in list is the MAE loss for each step.
    """
    B, P, L, H, W = amp.shape
    assert amp.shape == phase.shape
    assert (
        len(wavelength_set_m) == L
    ), "Wavelength list should match amp,phase wavelength dim (dim3)."

    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running optimization with device {device}")
    model = load_optical_model(model_name).to(device)
    pg = model.dim_out // 3
    assert pg == P, f"Polarization dimension of amp, phase (dim1) expected to be {pg}."

    shape_dim = model.dim_in - 1
    # z = np.random.rand(B, H, W, shape_dim)
    z = np.zeros((B, H, W, shape_dim))
    z = torch.tensor(z, device=device, dtype=torch.float32, requires_grad=True)

    wavelength = (
        torch.tensor(wavelength_set_m)
        if not torch.is_tensor(wavelength_set_m)
        else wavelength_set_m
    )
    wavelength = wavelength.to(dtype=torch.float32, device=device)
    wavelength = model.normalize_wavelength(wavelength)
    torch_zero = torch.tensor(0.0, dtype=z.dtype, device=device)
    amp = (
        torch.tensor(amp, dtype=torch.float32, device=device)
        if not torch.is_tensor(amp)
        else amp.to(dtype=torch.float32, device=device)
    )
    phase = (
        torch.tensor(phase, dtype=torch.float32, device=device)
        if not torch.is_tensor(phase)
        else phase.to(dtype=torch.float32, device=device)
    )
    target_field = torch.complex(amp, torch_zero) * torch.exp(
        torch.complex(torch_zero, phase)
    )

    # Optimize
    err = 1e3
    steps = 0
    err_list = []
    optimizer = optim.AdamW([z], lr=lr)
    pbar = tqdm(total=max_iter, desc="Optimization Progress")
    while err > err_thresh:
        if steps >= max_iter:
            pbar.close()
            break

        optimizer.zero_grad()
        pred_amp, pred_phase = model(
            latent_to_param(z), wavelength, pre_normalized=True, batch_size=batch_size
        )

        if opt_phase_only:
            loss = torch.mean(
                torch.abs(
                    torch.exp(torch.complex(torch_zero, pred_phase))
                    - torch.exp(torch.complex(torch_zero, phase))
                )
            )
        else:
            pred_field = torch.complex(pred_amp, torch_zero) * torch.exp(
                torch.complex(torch_zero, pred_phase)
            )
            loss = torch.mean(torch.abs(pred_field - target_field))
        loss.backward()
        optimizer.step()
        err = loss.item()
        steps += 1
        err_list.append(err)
        pbar.update(1)
        pbar.set_description(f"Loss: {err:.4f}")
    pbar.close()

    op_param = latent_to_param(z).detach().cpu().numpy()
    return op_param, model.denormalize(op_param), err_list
