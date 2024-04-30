import torch


def latent_to_param(z_latent, pmin=0, pmax=1, func="sigmoid"):
    """Takes a latent tensor (defined on domain R[-inf, inf]) and returns the corresponding param tensor
        (defined on domain R[pmin, pmax]).

    Args:
        `z_latent` (float): Latent tensor
        `pmin` (float, optional): minimum param value. Defaults to 0.
        `pmax` (float, optional): maximum param value. Defaults to 1.

    Returns:
        `float`: param tensor of equivalent shape

    """
    assert func in ["sigmoid", "tanh"]
    z_latent = torch.tensor(z_latent) if not torch.is_tensor(z_latent) else z_latent

    if func == "sigmoid":
        p = 1 / (1 + torch.exp(-z_latent))
    elif func == "tanh":
        p = (torch.tanh(z_latent) + 1) / 2
    else:
        raise ValueError("Unexpected error, invalid func method encountered.")

    # Allow remapping of min and max parameter bounds for subsections of libraries
    return p * (pmax - pmin) + pmin


def param_to_latent(p_param, pmin=0.0, pmax=1.0, func="sigmoid", eps=1e-6):
    """Takes a param tensor (defined on domain R[pmin, pmax]) and returns the corresponding latent tensor
    (defined on domain R[-inf, inf]).

    Args:
        `p_param` (float): Param tensor
        `pmin` (float, optional): minimum param value. Defaults to 0.
        `pmax` (float, optional): maximum param value. Defaults to 1.

    Returns:
        `float`: Latent tensor of equivalent shape
    """
    assert func in ["sigmoid", "tanh"]

    # Avoid division by zero
    minp = max(eps, pmin)
    maxp = min(1 - eps, pmax)
    p = (torch.clip(p_param, minp, maxp) - minp) / (pmax - pmin)

    if func == "sigmoid":
        z = torch.log(p / (1 - p))
    elif func == "tanh":
        z = torch.atanh(p * 2 - 1)
    else:
        raise ValueError("unexpected error, invalid func method encountered.")

    return z
