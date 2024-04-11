import torch


def latent_to_param(z_latent, pmin=0, pmax=1, func="tanh"):
    """Takes a latent tensor (defined on domain R[-inf, inf]) and returns the corresponding param tensor
        (defined on domain R[pmin, pmax]).

    Args:
        `z_latent` (tf.float): Latent tensor
        `pmin` (tf.float, optional): minimum param value. Defaults to 0.
        `pmax` (tf.float, optional): maximum param value. Defaults to 1.

    Returns:
        `tf.float`: param tensor of equivalent shape

    """
    assert func in ["sigmoid", "tanh", "sine"]
    z_latent = torch.tensor(z_latent) if not torch.is_tensor(z_latent) else z_latent

    if func == "tanh":
        return ((torch.tanh(z_latent) + 1) / 2 * (pmax - pmin)) + pmin
    elif func == "sigmoid":
        return 1 / (1 + torch.exp(-z_latent))


def param_to_latent(p_param, pmin=0.0, pmax=1.0, func="tanh"):
    """Takes a param tensor (defined on domain R[pmin, pmax]) and returns the corresponding latent tensor
    (defined on domain R[-inf, inf]).

    Args:
        `p_param` (tf.float): Param tensor
        `pmin` (tf.float, optional): minimum param value. Defaults to 0.
        `pmax` (tf.float, optional): maximum param value. Defaults to 1.

    Returns:
        `tf.float`: Latent tensor of equivalent shape
    """
    assert func in ["sigmoid", "tanh", "sine"]
    p_param = torch.tensor(p_param) if not torch.is_tensor(p_param) else p_param

    if func == "tanh":
        return torch.atanh(
            (torch.clip(p_param, pmin, pmax) - pmin) / (pmax - pmin) * 2 - 1
        )
    elif func == "sigmoid":
        return torch.log(p_param / (1 - p_param))
