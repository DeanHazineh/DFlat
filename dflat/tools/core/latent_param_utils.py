import torch


def latent_to_param(z_latent, pmin=0, pmax=1):
    """Takes a latent tensor (defined on domain R[-inf, inf]) and returns the corresponding param tensor
        (defined on domain R[pmin, pmax]).

    Args:
        `z_latent` (tf.float): Latent tensor
        `pmin` (tf.float, optional): minimum param value. Defaults to 0.
        `pmax` (tf.float, optional): maximum param value. Defaults to 1.

    Returns:
        `tf.float`: param tensor of equivalent shape

    """
    z_latent = torch.tensor(z_latent) if not torch.is_tensor(z_latent) else z_latent
    pmin = torch.tensor(pmin).type_as(z_latent) if not torch.is_tensor(pmin) else pmin
    pmax = torch.tensor(pmax).type_as(z_latent) if not torch.is_tensor(pmax) else pmax
    return ((torch.tanh(torch.clip(z_latent, -10, 10)) + 1) / 2 * (pmax - pmin)) + pmin


def param_to_latent(p_param, pmin=0.0, pmax=1.0):
    """Takes a param tensor (defined on domain R[pmin, pmax]) and returns the corresponding latent tensor
    (defined on domain R[-inf, inf]).

    Args:
        `p_param` (tf.float): Param tensor
        `pmin` (tf.float, optional): minimum param value. Defaults to 0.
        `pmax` (tf.float, optional): maximum param value. Defaults to 1.

    Returns:
        `tf.float`: Latent tensor of equivalent shape
    """
    p_param = torch.tensor(p_param) if not torch.is_tensor(p_param) else p_param
    pmin = torch.tensor(pmin).type_as(p_param) if not torch.is_tensor(pmin) else pmin
    pmax = torch.tensor(pmax).type_as(p_param) if not torch.is_tensor(pmax) else pmax

    return torch.atanh((torch.clip(p_param, pmin, pmax) - pmin) / (pmax - pmin) * 2 - 1)
