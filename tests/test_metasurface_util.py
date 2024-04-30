import pytest
import torch
from dflat.metasurface.latent import latent_to_param, param_to_latent


@pytest.mark.parametrize(
    "initial_params, func",
    [
        (0.01, "sigmoid"),
        (0.5, "sigmoid"),
        (0.99, "sigmoid"),
        (0.01, "tanh"),
        (0.5, "tanh"),
        (0.99, "tanh"),
    ],
)
def test_param_to_latent_to_param(initial_params, func):
    """Test the round-trip conversion from params to latent and back to params."""
    p_params = torch.tensor([initial_params], dtype=torch.float)
    z_latent = param_to_latent(p_params, func=func)
    p_recovered = latent_to_param(z_latent, func=func)

    assert torch.isclose(
        p_params, p_recovered, atol=1e-6
    ), f"Round-trip conversion failed for func={func} with initial params={initial_params}"
