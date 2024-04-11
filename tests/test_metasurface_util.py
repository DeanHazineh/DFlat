import pytest
import torch
from dflat.metasurface.latent import latent_to_param, param_to_latent  

@pytest.mark.parametrize("func", ["tanh", "sigmoid"])
@pytest.mark.parametrize("pmin_pmax", [(0, 1), (-1, 1), (0, 10)])
def test_latent_to_param_and_back_consistency(func, pmin_pmax):
    """Test if converting a latent value to a param value and back is consistent."""
    pmin, pmax = pmin_pmax
    z_latent_original = torch.randn(10, 10)  # Generating a random tensor
    p_param = latent_to_param(z_latent_original, pmin=pmin, pmax=pmax, func=func)
    z_latent_converted = param_to_latent(p_param, pmin=pmin, pmax=pmax, func=func)

    # Using allclose since the conversion might not be perfectly precise due to numerical errors
    assert torch.allclose(z_latent_original, z_latent_converted, atol=1e-6), \
        f"Conversion failed for func={func} and pmin_pmax={pmin_pmax}"

@pytest.mark.parametrize("func", ["tanh", "sigmoid"])
def test_latent_to_param_values(func):
    """Test specific known values for latent to param conversion."""
    z_latent = torch.tensor([-float('inf'), 0, float('inf')])
    expected = {
        "tanh": torch.tensor([0, 0.5, 1]),
        "sigmoid": torch.tensor([0, 0.5, 1]),
        "sine": torch.tensor([-1, 0, 1])  # Assuming clipping in sine, or it would be NaN
    }[func]
    p_param = latent_to_param(z_latent, func=func)
    assert torch.allclose(p_param, expected, atol=1e-6), f"Values mismatch for func={func}"

@pytest.mark.parametrize("func", ["tanh", "sigmoid"])
def test_param_to_latent_boundaries(func):
    """Test conversion boundaries for param to latent."""
    p_param = torch.tensor([0, 0.5, 1])
    if func in ["tanh", "sigmoid"]:
        z_latent = param_to_latent(p_param, func=func)
        # For tanh and sigmoid, the boundaries should be -inf, 0, inf
        expected_inf = torch.tensor([float('-inf'), 0, float('inf')])
        assert torch.isinf(z_latent[0]) and z_latent[0] < 0, f"-inf boundary check failed for func={func}"
        assert torch.isinf(z_latent[-1]) and z_latent[-1] > 0, f"+inf boundary check failed for func={func}"
        assert z_latent[1] == 0, f"Middle value check failed for func={func}"
    elif func == "sine":
        with pytest.raises(ValueError):
            param_to_latent(p_param, func=func)  # sine might not handle values exactly at 0 or 1 well
