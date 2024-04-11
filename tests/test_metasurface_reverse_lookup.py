import pytest
import numpy as np
from unittest.mock import patch

from dflat.metasurface import reverse_lookup_optimize

test_cases = [
    ("Nanocylinders_Si3N4_U250H600", 1),  # Pol dim should be 1
    ("Nanofins_TiO2_U350H600", 2),  # Pol should be 2
]


@pytest.mark.parametrize("model_name, pol_dim", test_cases)
@patch("torch.cuda.is_available", return_value=False)
def test_reverse_lookup_optimize_real_execution(
    mock_cuda_available, model_name, pol_dim
):
    B, L, H, W = 1, 1, 3, 4  # Specified dimensions
    # Create random amplitude and phase matrices
    amp = np.random.rand(B, pol_dim, L, H, W).astype(np.float32)
    phase = np.random.rand(B, pol_dim, L, H, W).astype(np.float32)
    wavelength_set_m = [550e-9]  # Single wavelength for simplicity

    # Run the reverse lookup function with limited iterations for the test
    results = reverse_lookup_optimize(
        amp=amp,
        phase=phase,
        wavelength_set_m=wavelength_set_m,
        model_name=model_name,
        max_iter=3,  # Limiting to 3 iterations
        lr=1e-1,
        err_thresh=1e-2,
        opt_phase_only=False,
        force_cpu=True,
    )

    # Unpack results
    op_param, denorm_op_param, err_list = results

    # Basic assertions to ensure outputs are correctly shaped and no errors occurred
    assert op_param.shape == (
        B,
        H,
        W,
        results[0].shape[-1],
    ), "Output parameter shape mismatch"
    assert len(err_list) == 3, "Should have exactly 3 entries in error list"
    print(
        f"Test completed for model {model_name} with polarization dimension {pol_dim}."
    )
