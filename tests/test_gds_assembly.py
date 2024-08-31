import pytest
import numpy as np
import os
import tempfile

from dflat.GDSII import (
    assemble_cylinder_gds,
    assemble_ellipse_gds,
    assemble_fin_gds,
)


@pytest.fixture
def temp_gds_file():
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as tmp_file:
        yield tmp_file.name
    os.unlink(tmp_file.name)


@pytest.fixture
def sample_data():
    return {
        "params_1d": np.random.rand(10, 10, 1),
        "params_2d": np.random.rand(10, 10, 2),
        "mask": np.random.choice([True, False], size=(10, 10)),
        "cell_size": [1e-6, 1e-6],
        "block_size": [2e-6, 2e-6],
    }


def test_assemble_cylinder_gds(sample_data, temp_gds_file):
    try:
        assemble_cylinder_gds(
            sample_data["params_1d"],
            sample_data["mask"],
            sample_data["cell_size"],
            sample_data["block_size"],
            temp_gds_file,
        )
    except Exception as e:
        pytest.fail(f"assemble_nanocylinder_gds raised an exception: {e}")


def test_assemble_ellipse_gds(sample_data, temp_gds_file):
    try:
        assemble_ellipse_gds(
            sample_data["params_2d"],
            sample_data["mask"],
            sample_data["cell_size"],
            sample_data["block_size"],
            temp_gds_file,
        )
    except Exception as e:
        pytest.fail(f"assemble_ellipse_gds raised an exception: {e}")


def test_assemble_fin_gds(sample_data, temp_gds_file):
    try:
        assemble_fin_gds(
            sample_data["params_2d"],
            sample_data["mask"],
            sample_data["cell_size"],
            sample_data["block_size"],
            temp_gds_file,
        )
    except Exception as e:
        pytest.fail(f"assemble_nanofin_gds raised an exception: {e}")
