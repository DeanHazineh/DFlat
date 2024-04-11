## THe model ckpt files are not particularly large so it should be fine to actually download them during the unit test 
# This might change in the future

import pytest
from pathlib import Path
from dflat.metasurface import load_optical_model  

# Model names from the req_paths
model_names = [
    "Nanocylinders_Si3N4_U250H600",
    "Nanocylinders_Si3N4_U300H600",
    "Nanocylinders_Si3N4_U350H600",
    "Nanocylinders_TiO2_U200H600",
    "Nanocylinders_TiO2_U250H600",
    "Nanocylinders_TiO2_U300H600",
    "Nanocylinders_TiO2_U350H600",
    "Nanoellipse_TiO2_U350H600",
    "Nanofins_TiO2_U350H600",
]

@pytest.mark.parametrize("model_name", model_names)
def test_load_optical_model(model_name):
    model = load_optical_model(model_name)
    assert model is not None

    # Optionally, clean up by deleting the downloaded and extracted files 
    model_dir = Path("DFlat/Models/") / model_name
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir():
                for sub_item in item.iterdir():
                    sub_item.unlink()
                item.rmdir()
            else:
                item.unlink()
        model_dir.rmdir()
