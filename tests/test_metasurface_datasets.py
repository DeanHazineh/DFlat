import pytest
import os

from dflat.metasurface.datasets import ( 
    Nanocylinders_TiO2_U200nm_H600nm, Nanocylinders_TiO2_U250nm_H600nm,
    Nanocylinders_TiO2_U300nm_H600nm, Nanocylinders_TiO2_U350nm_H600nm,
    Nanocylinders_Si3N4_U250nm_H600nm, Nanocylinders_Si3N4_U300nm_H600nm,
    Nanocylinders_Si3N4_U350nm_H600nm, Nanofins_TiO2_U350nm_H600nm,
    Nanoellipse_TiO2_U350nm_H600nm
)

# List of dataset classes to test
dataset_classes = [
    Nanocylinders_TiO2_U200nm_H600nm, Nanocylinders_TiO2_U250nm_H600nm,
    Nanocylinders_TiO2_U300nm_H600nm, Nanocylinders_TiO2_U350nm_H600nm,
    Nanocylinders_Si3N4_U250nm_H600nm, Nanocylinders_Si3N4_U300nm_H600nm,
    Nanocylinders_Si3N4_U350nm_H600nm, Nanofins_TiO2_U350nm_H600nm,
    Nanoellipse_TiO2_U350nm_H600nm
]

@pytest.mark.parametrize("dataset_class", dataset_classes)
def test_dataset_initialization(dataset_class):
    """
    Test that each dataset class initializes without error.
    """
    dataset_instance = dataset_class()
    assert dataset_instance is not None

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "out/")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dataset_instance.plot(savepath=output_path)
 
    size_data = len(dataset_instance)
    