import pkg_resources
import importlib
import torch
from omegaconf import OmegaConf
import os
import requests
import zipfile
from pathlib import Path

req_paths = {
    "Nanocylinders_Si3N4_U250H600": "https://www.dropbox.com/scl/fi/vpb5okjhzj3c1buy5tly9/Nanocylinders_Si3N4_U250H600.zip?rlkey=2ijbzy9kq9ub6mgwl32n3vffp&dl=1",
    "Nanocylinders_Si3N4_U300H600": "https://www.dropbox.com/scl/fi/2zv2ipx2jkncj4yndj3e2/Nanocylinders_Si3N4_U300H600.zip?rlkey=ctvejtgika2po25m25etug3x9&dl=1",
    "Nanocylinders_Si3N4_U350H600": "https://www.dropbox.com/scl/fi/524ax6e8undt9qoahlo65/Nanocylinders_Si3N4_U350H600.zip?rlkey=cd413i3cncplfddnmhxrciybl&dl=1",
    "Nanocylinders_TiO2_U200H600": "https://www.dropbox.com/scl/fi/nl9fhmfmsz41dilth6uis/Nanocylinders_TiO2_U200H600.zip?rlkey=i5xol70u0wq7k19q3dsmyuz3a&dl=1",
    "Nanocylinders_TiO2_U250H600": "https://www.dropbox.com/scl/fi/n5coocd6bcbdrl88jaler/Nanocylinders_TiO2_U250H600.zip?rlkey=am9vad3ssskoc7bntqdxy0ts2&dl=1",
    "Nanocylinders_TiO2_U300H600": "https://www.dropbox.com/scl/fi/sn44f2xzadcrag0jgzdsq/Nanocylinders_TiO2_U300H600.zip?rlkey=5hivknv8cvfy3gzzsyolb8bz5&dl=1",
    "Nanocylinders_TiO2_U350H600": "https://www.dropbox.com/scl/fi/43mf1xidor3mti9dv8bce/Nanocylinders_TiO2_U350H600.zip?rlkey=cyj6xb3reh5iv2rj9l1byxk27&dl=1",
    "Nanoellipse_TiO2_U350H600": "https://www.dropbox.com/scl/fi/6phh6a0kztbccy76vzwjd/Nanoellipse_TiO2_U350H600.zip?rlkey=0hn8cr2kgs3t9134kmrhf1ogx&dl=1",
    "Nanofins_TiO2_U350H600": "https://www.dropbox.com/scl/fi/co65yfwugkvugi7r8bqaj/Nanofins_TiO2_U350H600.zip?rlkey=8e0pzvzul8xlzl9szf15lbrzx&dl=1",
}


def model_config_path(model_name):
    dir_path = Path("DFlat/Models/")
    dir_path.mkdir(parents=True, exist_ok=True)

    config_exists = os.path.exists(os.path.join(dir_path, model_name, "config.yaml"))
    if not config_exists:
        print("Downloading the model from online storage.")
        zip_path = dir_path / "data.zip"
        load_url = req_paths[model_name]

        with requests.get(load_url, stream=True) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dir_path)

        zip_path.unlink()

    config_path = os.path.join(dir_path, model_name, "config.yaml")
    ckpt_path = os.path.join("DFlat/Models/", model_name, "model.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = None

    return config_path, ckpt_path


def load_optical_model(model_name):
    """Loads a neural optical model.

    Args:
        model_name (str): Name of the model from path "./DFlat/Models/NAME" or if not present, downloads the pre-trained model from online.

    Returns:
        nn.Module: Model with pretrained weights loaded if ckpt_path is specified.
    """

    config_path, ckpt_path = model_config_path(model_name)
    config = OmegaConf.load(config_path)

    optical_model = instantiate_from_config(config.model, ckpt_path, strict=True)
    return optical_model


def instantiate_from_config(config_model, ckpt_path=None, strict=False):
    assert "target" in config_model, "Expected key `target` to instantiate."
    target_str = config_model["target"]
    print(f"Target Module: {target_str}")
    loaded_module = get_obj_from_str(target_str)(**config_model.get("params", dict()))

    if ckpt_path is not None and ckpt_path != "None":
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        print(
            f"Target: {config_model['target']} Loading from checkpoint {ckpt_path} as strict={strict}"
        )
        missing, unexpected = loaded_module.load_state_dict(sd, strict=strict)
        print(
            f"Restored {target_str} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )

    return loaded_module


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def load_trainer(config_path):
    """Loads a model trainer based on a full config file which contains the trainer identifier. The trainer can be run by calling load_trainer().train().

    Args:
        config_path (str): Relative path to the model config file like 'metasurface/ckpt/Nanocylinders_TiO2_U180H600_Medium/config.yaml'.

    Returns:
        object: Trainer
    """
    config = OmegaConf.load(config_path)
    config_model = config.model

    config_trainer = config.trainer
    ckpt_path = config_trainer["ckpt_path"]

    dataset = get_obj_from_str(config_trainer["data"])()

    trainer = get_obj_from_str(config_trainer["target"])(
        config_model,
        ckpt_path,
        dataset,
        **config_trainer.get("params", dict()),
    )
    return trainer
