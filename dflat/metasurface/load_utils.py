import pkg_resources
import importlib
import torch
from omegaconf import OmegaConf


def load_optical_model(config_path):
    """Loads a neural optical model.

    Args:
        config_path (str): Relative path to the model config file like 'metasurface/ckpt/Nanocylinders_TiO2_U180H600_Medium/config.yaml'.

    Returns:
        nn.Module: Model with pretrained weights loaded if ckpt_path is specified in the config file.
    """
    config = load_config_from_path(config_path)
    ckpt_path = config.model["ckpt_path"]
    optical_model = instantiate_from_config(config.model, ckpt_path, strict=True)
    return optical_model


def load_trainer(config_path):
    """Loads a model trainer based on a full config file which contains the trainer identifier. The trainer can be run by calling load_trainer().train().

    Args:
        config_path (str): Relative path to the model config file like 'metasurface/ckpt/Nanocylinders_TiO2_U180H600_Medium/config.yaml'.

    Returns:
        object: Trainer
    """
    config = load_config_from_path(config_path)
    config_model = config.model
    config_trainer = config.trainer

    rel_ckpt = config_trainer["relative_ckpt"]
    ckpt_path = config_trainer["ckpt_path"]
    ckpt_path = (
        ckpt_path
        if not rel_ckpt
        else pkg_resources.resource_filename("dflat", ckpt_path)
    )
    dataset = get_obj_from_str(config_trainer["data"])()

    trainer = get_obj_from_str(config_trainer["target"])(
        config_model,
        ckpt_path,
        dataset,
        **config_trainer.get("params", dict()),
    )
    return trainer


def load_config_from_path(config_path):
    try:
        use_path = pkg_resources.resource_filename("dflat", config_path)
        config = OmegaConf.load(use_path)
    except Exception as e1:
        try:
            use_path = config_path
            config = OmegaConf.load(use_path)
        except Exception as e2:
            print(f"Failed absolute path identification. Errors \n {e1} \n {e2}.")

    return config


def instantiate_from_config(config_model, ckpt_path=None, strict=False):
    assert "target" in config_model, "Expected key `target` to instantiate."
    target_str = config_model["target"]
    print(f"Target Module: {target_str}")
    loaded_module = get_obj_from_str(target_str)(**config_model.get("params", dict()))

    if ckpt_path is not None and ckpt_path != "None":
        ## Try and Except to handle relative pathing vs absolute pathing
        try:
            use_path = pkg_resources.resource_filename("dflat", ckpt_path)
            sd = torch.load(use_path, map_location="cpu")["state_dict"]
        except Exception as e1:
            try:
                use_path = ckpt_path
                sd = torch.load(use_path, map_location="cpu")["state_dict"]
            except Exception as e2:
                print(f"Failed absolute path identification. Errors \n {e1} \n {e2}.")

        print(
            f"Target: {config_model['target']} Loading from checkpoint {use_path} as strict={strict}"
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
