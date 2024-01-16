import torch
import torch.nn as nn
from .load_utils import instantiate_from_config, get_obj_from_str


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class NeuralCells(nn.Module):
    def __init__(self, nn_config, base_lib, trainable_model=False):
        super().__init__()
        self.model = self._initialize_model(nn_config, trainable_model)
        self.base_library = self._initialize_base(base_lib)
        self.param_limits = self.base_library.param_limits
        self.loss = nn.L1Loss()

    def training_step(self, x, y):
        pred = self.model(x)
        return self.loss(pred, y)

    def forward(self, x):
        # Call the model, reshape to sets of 3 and convert to real trans, phase
        x = torch.tensor(x, torch.float32)
        y = self.model(x)

        return

    def _initialize_model(self, config, trainable_model):
        model = instantiate_from_config(
            config, ckpt_path=config["ckpt_path"], strict=False
        )

        if not trainable_model:
            model = model.eval()
            model.train = disabled_train
            for param in model.parameters():
                param.requires_grad = False

        return model

    def _initialize_base(self, obj_str):
        return get_obj_from_str(obj_str)()


class NeuralFields(nn.Module):
    def __init__(self, nn_config, trainable_model=False):
        pass
