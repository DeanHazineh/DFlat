from load_utils import instantiate_from_config


class Trainer_v1:
    def __init__(self, config_model, *args, **kwargs):
        config_model.params.trainable_model = True
        self.model = instantiate_from_config(config_model)

    def train(self):
        return
