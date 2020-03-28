from .td3.trainer import TD3Trainer
from .td4.trainer import TD4Trainer


def load_trainer(config):
    if config["policy"] == "TD3":
        return TD3Trainer(config)
    elif config["policy"] == "TD4":
        return TD4Trainer(config)
    else:
        raise ValueError("Unknown policy")
