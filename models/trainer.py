from .td3.trainer import TD3Trainer


def load_trainer(config):
    if config["policy"] == "TD3":
        return TD3Trainer(config)
    else:
        raise ValueError("Unknown policy")
