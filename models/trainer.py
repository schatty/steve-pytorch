from .td3.td3_trainer import TD3TRainer


def load_trainer(config):
    if config["policy"] == "TD3":
        return TD3Trainer(config)
    else:
        raise ValueError("Unknown policy")
