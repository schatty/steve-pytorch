import torch.multiprocessing as mp


class TD4Trainer:
    def __init__(self, config):
        self.config = config

    def train(self, env, eval_func):
        config = self.config

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=64)
        update_step = mp.VAlue('i', 0)
        learner_w_queue = mp.Queue(maxsize=n_agents)
