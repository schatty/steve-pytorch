import torch.multiprocessing as mp


from .td3 import TD3


def sampler_worker(config, replay_queue, batch_queue, update_step):
    """Worker that moves transitions from replay buffer
    to the batch_queue.

    Args:
        config: configuration dict.
        replay_queue: queue with transitions.
        batch_queue: queue with batches for NN
        update_step: overall system step (learner update step).
    """
    pass


def learner_worker(config, policy_class, learner_queue,
                   batch_queue, update_step):
    """Policy update logic.

    Args:
        config: configuration dict.
        policy_class: class of the policy.
        learner_queue: queue with policy weights.
        batch_queue: queue with batches for policy.
        udpate_step: overall system step (learner udpate step).
    """
    pass


def agent_worker(config, agent_n, policy_class, replay_queue,
                 learner_queue, update_step):
    """Worker that gathers data from sim.

    Args:
        config: configruation dict.
        agent_n: index of the agent worker.
        policy_class: class of the policy.
        replay_queue: queue with transitions.
        learner_queue: queue with policy weights.
        udpate_step: overall system step (learner update step).
    """
    pass


class TD4Trainer:
    def __init__(self, config):
        self.config = config

    def train(self, env, eval_func):
        config = self.config

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=64)
        batch_queue = mp.Queue()
        update_step = mp.VAlue('i', 0)
        learner_queue = mp.Queue(maxsize=config["n_agents"])

        # Data sampler (replay_queue -> batches)
        p = mp.Process(
                target=sampler_worker,
                args=(config,
                      replay_queue,
                      batch_queue,
                      update_step)
        )
        processes.append(p)

        # Policy learner
        policy_class = TD3
        p = mp.Process(
                target=learner_worker,
                args=(config,
                      policy_class,
                      learner_queue,
                      batch_queue,
                      update_step)
        )

        # Data gathering agents
        for i in range(config["n_agents"]):
            p = mp.Process(
                    target=agent_worker,
                    args=(config,
                          i,
                          policy_class,
                          replay_queue,
                          learner_queue,
                          update_step)
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.end()
