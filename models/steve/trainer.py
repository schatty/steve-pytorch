import os
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

try:
    set_start_method("spawn")
    print("spawn success")
except RuntimeError:
    print("Failed to set spawn method")


from .td3 import TD3
from .model import Model
from ..agent import Agent
from ..utils import ReplayBuffer, Logger


def sampler_worker(config, replay_queue, batch_queue, update_step):
    """Worker that moves transitions from replay buffer
    to the batch_queue.

    Args:
        config: configuration dict.
        replay_queue: queue with transitions.
        batch_queue: queue with batches for NN
        update_step: overall system step (learner update step).
    """
    logger_path = os.path.join(config["experiment_dir"], "data_struct")
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    logger = Logger(logger_path)
    replay_buffer = ReplayBuffer(config["state_dim"], config["action_dim"])
    step_prev = 0

    while True:
        n = replay_queue.qsize()
        for _ in range(n):
            transition = replay_queue.get()
            replay_buffer.add(*transition)

        if len(replay_buffer) < config["batch_size"]:
            continue

        batch = replay_buffer.sample(config["batch_size"])
        try:
            batch_queue.put_nowait(batch)
        except:
            continue

        # Log data structures
        s = update_step.value
        if s != step_prev and s % 100 == 0:
            step_prev = s
            logger.log_scalar("replay_queue", replay_queue.qsize(), s)
            logger.log_scalar("batch_queue", batch_queue.qsize(), s)
            logger.log_scalar("replay_buffer", len(replay_buffer), s)


def model_worker(config, model, batch_queue, update_step):
    """Dynamics-model learner worker.

    Args:
        config: configuration dict.
        batch_queue: queue with batches from buffer.
        update_step: overall system step (learner update step).
    """
    #model = Model(**config)
    model.train(batch_queue, update_step)


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
    policy = policy_class(**config)
    policy.train(learner_queue, batch_queue, update_step)


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
    agent = Agent(config, agent_n, policy_class)
    agent.run(replay_queue, learner_queue, update_step)


class STEVETrainer:
    def __init__(self, config):
        self.config = config

    def train(self, env):
        config = self.config

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=64)
        batch_queue = mp.Queue(maxsize=64)
        update_step = mp.Value('i', 0)
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

        # Model learner
        model = Model(**config)
        model.share_memory()
        p = mp.Process(
                target=model_worker,
                args=(config,
                      model,
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
        processes.append(p)

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
            p.join()
