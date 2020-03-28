import os
import gym
import numpy as np
import torch

from .utils import Logger


class Agent:
    def __init__(self, config, agent_n, policy_class):
        self.config = config
        self.agent_n = agent_n
        logger_path = os.path.join(config["experiment_dir"],
                                   f"agent_{agent_n}")

        if not os.path.exists(logger_path):
            os.makedirs(logger_path)

        self.logger = Logger(logger_path)
        self.policy = policy_class(**config)

    def update_policy(self, learner_queue):
        """Update local policy to one from learner queue. """
        print("Updating policy for agent: ", self.agent_n)
        try:
            source = learner_queue.get_nowait()
        except:
            return
        target = self.policy.actor
        for target_p, source_p in zip(target.parameters(), source):
            w = torch.tensor(source_p).float()
            target_p.data.copy_(w)
        del source

    def run(self, replay_queue, learner_queue, step):
        """Run agent collecting data from sim.

        Args:
            replay_queue: queue with transitions.
            learner_queue: queue with latest weights from learner.
            step: overall system state (learner update step).
        """
        config = self.config

        env = gym.make(config["env"])
        max_action = config["max_action"]
        action_dim = config["action_dim"]
        # TODO: All logic with expl noise should be done within policy
        expl_noise = 0.1

        state, done = env.reset(), False
        episode_reward = 0
        episode_num = 0

        t = 0
        episode_timesteps = 0
        while True:
            t += 1

            if t < config["max_timesteps"]:
                action = env.action_space.sample()
            else:
                action = (
                    self.policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise,
                                       size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_queue.put((state, action, next_state, reward, done_bool))

            state = next_state
            episode_reward += reward

            # Update policy once in a while
            if t % config["agent_update"] == 0:
                self.update_policy(learner_queue)

            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
