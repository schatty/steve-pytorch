import numpy as np

from .utils import ReplayBuffer
from .td3 import TD3


class TD3Trainer:
    def __init__(self, config):
        # Speicific TD3 config
        config["policy_noise"] = 0.2 * config["max_action"]
        config["noise_clip"] = 0.5 * config["max_action"]
        config["policy_freq"] = 2

        self.config = config

    def train(self, env, eval_func):
        config = self.config
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        max_action = config["max_action"]
        expl_noise = config["expl_noise"]
        batch_size = config["batch_size"]

        replay_buffer = ReplayBuffer(state_dim, action_dim,
                                     device=config["device"])
        policy = TD3(**config)

        # Evaluate untrained policy
        evaluations = [eval_func(policy, config["env"], config["seed"])]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(config["max_timesteps"])):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < config["start_timesteps"]:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise,
                                       size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= config["start_timesteps"]:
                policy.train(replay_buffer, batch_size, t)

            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % config["eval_freq"] == 0:
                # TODO: Save policy
                evaluations.append(eval_func(policy, config["env"], config["seed"]))
                np.save(f"{config['experiment_dir']}/reward", evaluations)
