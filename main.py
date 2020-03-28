import numpy as np
import torch
import gym
import pybulletgym
import argparse
import os
import yaml

from models.trainer import load_trainer


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config


# TODO: Move comments to the help arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="STEVE")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=100_000, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--device", default="cuda")                 # Name of the GPU device
    parser.add_argument("--eval_freq", default=5e3, type=int) # How often  (time steps) we evaluate
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # TODO: Create proper experiment folder
    experiment_path = os.path.join("./results", "experiment")
    print("Experiment path: ", experiment_path)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    config = {**read_config(args.config), **vars(args)}

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    config["state_dim"] = state_dim
    config["action_dim"] = action_dim
    config["max_action"] = max_action
    config["experiment_dir"] = experiment_path

    # Initialize policy
    trainer = load_trainer(config)
    trainer.train(env)
