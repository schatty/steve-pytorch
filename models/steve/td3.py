import os
import copy
import torch
import torch.nn.functional as F

from ..utils import Logger, eval_policy
from ..networks import Actor, Critic

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device="cuda",
        **kwargs
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.device = device

        self.upload_policy_step = int(kwargs["upload_policy"])
        self.eval_freq = int(kwargs["eval_freq"])
        self.max_timesteps = kwargs["max_timesteps"]
        self.agent_device = kwargs["agent_device"]

        # for evaluation
        self.env_name = kwargs["env"]
        self.seed = kwargs["seed"]

        logger_path = os.path.join(kwargs["experiment_dir"], "learner")
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)
        self.logger = Logger(logger_path)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, learner_queue, batch_queue, update_step):
        while self.total_it < self.max_timesteps:
            self.total_it += 1
            update_step.value += 1

            if self.total_it % 100 == 0:
                print("Training step : ", self.total_it)

            # Sample replay buffer
            state, action, next_state, reward, not_done = batch_queue.get()
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.from_numpy(reward).float().to(self.device)
            not_done = torch.from_numpy(not_done).float().to(self.device)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor losse
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.total_it % self.upload_policy_step == 0:
                print("UPLOADIGN!!")
                self.upload_policy(learner_queue)

            # TODO: This evalution logic slows down the training in a bad way
            if self.total_it % self.eval_freq == 0:
                eval_reward = eval_policy(self, self.env_name, self.seed)
                self.logger.log_scalar("eval_reward", eval_reward, self.total_it)

            del state
            del action
            del next_state
            del reward
            del not_done

    def upload_policy(self, learner_queue):
        print("Uploading new policy")
        params = [p.data.to(self.agent_device).detach().cpu().numpy()
                  for p in self.actor.parameters()]
        learner_queue.put(params)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
