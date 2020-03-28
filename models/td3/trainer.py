class TD3TRainer:
    def __init__(self, config):
        self.config = config

        # Speicific TD3 config
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq

    def train(self):
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device=args.device)
        
        # Evaluate untrained policy
        evaluations = [eval_policy(policy, args.env, args.seed)]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(args.max_timesteps)):
            
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size, t)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % args.eval_freq == 0:
                evaluations.append(eval_policy(policy, args.env, args.seed))
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model: policy.save(f"./models/{file_name}")
