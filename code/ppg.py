from collections import deque

import gym
import numpy as np
import torch as th
from helpers import (
    calculate_gae,
    clipped_value_loss,
    create_agent,
    load_model_parameters,
    normalize,
    update_network_,
)
from memory import AuxMemory, Memory, create_dataloader

# from memory import AuxMemory, Memory
from openai_vpt.lib.tree_util import tree_map
from torch.optim import Adam
from tqdm import tqdm


class PPG:
    def __init__(
        self,
        env_name,
        model,
        weights,
        out_weights,
        device,
        reward_predictor,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        gamma,
        lam,
        clip,
        value_clip,
    ):
        self.epochs = epochs
        self.epochs_aux = epochs_aux
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.value_clip = value_clip
        self.env_name = env_name
        self.reward_predictor = reward_predictor
        self.device = device
        self.out_weights = out_weights

        # Load model parameters and create agents
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(model)

        # Create the main agent and auxiliary agent here
        self.agent = create_agent(
            env_name, agent_policy_kwargs, agent_pi_head_kwargs, weights
        )
        self.critic = create_agent(
            env_name, agent_policy_kwargs, agent_pi_head_kwargs, weights
        )
        self.agent_optim = Adam(
            self.agent.policy.parameters(),
            lr=self.lr,
            betas=self.betas,
        )

        critic_params = list(self.critic.policy.value_head.parameters()) + list(
            self.critic.policy.net.parameters()
        )
        self.critic_optim = Adam(
            critic_params,
            lr=self.lr,
            betas=self.betas,
        )

        # TODO: Implement scheduler

    def save(self):
        """
        Saves the model weights to out_weights path
        """
        th.save(self.agent.policy.state_dict(), self.out_weights)
        print(f"Saved model weights to {self.out_weights}")

    def collect_episodes(self, num_episodes_to_collect, max_timesteps, env):
        all_episodes_memories = []
        inital_hidden_states = {"agent": [], "critic": []}

        for _ in range(num_episodes_to_collect):
            episode_memories = deque([])
            obs = env.reset()
            dummy_first = th.from_numpy(np.array((False,))).to(self.device)
            # Reset the hidden state for the policy
            initial_critic_hidden_state = self.critic.policy.initial_state(1)
            initial_agent_hidden_state = self.agent.policy.initial_state(1)
            agent_hidden_state = initial_agent_hidden_state
            critic_hidden_state = initial_critic_hidden_state
            inital_hidden_states["agent"].append(agent_hidden_state)
            inital_hidden_states["critic"].append(critic_hidden_state)

            for timestep in range(max_timesteps):
                # Preprocess the observation
                agent_obs = self.agent._env_obs_to_agent(obs)

                # get the action and next hidden state, log prob
                action, new_agent_state, result = self.agent.policy.act(
                    agent_obs, dummy_first, agent_hidden_state
                )

                # get the value prediction and next hidden state for the critic
                value, new_critic_state = self.critic.policy.v(
                    agent_obs, dummy_first, critic_hidden_state
                )

                # Take the action in the environment
                minerl_action = self.agent._agent_action_to_env(action)
                minerl_action["ESC"] = 0
                next_obs, _, done, _ = env.step(minerl_action)

                # convert to NCHW
                input_tensor = agent_obs["img"].permute(0, 3, 1, 2)
                # Predict the reward for the observation
                reward = self.reward_predictor.reward(input_tensor)

                if done or timestep == max_timesteps - 1:
                    # Compute the value for the next observation
                    next_value, _ = self.critic.policy.v(
                        self.agent._env_obs_to_agent(next_obs),
                        dummy_first,
                        critic_hidden_state,
                    )

                    # Add the next_value to the last memory of the episode
                    memory = Memory(
                        agent_obs,
                        action,
                        result["log_prob"],
                        reward,
                        done,
                        value,
                        next_value,
                    )

                    episode_memories.append(memory)

                    break

                memory = Memory(
                    agent_obs,
                    action,
                    result["log_prob"],
                    reward,
                    done,
                    value,
                    None,
                )

                episode_memories.append(memory)

                # Update the hidden state for the next timestep
                agent_hidden_state = new_agent_state
                critic_hidden_state = new_critic_state
                obs = next_obs
            all_episodes_memories.append(episode_memories)
        return all_episodes_memories, inital_hidden_states

    def train(
        self,
        num_episodes,
        max_timesteps,
        render,
        render_every_eps,
        save_every,
        update_timesteps,
        num_policy_updates_per_aux,
    ):
        env = gym.make(self.env_name)

        memories = deque([])
        aux_memories = deque([])
        num_policy_updates = 0

        for eps in tqdm(range(num_episodes), desc="episodes"):
            render_eps = render and eps % render_every_eps == 0

            (
                all_episodes_memories,
                initial_hidden_states,
            ) = self.collect_episodes(1, max_timesteps, env)

            # Perform learning for each episode's memories
            for episode_id, episode_memories in enumerate(
                all_episodes_memories
            ):
                self.learn(
                    episode_memories,
                    aux_memories,
                    initial_hidden_states,
                    episode_id,
                )

                num_policy_updates += 1

                # Perform auxiliary learning if required
                if num_policy_updates % num_policy_updates_per_aux == 0:
                    self.learn_aux(aux_memories)
                    aux_memories.clear()

            if render_eps:
                env.close()

            if eps % save_every == 0:
                self.save()

    def learn(self, memories, aux_memories, initial_hidden_states, episode_id):
        # prepare the memories
        agent_observations = [memory.agent_obs for memory in memories]
        actions = [memory.action for memory in memories]
        old_log_probs = [memory.action_log_prob for memory in memories]
        rewards = [memory.reward for memory in memories]
        masks = [1 - float(memory.done) for memory in memories]
        values = [memory.value for memory in memories]

        # Use the next_value from the last memory in the episode
        next_value = memories[-1].next_value
        values = values + [next_value]

        returns = calculate_gae(
            rewards,
            values,
            masks,
            self.gamma,
            self.lam,
        )

        # Convert to tensors and move to device
        rewards = th.tensor(returns).float().to(self.device)
        old_values = th.tensor(values[:-1]).to(self.device)
        old_log_probs = th.tensor(old_log_probs).to(self.device)

        aux_memory = AuxMemory(agent_observations, rewards, old_values)
        aux_memories.append(aux_memory)

        # Dont shuffle the data
        dl = create_dataloader(
            [agent_observations, actions, old_log_probs, rewards, old_values],
            batch_size=self.minibatch_size,
        )

        # Policy phase training (PPO)
        for _ in range(self.epochs):
            agent_hidden_states = tree_map(
                lambda x: x.detach(), initial_hidden_states["agent"][episode_id]
            )
            critic_hidden_state = tree_map(
                lambda x: x.detach(),
                initial_hidden_states["critic"][episode_id],
            )

            dummy_first = th.from_numpy(np.array((False,))).to(self.device)

            for (
                agent_obs_batch,
                actions_batch,
                old_log_probs_batch,
                rewards_batch,
                old_values_batch,
            ) in dl:
                log_probs = []
                values = []

                # Unstack the agent_obs_batch and actions_batch
                obs_tensors = th.unbind(agent_obs_batch["img"])
                buttons_tensors = th.unbind(actions_batch["buttons"])
                camera_tensors = th.unbind(actions_batch["camera"])

                # Iterate over the unstacked tensors directly
                for i, (
                    obs_tensor,
                    buttons_tensor,
                    camera_tensor,
                ) in enumerate(
                    zip(obs_tensors, buttons_tensors, camera_tensors)
                ):
                    agent_obs_batch["img"] = obs_tensor
                    actions_batch["buttons"] = buttons_tensor
                    actions_batch["camera"] = camera_tensor
                    # Run the agent to get the policy distribution
                    (
                        pi_distribution,
                        _,
                        next_agent_hidden_state,
                    ) = self.agent.policy.get_output_for_observation(
                        agent_obs_batch, agent_hidden_states, dummy_first
                    )

                    # Run the disjoint value network
                    (
                        _,
                        v_prediction,
                        next_critic_hidden_state,
                    ) = self.critic.policy.get_output_for_observation(
                        agent_obs_batch,
                        critic_hidden_state,
                        dummy_first,
                    )

                    # Calculate the log prob
                    action_log_prob = self.agent.policy.get_logprob_of_action(
                        pi_distribution, actions_batch
                    )

                    values.append(v_prediction)
                    log_probs.append(action_log_prob)

                    # Update hidden states for the next iteration
                    agent_hidden_states = next_agent_hidden_state
                    critic_hidden_state = next_critic_hidden_state

                # Stack the values and log_probs
                values = th.stack(values).to(self.device)
                log_probs = th.stack(log_probs).to(self.device)

                ratios = (log_probs - old_log_probs_batch).exp()
                advantages = normalize(
                    rewards_batch.detach() - old_values_batch.detach()
                )
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.clip, 1 + self.clip) * advantages
                policy_loss = -th.min(surr1, surr2)  # - self.betas  # * entropy

                update_network_(policy_loss, self.agent_optim)

                value_loss = clipped_value_loss(
                    values,
                    rewards_batch,
                    old_values_batch,
                    self.value_clip,
                )

                update_network_(value_loss, self.critic_optim)

            # for start in range(0, len(memories), self.minibatch_size):
            #     end = start + self.minibatch_size
            #     minibatch = slice(start, end)
            #     # Get the minibatch
            #     agent_obs_batch = batch_agent_obs(agent_observations[minibatch])
            #     actions_batch = actions[minibatch]
            #     old_log_probs_batch = old_log_probs[minibatch]
            #     old_values_batch = old_values[minibatch]
            #     rewards_batch = rewards[minibatch]

            #     action_log_probs = []
            #     values = []

            #     # # Calculate pi distribution, log probs, and values for the batch
            #     # for a_obs, ac in zip(agent_obs, actions_batch):
            #     #     (
            #     #         pi_distribution,
            #     #         _,
            #     #         new_agent_state,
            #     #     ) = self.agent.policy.get_output_for_observation(
            #     #         a_obs, agent_hidden_state, dummy_first
            #     #     )
            #     #     log_prob = self.agent.policy.get_logprob_of_action(
            #     #         pi_distribution, ac
            #     #     )
            #     #     action_log_probs.append(log_prob)

            #     #     (
            #     #         _,
            #     #         v_prediction,
            #     #         new_critic_state,
            #     #     ) = self.critic.policy.get_output_for_observation(
            #     #         a_obs, critic_hidden_state, dummy_first
            #     #     )
            #     #     values.append(v_prediction)

            #     #     agent_state = new_agent_state
            #     #     critic_state = new_critic_state

            #     # action_log_probs = th.stack(action_log_probs).to(self.device)
            #     # values = th.stack(values).to(self.device)

            #     ratios = (action_log_probs - old_log_probs_batch).exp()
            #     advantages = normalize(
            #         (rewards_batch - old_values_batch.detach())
            #     )
            #     surr1 = ratios * advantages
            #     surr2 = ratios.clamp(1 - self.clip, 1 + self.clip) * advantages
            #     policy_loss = -th.min(surr1, surr2)  # - self.betas  # * entropy

            #     update_network_(policy_loss, self.agent_optim)

            #     value_loss = clipped_value_loss(
            #         values,
            #         rewards_batch,
            #         old_values_batch,
            #         self.value_clip,
            #     )

            #     update_network_(value_loss, self.critic_optim)

    def learn_aux(self, aux_memories):
        # Gather states and target values into one tensor
        agent_observations = [aux.agent_obs for aux in aux_memories]
        rewards = [aux.rewards for aux in aux_memories]
        old_values = [aux.old_values for aux in aux_memories]

        agent_observations = th.cat(agent_observations)
        rewards = th.cat(rewards)
        old_values = th.cat(old_values)

        # Get old action predictions for minimizing kl divergence and clipping respectively

        # Create the dataloader for auxiliary phase training

        # The proposed auxiliary phase training
        pass
