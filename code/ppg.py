import os
import random
import sys
from collections import deque, namedtuple
from dataclasses import dataclass

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from helpers import (
    clipped_value_loss,
    create_agent,
    load_model_parameters,
    normalize,
    update_network_,
)

# from memory import AuxMemory, Memory
from openai_vpt.lib.tree_util import tree_map
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Memory = namedtuple(
    "Memory",
    [
        "agent_obs",
        "action",
        "action_log_prob",
        "reward",
        "done",
        "value",
        "agent_hidden_state",
        "critic_hidden_state",
    ],
)
AuxMemory = namedtuple("Memory", ["agent_obs", "target_value", "old_values"])


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
        # Save the main agent and auxiliary agent models
        pass

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

        time = 0
        updated = False
        num_policy_updates = 0

        for eps in tqdm(range(num_episodes), desc="episodes"):
            render_eps = render and eps % render_every_eps == 0
            obs = env.reset()
            dummy_first = th.from_numpy(np.array((False,))).to(self.device)

            # Reset the hidden state for the policy
            initial_critic_hidden_state = self.critic.policy.initial_state(1)
            initial_agent_hidden_state = self.agent.policy.initial_state(1)
            agent_hidden_state = initial_agent_hidden_state
            critic_hidden_state = initial_critic_hidden_state

            for timestep in range(max_timesteps):
                time += 1

                # if updated and render_eps:
                #     env.render()
                env.render()

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

                # Can't store hidden states since we run out of memory
                memory = Memory(
                    agent_obs,
                    action,
                    result["log_prob"],
                    reward,
                    done,
                    value,
                    0,
                    0,
                )

                memories.append(memory)

                # Update the hidden state for the next timestep
                agent_hidden_state = new_agent_state
                critic_hidden_state = new_critic_state
                obs = next_obs

                if time % update_timesteps == 0:
                    self.learn(
                        memories,
                        aux_memories,
                        next_obs,
                        new_critic_state,
                        initial_agent_hidden_state,
                        initial_critic_hidden_state,
                    )
                    num_policy_updates += 1
                    memories.clear()

                    if num_policy_updates % num_policy_updates_per_aux == 0:
                        self.learn_aux(aux_memories)
                        aux_memories.clear()

                    updated = True

                if done:
                    if render_eps:
                        updated = False
                    break

            if render_eps:
                env.close()

            if eps % save_every == 0:
                self.save()

    def learn(
        self,
        memories,
        aux_memories,
        next_obs,
        critic_hidden_state,
        initial_agent_hidden_state,
        initial_critic_hidden_state,
    ):
        # prepare the memories
        agent_observations = [memory.agent_obs for memory in memories]
        actions = [memory.action for memory in memories]
        old_log_probs = [memory.action_log_prob for memory in memories]
        rewards = [memory.reward for memory in memories]
        masks = [1 - float(memory.done) for memory in memories]
        values = [memory.value for memory in memories]

        # Get the value prediction for the next observation
        dummy_first = th.from_numpy(np.array((False,))).to(self.device)
        agent_obs = self.agent._env_obs_to_agent(next_obs)
        next_value, _ = self.critic.policy.v(
            agent_obs, dummy_first, critic_hidden_state
        )
        values = values + [next_value]

        returns = self.calculate_gae(
            rewards,
            values,
            masks,
            self.gamma,
            self.lam,
        )

        # convert values to torch tensors
        to_torch_tensor = lambda t: th.stack(t).to(self.device).detach()
        # Detach all the tensors
        old_log_probs = to_torch_tensor(old_log_probs)
        old_values = to_torch_tensor(values[:-1])

        rewards = th.tensor(returns).float().to(self.device)

        aux_memory = AuxMemory(agent_observations, rewards, old_values)
        aux_memories.append(aux_memory)

        # Policy phase training (PPO)
        for _ in range(self.epochs):
            agent_hidden_state = initial_agent_hidden_state
            critic_hidden_state = initial_critic_hidden_state

            # Try to create a dummy first tensor for the batch
            dummy_first = th.from_numpy(
                np.array((False,) * self.minibatch_size)
            ).to(self.device)

            print(f"dummy_first: {dummy_first.shape}")

            for start in range(0, len(memories), self.minibatch_size):
                end = start + self.minibatch_size
                minibatch = slice(start, end)
                # Get the minibatch
                agent_obs_batch = batch_agent_obs(agent_observations[minibatch])
                actions_batch = actions[minibatch]
                old_log_probs_batch = old_log_probs[minibatch]
                old_values_batch = old_values[minibatch]
                rewards_batch = rewards[minibatch]

                action_log_probs = []
                values = []

                print(agent_obs_batch)

                # Test batch pi and value predictions
                (pi_dist, _), _ = self.agent.policy.net(
                    agent_obs_batch,
                    agent_hidden_state,
                    context={"first": dummy_first},
                )

                (_, values), _ = self.critic.policy.net(
                    agent_obs_batch,
                    critic_hidden_state,
                    context={"first": dummy_first},
                )

                # get log probs

                action_log_probs = self.agent.policy.pi_head.logprob(
                    actions_batch, pi_dist
                )

                # # Calculate pi distribution, log probs, and values for the batch
                # for a_obs, ac in zip(agent_obs, actions_batch):
                #     (
                #         pi_distribution,
                #         _,
                #         new_agent_state,
                #     ) = self.agent.policy.get_output_for_observation(
                #         a_obs, agent_hidden_state, dummy_first
                #     )
                #     log_prob = self.agent.policy.get_logprob_of_action(
                #         pi_distribution, ac
                #     )
                #     action_log_probs.append(log_prob)

                #     (
                #         _,
                #         v_prediction,
                #         new_critic_state,
                #     ) = self.critic.policy.get_output_for_observation(
                #         a_obs, critic_hidden_state, dummy_first
                #     )
                #     values.append(v_prediction)

                #     agent_state = new_agent_state
                #     critic_state = new_critic_state

                # action_log_probs = th.stack(action_log_probs).to(self.device)
                # values = th.stack(values).to(self.device)

                ratios = (action_log_probs - old_log_probs_batch).exp()
                advantages = normalize(
                    (rewards_batch - old_values_batch.detach())
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

    def learn_aux(self, aux_memories):
        # Gather states and target values into one tensor

        # Get old action predictions for minimizing kl divergence and clipping respectively

        # Create the dataloader for auxiliary phase training

        # The proposed auxiliary phase training
        pass

    def calculate_gae(self, rewards, values, masks, gamma, lam):
        """
        Calculate the generalized advantage estimate
        """
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            next_value = values[step + 1] if step < len(rewards) - 1 else 0
            delta = (
                rewards[step] + gamma * next_value * masks[step] - values[step]
            )
            gae = delta + gamma * lam * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns


def batch_agent_obs(agent_obs_list):
    keys = agent_obs_list[0].keys()
    batched_agent_obs = {}

    for key in keys:
        batched_agent_obs[key] = th.stack(
            [agent_obs[key] for agent_obs in agent_obs_list]
        )

    return batched_agent_obs
