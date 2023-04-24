import os
import random
import sys
from collections import deque
from dataclasses import dataclass

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from helpers import create_agent, load_model_parameters
from memory import AuxMemory, Memory
from openai_vpt.lib.tree_util import tree_map
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class PPOParams:
    epochs: int
    minibatch_size: int
    lr: float
    betas: tuple
    gamma: float
    lam: float
    eps_clip: float
    value_clip: float


@dataclass
class AuxParams:
    epochs: int
    num_policy_updates_per_aux: int


class PPG:
    def __init__(
        self,
        env_name,
        model,
        weights,
        out_weights,
        ppo_params: PPOParams,
        aux_params: AuxParams,
        device,
        reward_predictor=None,
    ):
        self.ppo_params = ppo_params
        self.aux_params = aux_params
        self.env_name = env_name
        self.reward_predictor = reward_predictor
        self.device = device

        # Load model parameters and create agents
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(model)

        # Create the main agent and auxiliary agent here
        self.agent = create_agent(
            env_name, agent_policy_kwargs, agent_pi_head_kwargs, weights
        )
        self.auxiliary = create_agent(
            env_name, agent_policy_kwargs, agent_pi_head_kwargs, weights
        )
        self.agent_optim = Adam(
            self.agent.policy.parameters(),
            lr=self.ppo_params.lr,
            betas=self.ppo_params.betas,
        )

        auxiliary_params = list(
            self.auxiliary.policy.value_head.parameters()
        ) + list(self.auxiliary.policy.net.parameters())
        self.auxiliary_optim = Adam(
            auxiliary_params,
            lr=self.ppo_params.lr,
            betas=self.ppo_params.betas,
        )

        # TODO: Implement scheduler

    def save(self):
        # Save the main agent and auxiliary agent models
        pass

    def train(
        self,
        num_episodes,
        max_timesteps,
        render=False,
        render_every_eps=250,
        save_every=1000,
        update_timesteps=32,
        num_policy_updates_per_aux=5,
    ):
        env = gym.make(self.env_name)

        memories = Memory()
        aux_memories = AuxMemory()

        time = 0
        updated = False
        num_policy_updates = 0

        for eps in tqdm(range(num_episodes), desc="episodes"):
            render_eps = render and eps % render_every_eps == 0
            obs = env.reset()
            dummy_first = th.from_numpy(np.array((False,))).to(self.device)
            hidden_state = None
            done = False
            for timestep in range(max_timesteps):
                time += 1

                if updated and render_eps:
                    env.render()

                if hidden_state is None:
                    # Get initial state for policy
                    hidden_state = self.agent.policy.initial_state(1)

                # Preprocess the observation
                agent_obs = self.agent._env_obs_to_agent(obs)

                with th.no_grad():
                    # Get the action and value prediction
                    (
                        pi_distribution,
                        v_prediction,
                        next_hidden_state,
                    ) = self.agent.policy.get_output_for_observation(
                        agent_obs, hidden_state, dummy_first
                    )

                action = self.agent.get_action(obs)

                agent_action = self.agent._env_action_to_agent(
                    action, to_torch=True, check_if_null=True
                )

                log_prob = self.agent.policy.get_logprob_of_action(
                    pi_distribution, agent_action
                )

                # Take the action in the environment
                minerl_action = self.agent._agent_action_to_env(agent_action)
                minerl_action["ESC"] = 0
                next_obs, _, next_done, _ = env.step(minerl_action)

                # convert to NCHW
                input_tensor = agent_obs["img"].permute(0, 3, 1, 2)
                # Predict the reward for the observation
                reward = self.reward_predictor.reward(input_tensor)

                memories.save(
                    agent_obs=agent_obs,
                    action=agent_action,
                    action_log_prob=log_prob,
                    value=v_prediction,
                    reward=reward,
                    done=done,
                    hidden_state=hidden_state,
                )

                # Update the hidden state for the next iteration
                hidden_state = next_hidden_state
                obs = next_obs
                done = next_done

                if time % update_timesteps == 0:
                    self.learn(memories, aux_memories)
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

    def learn(self, memories, aux_memories):
        # Calculate the generalized advantage estimate
        v_predictions = memories.values
        rewards = memories.rewards
        masks = [1 - float(done) for done in memories.dones]

        returns = self.calculate_gae(
            rewards,
            v_predictions,
            masks,
            self.ppo_params.gamma,
            self.ppo_params.lam,
        )

        # Store obs and target values to auxiliary memory buffer for later training
        aux_memories.save_all(
            agent_obs=memories.agent_obs,
            returns=returns,
            dones=memories.dones,
        )

        dummy_first = th.from_numpy(np.array((False,))).to(self.device)

        indices = list(range(len(memories)))
        random.shuffle(indices)

        # Policy phase training
        for _ in range(self.ppo_params.epochs):
            for i in range(0, len(indices), self.ppo_params.minibatch_size):
                batch_indices = indices[i : i + self.ppo_params.minibatch_size]
                batch_agent_obs = [
                    memories.agent_obs[idx] for idx in batch_indices
                ]
                batch_hidden_states = [
                    memories.hidden_states[idx] for idx in batch_indices
                ]
                batch_actions = [memories.actions[idx] for idx in batch_indices]
                batch_action_log_probs = [
                    memories.action_log_probs[idx] for idx in batch_indices
                ]
                batch_rewards = [memories.rewards[idx] for idx in batch_indices]
                batch_dones = [memories.dones[idx] for idx in batch_indices]
                batch_values = [memories.values[idx] for idx in batch_indices]

                # Convert actions and old_log_probs to tensors
                old_log_probs = th.tensor(
                    batch_action_log_probs, dtype=th.float32, requires_grad=True
                ).to(self.device)

                rewards = th.tensor(batch_rewards, dtype=th.float32).to(
                    self.device
                )
                values = th.tensor(
                    batch_values, dtype=th.float32, requires_grad=True
                ).to(self.device)

                log_probs = []

                # Calculate pi distribution and log probs for batch
                for i in range(len(batch_dones)):
                    hidden_state = batch_hidden_states[i]
                    agent_obs = batch_agent_obs[i]
                    agent_action = batch_actions[i]

                    with th.no_grad():
                        (
                            pi_distribution,
                            _,
                            _,
                        ) = self.agent.policy.get_output_for_observation(
                            agent_obs, hidden_state, dummy_first
                        )

                    log_prob = self.agent.policy.get_logprob_of_action(
                        pi_distribution, agent_action
                    )

                    log_probs.append(log_prob)

                log_probs = th.stack(log_probs)

                # Calculate the policy ratios
                policy_ratio = (log_probs - old_log_probs).exp()

                # Calculate the advantages
                advantages = rewards - values.detach()

                # Calculate the surrogate objective function (unclipped)
                surrogate_obj = policy_ratio * advantages
                # Calculate the clipped surrogate objective
                surrogate_clip = th.clamp(
                    policy_ratio,
                    1 - self.ppo_params.eps_clip,
                    1 + self.ppo_params.eps_clip,
                )
                clipped_surrogate_obj = surrogate_clip * advantages

                # Calculate the final policy loss
                policy_loss = -th.min(
                    surrogate_obj, clipped_surrogate_obj
                ).mean()

                # Calculate the value loss using the Huber loss (smooth L1 loss)
                value_loss = F.smooth_l1_loss(values, rewards)

                # Backprop for policy
                policy_loss.backward(retain_graph=True)
                # Backprop for value
                value_loss.backward()

            self.agent_optim.step()
            self.agent_optim.zero_grad()
            self.auxiliary_optim.step()
            self.auxiliary_optim.zero_grad()

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
