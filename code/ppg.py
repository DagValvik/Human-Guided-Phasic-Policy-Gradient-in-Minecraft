import os

import numpy as np
import torch as th
import torch.nn.functional as F
from helpers import create_agent, load_model_parameters
from memory import AuxMemory, Memory, MemoryDataset
from openai_vpt.lib.tree_util import tree_map
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


class PPG:
    def __init__(
        self,
        env_name,
        model,
        weights,
        out_weights,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        device,
    ):
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
        self.opt_agent = Adam(
            self.agent.policy.parameters(), lr=lr, betas=betas
        )
        self.opt_aux = Adam(
            self.auxiliary.policy.parameters(), lr=lr, betas=betas
        )
        # Other PPG parameters initialization
        self.epochs = epochs
        self.epochs_aux = epochs_aux
        self.minibatch_size = minibatch_size
        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip

    def save(self):
        # Save the main agent and auxiliary agent models
        pass

    def learn(self, memories, aux_memories):
        # Calculate the generalized advantage estimate
        v_predictions = [mem.value for mem in memories]
        rewards = [mem.reward for mem in memories]
        masks = [1 - float(mem.done) for mem in memories]

        returns = self.calculate_gae(
            rewards, v_predictions, masks, self.gamma, self.lam
        )

        # Store obs and target values to auxiliary memory buffer for later training
        aux_memory = AuxMemory(
            [mem.agent_obs for mem in memories],
            returns,
            [mem.done for mem in memories],
        )
        aux_memories.append(aux_memory)

        # Create the dataloader for policy phase training
        dataset = MemoryDataset(memories)
        dataloader = DataLoader(
            dataset, batch_size=self.minibatch_size, shuffle=True
        )
        # Policy phase training
        for _ in range(self.epochs):
            # Iterate through the minibatches
            for batch in dataloader:
                # Calculate the policy loss and value loss
                policy_losses = []
                value_losses = []
                for mem in batch:
                    # Get required data from the memory object
                    agent_obs, action, old_log_prob, reward, done, value = mem

                    # Get the action probabilities and value predictions from the current policy
                    (pi_h, v_h), _ = self.agent.policy.net(agent_obs, hidden_state)
                    pi_distribution = self.agent.policy.pi_head(pi_h)
                    value_prediction = self.agent.policy.value_head(v_h)

                    # Get the new log probabilities for the taken actions
                    log_probs = self.agent.policy.get_logprob_of_action(pi_distribution, action)

                    # Calculate the policy ratios = (action_log_probs - old_action_log_probs).exp().to(device)
                    policy_ratio = (log_probs - old_log_prob).exp()

                    # Calculate the advantages = returns - v_prediction.detach().to(device)
                    advantages = reward - value.detach()

                    # Calculate the surrogate objective function (unclipped)
                    surrogate_obj = policy_ratio * advantages
                    # Calculate the clipped surrogate objective
                    surrogate_clip = th.clamp(
                        policy_ratio, 1 - self.eps_clip, 1 + self.eps_clip
                    )
                    clipped_surrogate_obj = surrogate_clip * advantages

                    # Calculate the final policy loss
                    policy_loss = -th.min(surrogate_obj, clipped_surrogate_obj).mean()

                    # Calculate the value loss using the Huber loss (smooth L1 loss)
                    value_loss = F.smooth_l1_loss(value_prediction, reward)

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)

                # Calculate the total policy loss and value loss
                total_policy_loss = th.stack(policy_losses).mean()
                total_value_loss = th.stack(value_losses).mean()

                # Backprop for policy
                self.opt_agent.zero_grad()
                total_policy_loss.backward()
                self.opt_agent.step()

                # Backprop for value
                self.opt_aux.zero_grad()
                total_value_loss.backward()
                self.opt_aux.step()

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
