from dataclasses import dataclass
from typing import List

import numpy as np
import torch as th
from torch.utils.data import Dataset

# @dataclass
# class Memory:
#     """
#     This class represents a single frame/step of the agent
#     A full episode should be of type List[Memory]
#     """

#     # Raw pixel observation for this frame
#     agent_obs: dict
#     hidden_state: list
#     pi_h: th.tensor
#     v_h: th.tensor
#     action: np.ndarray
#     action_log_prob: np.ndarray
#     reward: float
#     done: bool
#     value: float


# @dataclass
# class AuxMemory:
#     """
#     This class represents auxillary memory for PPG
#     Only has an obs and the target value (return)
#     Stored in `B` in paper
#     """

#     # Raw pixel observation for this frame
#     agent_obs: dict
#     v_targ: float
#     done: bool


class Memory(Dataset):
    """
    This is a dataset of memory objects (potentially multiple episodes!)
    This is to be used with the PyTorch DataLoader
    """

    def __init__(self):
        self.agent_obs = []
        self.hidden_states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return (
            self.agent_obs[idx],
            self.hidden_states[idx],
            self.actions[idx],
            self.action_log_probs[idx],
            self.rewards[idx],
            self.dones[idx],
            self.values[idx],
        )

    def save(
        self,
        agent_obs,
        hidden_state,
        action,
        action_log_prob,
        reward,
        done,
        value,
    ):
        self.agent_obs.append(agent_obs)
        self.hidden_states.append(hidden_state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def save_all(
        self,
        agent_obs,
        hidden_states,
        actions,
        action_log_probs,
        rewards,
        dones,
        values,
    ):
        self.agent_obs.extend(agent_obs)
        self.hidden_states.extend(hidden_states)
        self.actions.extend(actions)
        self.action_log_probs.extend(action_log_probs)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        self.values.extend(values)

    def clear(self):
        del self.agent_obs[:]
        del self.hidden_states[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]


class AuxMemory(Dataset):
    """
    This is a dataset of memory objects (potentially multiple episodes!)
    This is to be used with the PyTorch DataLoader
    """

    def __init__(self):
        self.agent_obs = []
        self.returns = []
        self.dones = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return (
            self.agent_obs[idx],
            self.returns[idx],
            self.dones[idx],
        )

    def save(self, agent_obs, return_, done):
        self.agent_obs.append(agent_obs)
        self.returns.append(return_)
        self.dones.append(done)

    def save_all(self, agent_obs, returns, dones):
        self.agent_obs.extend(agent_obs)
        self.returns.extend(returns)
        self.dones.extend(dones)

    def clear(self):
        del self.agent_obs[:]
        del self.returns[:]
        del self.dones[:]
        