import pickle

import gym
import torch
from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent


def load_model_parameters(path_to_model_file):
    with open(path_to_model_file, "rb") as f:
        agent_parameters = pickle.load(f)
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def create_agent(
    env_name, policy_kwargs, pi_head_kwargs, in_weights, DEVICE="cuda"
):
    env = gym.make(env_name)
    agent = MineRLAgent(
        env,
        device=DEVICE,
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
    )
    agent.load_weights(in_weights)
    env.close()
    return agent


def freeze_policy_layers(policy, trainable_layers):
    for param in policy.parameters():
        param.requires_grad = False
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True
    return [param for layer in trainable_layers for param in layer.parameters()]


def exists(val):
    return val is not None


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)


def update_network_(loss, optimizer, policy, max_grad_norm=0.5):
    optimizer.zero_grad()
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.squeeze() - rewards) ** 2
    value_loss_2 = (values.squeeze() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))


def calculate_gae(rewards, values, masks, gamma, lam):
    """
    Calculate the generalized advantage estimate
    """
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        next_value = values[step + 1] if step < len(rewards) - 1 else 0
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
