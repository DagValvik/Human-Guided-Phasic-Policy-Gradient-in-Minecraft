import os
import pickle
import time
from argparse import ArgumentParser

import gym
import minerl
import numpy as np
import torch as th
from data_loader import DataLoader
from helpers import create_agent, load_model_parameters
from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from openai_vpt.lib.tree_util import tree_map

EPOCHS = 2
BATCH_SIZE = 16
N_WORKERS = 24
DEVICE = "cuda"

LOSS_REPORT_RATE = 100
LEARNING_RATE = 0.000181  # OpenAI VPT Paper
WEIGHT_DECAY = 0.039428  # OpenAI VPT Paper
# WEIGHT_DECAY = 0.0
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0
MAX_BATCHES = 40000  # Max number of batches to train for


def behavior_cloning_train(
    data_dir, in_model, in_weights, out_weights, env_name
):
    # Load model parameters and create agents
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    agent = create_agent(
        env_name,
        agent_policy_kwargs,
        agent_pi_head_kwargs,
        in_weights,
    )
    original_agent = create_agent(
        env_name,
        agent_policy_kwargs,
        agent_pi_head_kwargs,
        in_weights,
    )

    policy = agent.policy
    original_policy = original_agent.policy

    # Freeze most params if using small dataset
    trainable_layers = [policy.net.lastlayer, policy.pi_head]
    trainable_parameters = policy.parameters()

    # Set up optimizer and learning rate scheduler
    optimizer = th.optim.Adam(
        trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Initialize data loader
    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
    )

    # Training loop
    start_time = time.time()
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)
    loss_sum = 0

    for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(
        data_loader
    ):
        batch_loss = 0
        for image, action, episode_id in zip(
            batch_images, batch_actions, batch_episode_id
        ):
            if image is None and action is None:
                # A work-item was done. Remove hidden state
                if episode_id in episode_hidden_states:
                    removed_hidden_state = episode_hidden_states.pop(episode_id)
                    del removed_hidden_state
                    continue
            agent_action = agent._env_action_to_agent(
                action, to_torch=True, check_if_null=True
            )
            if agent_action is None:
                # Action was null
                continue

            agent_obs = agent._env_obs_to_agent({"pov": image})
            if episode_id not in episode_hidden_states:
                episode_hidden_states[episode_id] = policy.initial_state(1)
            agent_state = episode_hidden_states[episode_id]

            (
                pi_distribution,
                _,
                new_agent_state,
            ) = policy.get_output_for_observation(
                agent_obs, agent_state, dummy_first
            )

            with th.no_grad():
                (
                    original_pi_distribution,
                    _,
                    _,
                ) = original_policy.get_output_for_observation(
                    agent_obs, agent_state, dummy_first
                )

            log_prob = policy.get_logprob_of_action(
                pi_distribution, agent_action
            )
            kl_div = policy.get_kl_of_action_dists(
                pi_distribution, original_pi_distribution
            )

            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(
                f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f} "
            )
            loss_sum = 0

    # Save the finetuned weights
    state_dict = policy.state_dict()
    th.save(state_dict, out_weights)
    print("Saved weights to", out_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing recordings to be trained on",
    )
    parser.add_argument(
        "--in-model",
        required=True,
        type=str,
        help="Path to the .model file to be finetuned",
    )
    parser.add_argument(
        "--in-weights",
        required=True,
        type=str,
        help="Path to the .weights file to be finetuned",
    )
    parser.add_argument(
        "--out-weights",
        required=True,
        type=str,
        help="Path where finetuned weights will be saved",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MineRLBasaltFindCave-v0",
        help="Name of the environment to be used",
    )
    args = parser.parse_args()
    behavior_cloning_train(
        args.data_dir,
        args.in_model,
        args.in_weights,
        args.out_weights,
        args.env_name,
    )
