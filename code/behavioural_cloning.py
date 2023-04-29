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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 2  # OpenAI VPT Paper
BATCH_SIZE = 16  # OpenAI VPT Paper
N_WORKERS = 24  # Should be more than batch size
DEVICE = "cuda"

LOSS_REPORT_RATE = 100  # also lr scheduler step rate (in batches)
LEARNING_RATE = 0.000181  # OpenAI VPT Paper
WEIGHT_DECAY = 0.039428  # OpenAI VPT Paper
# WEIGHT_DECAY = 0.0
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0
SAVE_EVERY = 1000


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

    # Set up optimizer and learning rate scheduler
    trainable_parameters = policy.parameters()
    optimizer = th.optim.Adam(
        trainable_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # lr_scheduler = ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.1, patience=10, verbose=True
    # )

    # Initialize data loader
    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
    )

    # Create the writer
    writer = SummaryWriter()

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
                    del episode_hidden_states[episode_id]
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

            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0 and batch_i > 0:
            time_since_start = time.time() - start_time
            avg_loss = loss_sum / LOSS_REPORT_RATE
            print(
                f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {avg_loss:.4f} "
            )
            # Log the loss values
            writer.add_scalar("Loss/train", avg_loss, batch_i)

            # # Update learning rate scheduler
            # lr_scheduler.step(avg_loss)

            loss_sum = 0

        # Save the model every SAVE_EVERY batches
        if batch_i % SAVE_EVERY == 0 and batch_i > 0:
            base_name = os.path.basename(out_weights)
            # remove the .weights extension
            base_name = os.path.splitext(base_name)[0]
            intermediate_weights_path = os.path.join(
                os.path.dirname(out_weights),
                f"{base_name}_{batch_i}.weights",
            )
            th.save(policy.state_dict(), intermediate_weights_path)
            print(f"Saved intermediate weights to {intermediate_weights_path}")

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
