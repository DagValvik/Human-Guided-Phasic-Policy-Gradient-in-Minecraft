import copy
import itertools
import os
import pickle
import random
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

EPOCHS = 1  # OpenAI VPT Paper
# BATCH_SIZE = 32  # OpenAI VPT Paper
N_WORKERS = 32  # Should be >= BATCH_SIZE
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

### OpenAI VPT Paper Hyperparameters ###
# LEARNING_RATE = 0.000181  # OpenAI VPT Paper
# WEIGHT_DECAY = 0.039428  # OpenAI VPT Paper

### Experiment Hyperparameters ###
LOSS_REPORT_RATE = 100  # also lr scheduler step rate (in batches)
# LEARNING_RATE = 0.001

# WEIGHT_DECAY = 0  # also try 0.01, 0.001  or 0 (no weight decay)
# KL_LOSS_WEIGHT = 0.5
# MAX_GRAD_NORM = 5.0
# SAVE_EVERY = 10000


def compute_validation_loss(
    data_dir,
    agent,
    policy,
    batch_size,
    max_batches=None,
):
    # Validation loop
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    val_loss_sum = 0

    # Create a DataLoader for validation data
    val_data_loader = DataLoader(
        dataset_dir=os.path.join(data_dir, "val"),
        n_workers=N_WORKERS,
        batch_size=batch_size,
        n_epochs=1,
        max_queue_size=batch_size * 2,
    )

    for batch_i, (
        batch_images,
        batch_actions,
        batch_episode_id,
    ) in enumerate(val_data_loader):
        if max_batches is not None and batch_i >= max_batches:
            break
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

            with th.no_grad():
                (
                    pi_distribution,
                    _,
                    new_agent_state,
                ) = policy.get_output_for_observation(
                    agent_obs, agent_state, dummy_first
                )

            log_prob = policy.get_logprob_of_action(
                pi_distribution, agent_action
            )

            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            loss = (-log_prob) / batch_size
            batch_loss += loss.item()
        val_loss_sum += batch_loss

    # Compute the average validation loss
    avg_val_loss = val_loss_sum / (batch_i + 1)

    return avg_val_loss


def random_search(
    data_dir,
    in_model,
    in_weights,
    env_name,
    hyperparameter_space,
    n_iter,
    max_batches,
):
    # Generate all possible combinations
    all_combinations = list(itertools.product(*hyperparameter_space.values()))

    # Randomly sample n_iter combinations
    random_combinations = random.sample(all_combinations, n_iter)

    # Convert the sampled combinations to dictionaries with corresponding keys
    random_combinations = [
        dict(zip(hyperparameter_space.keys(), combination))
        for combination in random_combinations
    ]

    best_val_loss = float("inf")
    best_hyperparameters = {}
    for i, hyperparameters in enumerate(random_combinations):
        print(f"[INFO] Random search iteration {i + 1}/{n_iter}")
        print(f"Current hyperparameters: {hyperparameters}")

        # Load model parameters and create agents'
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(
            in_model
        )

        agent = create_agent(
            env_name,
            agent_policy_kwargs,
            agent_pi_head_kwargs,
            in_weights,
        )
        original_agent = copy.deepcopy(agent)

        policy = agent.policy
        original_policy = original_agent.policy

        behavior_cloning_train(
            agent,
            policy,
            original_policy,
            data_dir,
            out_weights="we are not saving the weights",
            learning_rate=hyperparameters["learning_rate"],
            weight_decay=hyperparameters["weight_decay"],
            kl_loss_weight=hyperparameters["kl_loss_weight"],
            batch_size=hyperparameters["batch_size"],
            max_grad_norm=hyperparameters["max_grad_norm"],
            max_batches=max_batches,
            save=False,
        )

        val_loss = compute_validation_loss(
            data_dir,
            agent,
            policy,
            batch_size=hyperparameters["batch_size"],
            max_batches=max_batches // 10,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hyperparameters = hyperparameters

        print(f"Current validation loss: {val_loss}")
        print(f"Best validation loss so far: {best_val_loss}")
        print(f"Best hyperparameters so far: {best_hyperparameters}")

    return best_hyperparameters


def behavior_cloning_train(
    agent,
    policy,
    original_policy,
    data_dir,
    out_weights,
    learning_rate,
    weight_decay,
    kl_loss_weight,
    batch_size,
    max_grad_norm,
    max_batches=None,
    save_every=10000,
    save=True,
):
    policy = policy
    original_policy = original_policy

    # Set up optimizer and learning rate scheduler
    trainable_parameters = policy.parameters()
    optimizer = th.optim.Adam(
        trainable_parameters, lr=learning_rate, weight_decay=weight_decay
    )

    # Create the writer
    writer = SummaryWriter()

    # Training loop
    start_time = time.time()
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)
    loss_sum = 0

    for epoch in range(EPOCHS):
        # Initialize data loader
        data_loader = DataLoader(
            dataset_dir=os.path.join(data_dir, "train"),
            n_workers=N_WORKERS,
            batch_size=batch_size,
            n_epochs=1,
            max_queue_size=batch_size * 2,
        )

        for batch_i, (
            batch_images,
            batch_actions,
            batch_episode_id,
        ) in enumerate(data_loader):
            if max_batches is not None and batch_i >= max_batches:
                break
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

                new_agent_state = tree_map(
                    lambda x: x.detach(), new_agent_state
                )
                episode_hidden_states[episode_id] = new_agent_state

                loss = (-log_prob + kl_loss_weight * kl_div) / batch_size
                batch_loss += loss.item()
                loss.backward()

            th.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            loss_sum += batch_loss

            if batch_i % LOSS_REPORT_RATE == 0 and batch_i > 0:
                avg_loss = loss_sum / LOSS_REPORT_RATE
                time_since_start = time.time() - start_time
                print(
                    f"[INFO] Epoch: {epoch} | Time: {time_since_start:.2f} | Batches: {batch_i} | Train loss: {avg_loss:.4f}"  # | Val loss: {avg_loss:.4f} "
                )
                # Log the loss values
                writer.add_scalar("Loss/train", avg_loss, batch_i)

                loss_sum = 0

            # Save the model every SAVE_EVERY batches
            if batch_i % save_every == 0 and batch_i > 0 and save:
                base_name = os.path.basename(out_weights)
                # remove the .weights extension
                base_name = os.path.splitext(base_name)[0]
                folder = os.path.join(os.path.dirname(out_weights), base_name)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                intermediate_weights_path = os.path.join(
                    folder,
                    f"{base_name}_{batch_i}.weights",
                )
                th.save(policy.state_dict(), intermediate_weights_path)
                print(
                    f"[INFO] Saved intermediate weights to {intermediate_weights_path}"
                )

        # Save the finetuned weights
        if save:
            state_dict = policy.state_dict()
            th.save(state_dict, out_weights)
            print("[INFO] Saved weights to", out_weights)


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
