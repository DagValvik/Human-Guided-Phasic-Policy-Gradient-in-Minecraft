import copy
from collections import deque
from typing import Deque

import gym
import numpy as np
import torch as th
from helpers import (
    calculate_gae,
    clipped_value_loss,
    create_agent,
    create_fixed_length_segments,
    load_model_parameters,
    normalize,
    update_network_,
)
from memory import AuxMemory, Episode, Memory, create_dataloader
from openai_vpt.agent import resize_image

# from memory import AuxMemory, Memory
from openai_vpt.lib.tree_util import tree_map
from preference_interface import PreferenceInterface
from reward_predict import RewardPredictorNetwork
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
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
        preference_interface,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        max_grad_norm,
        betas,
        beta_s,
        gamma,
        lam,
        clip,
        value_clip,
        kl_beta,
        beta_clone,
    ):
        self.epochs = epochs
        self.epochs_aux = epochs_aux
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.betas = betas
        self.beta_s = beta_s
        self.kl_beta = kl_beta
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.value_clip = value_clip
        self.env_name = env_name
        self.reward_predictor = reward_predictor
        self.preference_interface = preference_interface
        self.device = device
        self.out_weights = out_weights
        self.beta_clone = beta_clone

        # Load model parameters and create agents
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(model)

        # Create the main agent and auxiliary agent here
        self.agent = create_agent(
            env_name, agent_policy_kwargs, agent_pi_head_kwargs, weights
        )

        self.critic = copy.deepcopy(self.agent)

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

        # Create original agent for kl divergence (not sure if this is necessary)
        self.original_agent = copy.deepcopy(self.agent)

        # TODO: Implement scheduler

        # Tensorboard logging
        self.writer = SummaryWriter()
        self.wake_updates = 0
        self.sleep_updates = 0

    def save(self):
        """
        Saves the model weights to out_weights path
        """
        th.save(self.agent.policy.state_dict(), self.out_weights)
        print(f"Saved model weights to {self.out_weights}")

    def collect_episodes(self, memories, num_episodes_to_collect, env):
        inital_hidden_states = {"agent": [], "critic": []}
        for _ in range(num_episodes_to_collect):
            episode = []
            obs = env.reset()
            dummy_first = th.from_numpy(np.array((False,))).to(self.device)
            # Reset the hidden state for the policy
            # initial_critic_hidden_state = self.critic.policy.initial_state(1)
            # initial_agent_hidden_state = self.agent.policy.initial_state(1)
            # Maybe we don't need to worry about hidden state reconstruction
            # agent_hidden_state = initial_agent_hidden_state
            # critic_hidden_state = initial_critic_hidden_state
            # inital_hidden_states["agent"].append(agent_hidden_state)
            # inital_hidden_states["critic"].append(critic_hidden_state)

            # Dont reconstruct hidden state
            critic_hidden_state = self.critic.policy.initial_state(1)
            agent_hidden_state = self.agent.policy.initial_state(1)

            done = False
            while not done:
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
                next_obs, _, next_done, _ = env.step(minerl_action)

                # Preprocess the next observation for the reward predictor
                image = resize_image(obs["pov"], (256, 256))[
                    None
                ]  # 128x128 is to small for comparison for humans, 256x256 is better
                input_tensor = (
                    th.from_numpy(image).to(self.device).permute(0, 3, 1, 2)
                )
                # Predict the reward for the observation
                reward = self.reward_predictor.reward(input_tensor)

                if next_done:
                    # Compute the value for the next observation
                    next_value, _ = self.critic.policy.v(
                        self.agent._env_obs_to_agent(next_obs),
                        dummy_first,
                        critic_hidden_state,
                    )

                    # Add the next_value to the last memory of the episode
                    memory = Memory(
                        agent_obs,
                        obs["pov"],
                        action,
                        result["log_prob"],
                        reward,
                        next_done,
                        value,
                        next_value,
                    )

                    episode.append(memory)
                    break

                memory = Memory(
                    agent_obs,
                    obs["pov"],
                    action,
                    result["log_prob"],
                    reward,
                    next_done,
                    value,
                    None,
                )

                episode.append(memory)

                # Update the hidden state for the next timestep
                agent_hidden_state = new_agent_state
                critic_hidden_state = new_critic_state
                obs = next_obs
                done = next_done
            memories.append(episode)
        print(
            f"[INFO] Finished collecting episodes | memories: {len(memories)}"
        )
        return inital_hidden_states

    def pretrain_reward_predictor(
        self, n_iterations, segment_length, n_rollouts, n_pairs, n_epochs
    ):
        env = gym.make(self.env_name)
        memories: Deque[Episode] = deque([])
        for eps in tqdm(range(n_iterations), desc="gather initial preferences"):
            _ = self.collect_episodes(memories, n_rollouts, env)
            while memories:
                episode = memories.popleft()
                segments = create_fixed_length_segments(
                    [mem.obs for mem in episode],
                    [mem.reward for mem in episode],
                    segment_length,
                )
                self.preference_interface.add_segments(segments)
            self.preference_interface.get_preferences(n_pairs)

        self.reward_predictor.train(n_epochs)  # Train for 200 epochs

        self.reward_predictor.save(
            f"train/RewardPredictor_{n_iterations*n_pairs}.pt"
        )

    def train(
        self,
        n_iterations,
        save_every,
        n_wake_cycle_per_preference_update,
        n_wake_cycle_per_aux_update,
        segment_length,
        n_rollouts,
        n_pairs=10,
    ):
        env = gym.make(self.env_name)

        memories: Deque[Episode] = deque([])
        aux_memories = deque([])
        n_wake_phases = 0

        for eps in tqdm(range(n_iterations), desc="iterations"):
            initial_hidden_states = self.collect_episodes(
                memories, n_rollouts, env
            )

            # Wake phase (policy/ppo phase)
            episode_id = 0
            while memories:
                # Get the episode
                episode = memories.popleft()

                # Split the episode into fixed-length segments
                segments = create_fixed_length_segments(
                    [memory.obs for memory in episode],
                    [memory.reward for memory in episode],
                    segment_length,
                )

                # Add the segments to the preference interface
                self.preference_interface.add_segments(segments)

                self.learn(
                    episode,
                    aux_memories,
                    initial_hidden_states,
                    episode_id,
                )

                episode_id += 1
            n_wake_phases += 1

            # Clear the memories just to be sure
            memories.clear()

            # Preference phase
            if n_wake_phases % n_wake_cycle_per_preference_update == 0:
                self.preference_interface.get_preferences(n_pairs=n_pairs)
                # update reward predictor with new preferences
                self.reward_predictor.train(n_epochs=1)

            # Sleep phase (auxiliary phase)
            if n_wake_phases % n_wake_cycle_per_aux_update == 0:
                self.learn_aux(aux_memories, initial_hidden_states)
                aux_memories.clear()

    def learn(self, episode, aux_memories, initial_hidden_states, episode_id):
        # prepare the memories
        agent_observations = [memory.agent_obs for memory in episode]
        actions = [memory.action for memory in episode]
        old_log_probs = [memory.action_log_prob for memory in episode]
        rewards = [memory.reward for memory in episode]
        masks = [1 - float(memory.done) for memory in episode]
        values = [memory.value for memory in episode]

        # Use the next_value from the last memory in the episode
        next_value = episode[-1].next_value
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
            agent_hidden_state = self.agent.policy.initial_state(
                self.minibatch_size
            )
            critic_hidden_state = self.critic.policy.initial_state(
                self.minibatch_size
            )
            orig_hidden_state = self.original_agent.policy.initial_state(
                self.minibatch_size
            )

            for (
                agent_obs_batch,
                actions_batch,
                old_log_probs_batch,
                rewards_batch,
                old_values_batch,
            ) in tqdm(dl, desc="Policy phase training"):
                agent_obs_batch = tree_map(
                    lambda x: x.squeeze(1), agent_obs_batch
                )

                dummy_firsts = th.from_numpy(
                    np.full(len(agent_obs_batch["img"]), False)
                ).to(device)

                # Run the agent to get the policy distribution
                (
                    pi_distribution,
                    _,
                    next_agent_hidden_state,
                ) = self.agent.policy.get_output_for_observation(
                    agent_obs_batch, agent_hidden_state, dummy_firsts
                )

                # Run the critic to get the value prediction
                (
                    _,
                    values,
                    next_critic_hidden_state,
                ) = self.critic.policy.get_output_for_observation(
                    agent_obs_batch,
                    critic_hidden_state,
                    dummy_firsts,
                )

                # Calculate the log prob1
                log_probs = self.agent.policy.get_logprob_of_action(
                    pi_distribution, actions_batch
                )

                with th.no_grad():
                    (
                        original_pi_distribution,
                        _,
                        next_orig_hidden_state,
                    ) = self.original_agent.policy.get_output_for_observation(
                        agent_obs_batch, orig_hidden_state, dummy_firsts
                    )

                # Calculate KL divergence
                kl_divs = self.agent.policy.pi_head.kl_divergence(
                    pi_distribution, original_pi_distribution
                )

                # Update hidden states for the next iteration
                agent_hidden_state = tree_map(
                    lambda x: x.detach(), next_agent_hidden_state
                )
                critic_hidden_state = tree_map(
                    lambda x: x.detach(), next_critic_hidden_state
                )
                orig_hidden_state = tree_map(
                    lambda x: x.detach(), next_orig_hidden_state
                )

                # Calculate entropy
                entropy = self.agent.policy.pi_head.entropy(pi_distribution).to(
                    self.device
                )

                ratios = (log_probs - old_log_probs_batch).exp().to(self.device)
                advantages = normalize(
                    rewards_batch - old_values_batch.detach()
                ).to(self.device)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.clip, 1 + self.clip) * advantages

                policy_loss = (
                    -th.min(surr1, surr2)
                    - self.beta_s * entropy
                    - self.kl_beta * kl_divs
                )

                update_network_(
                    policy_loss,
                    self.agent_optim,
                    self.agent.policy,
                    self.max_grad_norm,
                )

                value_loss = clipped_value_loss(
                    values,
                    rewards_batch,
                    old_values_batch,
                    self.value_clip,
                )

                update_network_(
                    value_loss,
                    self.critic_optim,
                    self.critic.policy,
                    self.max_grad_norm,
                )

                # Tensorboard logging
                self.writer.add_scalar(
                    "Wake Loss/Policy",
                    policy_loss.mean().item(),
                    self.wake_updates,
                )
                self.writer.add_scalar(
                    "Wake Loss/Value",
                    value_loss.mean().item(),
                    self.wake_updates,
                )

                self.wake_updates += 1

    def learn_aux(self, aux_memories, initial_hidden_states):
        for epoch in range(self.epochs_aux):
            for episode_id, aux in enumerate(aux_memories):
                # Create the dataloader for the episode
                aux_dl = create_dataloader(
                    [aux.agent_obs, aux.rewards, aux.old_values],
                    batch_size=self.minibatch_size,
                )

                agent_hidden_state = self.agent.policy.initial_state(
                    self.minibatch_size
                )
                critic_hidden_state = self.critic.policy.initial_state(
                    self.minibatch_size
                )
                orig_hidden_state = self.original_agent.policy.initial_state(
                    self.minibatch_size
                )

                for agent_obs, rewards, old_values in tqdm(
                    aux_dl, desc=f"Auxiliary epoch {epoch}"
                ):
                    # agent_hidden_states = initial_hidden_states["agent"][
                    #     episode_id
                    # ]
                    # critic_hidden_state = initial_hidden_states["critic"][
                    #     episode_id
                    # ]
                    # policy_values = []
                    # values = []
                    # kl_divs = []

                    # # Unstack the agent_obs_batch and actions_batch
                    # obs_tensors = th.unbind(obs["img"])

                    # # Iterate over the unstacked tensors directly
                    # for obs_tensor in obs_tensors:
                    #     obs["img"] = obs_tensor
                    #     # Run the agent to get the policy distribution
                    #     (
                    #         pi_distribution,
                    #         policy_value,
                    #         next_agent_hidden_state,
                    #     ) = self.agent.policy.get_output_for_observation(
                    #         obs, agent_hidden_states, dummy_first
                    #     )

                    #     with th.no_grad():
                    #         (
                    #             original_pi_distribution,
                    #             _,
                    #             _,
                    #         ) = self.original_agent.policy.get_output_for_observation(
                    #             obs, agent_hidden_states, dummy_first
                    #         )

                    #     # run the critic to get the value prediction
                    #     (
                    #         _,
                    #         v_prediction,
                    #         next_critic_hidden_state,
                    #     ) = self.critic.policy.get_output_for_observation(
                    #         obs, critic_hidden_state, dummy_first
                    #     )
                    #     # Calculate KL divergence
                    #     kl_div = self.agent.policy.pi_head.kl_divergence(
                    #         pi_distribution, original_pi_distribution
                    #     )

                    #     policy_values.append(policy_value)
                    #     values.append(v_prediction)
                    #     kl_divs.append(kl_div)

                    #     # Update hidden states for the next iteration
                    #     next_agent_hidden_state = tree_map(
                    #         lambda x: x.detach(), next_agent_hidden_state
                    #     )
                    #     next_critic_hidden_state = tree_map(
                    #         lambda x: x.detach(), next_critic_hidden_state
                    #     )
                    #     agent_hidden_states = next_agent_hidden_state
                    #     critic_hidden_state = next_critic_hidden_state

                    # # Stack the needed tensors
                    # policy_values = th.stack(policy_values).to(self.device)
                    # values = th.stack(values).to(self.device)
                    # kl_divs = th.stack(kl_divs).to(self.device)

                    agent_obs = tree_map(lambda x: x.squeeze(1), agent_obs)

                    dummy_firsts = th.from_numpy(
                        np.full(len(agent_obs["img"]), False)
                    ).to(device)

                    # Run the agent to get the policy distribution
                    (
                        pi_distribution,
                        policy_values,
                        next_agent_hidden_state,
                    ) = self.agent.policy.get_output_for_observation(
                        agent_obs, agent_hidden_state, dummy_firsts
                    )

                    # Run the critic to get the value prediction
                    (
                        _,
                        values,
                        next_critic_hidden_state,
                    ) = self.critic.policy.get_output_for_observation(
                        agent_obs,
                        critic_hidden_state,
                        dummy_firsts,
                    )

                    with th.no_grad():
                        (
                            original_pi_distribution,
                            _,
                            next_orig_hidden_state,
                        ) = self.original_agent.policy.get_output_for_observation(
                            agent_obs, orig_hidden_state, dummy_firsts
                        )

                    # Calculate KL divergence
                    kl_div = self.agent.policy.pi_head.kl_divergence(
                        pi_distribution, original_pi_distribution
                    )

                    # Update hidden states for the next iteration
                    agent_hidden_state = tree_map(
                        lambda x: x.detach(), next_agent_hidden_state
                    )
                    critic_hidden_state = tree_map(
                        lambda x: x.detach(), next_critic_hidden_state
                    )
                    orig_hidden_state = tree_map(
                        lambda x: x.detach(), next_orig_hidden_state
                    )

                    aux_loss = clipped_value_loss(
                        policy_values,
                        rewards,
                        old_values,
                        self.value_clip,
                    )

                    # policy loss is aux loss + kl_div_loss
                    policy_loss = aux_loss - self.beta_clone * kl_div

                    update_network_(
                        policy_loss,
                        self.agent_optim,
                        self.agent.policy,
                        self.max_grad_norm,
                    )

                    value_loss = clipped_value_loss(
                        values,
                        rewards,
                        old_values,
                        self.value_clip,
                    )

                    update_network_(
                        value_loss,
                        self.critic_optim,
                        self.critic.policy,
                        self.max_grad_norm,
                    )

                    # Tensorboard logging
                    self.writer.add_scalar(
                        "Sleep Loss/Value (Joint)",
                        policy_loss.mean().item(),
                        self.sleep_updates,
                    )
                    self.writer.add_scalar(
                        "Sleep Loss/Value (Critic)",
                        value_loss.mean().item(),
                        self.sleep_updates,
                    )

                    self.sleep_updates += 1


if __name__ == "__main__":
    VPT_MODEL_PATH = "data/VPT-models/foundation-model-1x.model"
    VPT_MODEL_WEIGHTS = "train/MineRLBasaltBuildVillageHouse/MineRLBasaltBuildVillageHouse_100000.weights"
    REWARD_PREDICTOR_PATH = (
        "train/MineRLBasaltBuildVillageHouse/RewardPredictor_1000.pt"
    )
    ENV = "MineRLBasaltBuildVillageHouse-v0"
    TASK = "Build a house in same style as village"
    DEVICE = "cuda"

    device = th.device(DEVICE if th.cuda.is_available() else "cpu")

    preference_queue = deque([])
    sequence_queue = []

    reward_predictor = RewardPredictorNetwork(pref_queue=preference_queue)
    pref_interface = PreferenceInterface(sequence_queue, preference_queue, TASK)

    ppg = PPG(
        ENV,
        VPT_MODEL_PATH,
        VPT_MODEL_WEIGHTS,
        f"train/{ENV}-ppg.weights",
        device,
        reward_predictor,
        pref_interface,
        epochs=1,
        epochs_aux=6,
        minibatch_size=32,
        max_grad_norm=5,
        lr=1e-5,
        betas=(0.9, 0.999),
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        value_clip=0.2,
        beta_s=0.01,
        kl_beta=0.5,
        beta_clone=1.0,
    )

    # Comment out if you want to train the reward predictor
    # ppg.pretrain_reward_predictor(
    #     # These parameters control how much variation there is in the
    #     # preferences that are generated for the reward predictor.
    #     # e.g 10 iterations * 50 pairs = 500 comparisons, with n_rollouts controlling the diversity of the pairs.
    #     n_iterations=1,
    #     segment_length=50,
    #     n_rollouts=1,
    #     n_pairs=10,
    #     # Number of epochs to train the reward predictor for
    #     n_epochs=200,
    # )

    ppg.reward_predictor.load(REWARD_PREDICTOR_PATH)

    ppg.train(
        n_iterations=100,
        n_wake_cycle_per_preference_update=1,
        n_wake_cycle_per_aux_update=1,
        segment_length=50,  # 2-3 seconds
        save_every=1000,
        n_rollouts=1,
        n_pairs=10,
    )
