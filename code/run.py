import pickle
from argparse import ArgumentParser
from collections import deque
from copy import deepcopy
from queue import Queue
from threading import Thread

import gym
import minerl
import numpy as np
import torch as th
from behavioural_cloning import load_model_parameters
from openai_vpt.agent import MineRLAgent
from ppo import compute_gae, ppo_iter, ppo_update, value_update
from preference_interface import PreferenceInterface
from reward_predict import RewardPredictorNetwork
from segment import Segment

VPT_MODEL_PATH = "data/VPT-models/foundation-model-1x.model"
VPT_MODEL_WEIGHTS = "data/VPT-models/foundation-model-1x.weights"
ENV = "MineRLBasaltFindCave-v0"
DEVICE = "cuda"

# Constants
ACTOR_STEPS = 1000  # Number of steps to collect data for the actor phase
CRITIC_STEPS = 1000  # Number of steps to collect data for the critic phase
PPO_EPOCHS = 4  # Number of epochs to train PPO
PPO_CLIP = 0.2  # Clipping parameter for PPO
VALUE_COEF = 0.5  # Coefficient for value loss
ENTROPY_COEF = 0.01  # Coefficient for entropy bonus
LR = 3e-4  # Learning rate
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # GAE lambda
MAX_GRAD_NORM = 0.5  # Max gradient norm for clipping
BATCH_SIZE = 64  # Batch size for PPO

# Make queues for the segments
pref_queue = Queue()
seg_queue = Queue()

# Create the network
r_predict = RewardPredictorNetwork(pref_queue=pref_queue)


# Wrap the train function call
def train_reward_predictor():
    r_predict.train()


def optimize_Ljoint(policy, policy_old, buffer_B, optimizer, clip_epsilon=0.2):
    states, returns, actions = zip(*buffer_B)
    states = th.stack(states)
    returns = th.tensor(returns, dtype=th.float32)
    actions = th.stack(actions)

    pi_old, _, _ = policy_old.get_output_for_observation(states)
    pi, values, _ = policy.get_output_for_observation(states)

    log_probs_old = pi_old.log_prob(actions)
    log_probs = pi.log_prob(actions)

    ratio = (log_probs - log_probs_old).exp()
    surr1 = ratio * advantages
    surr2 = th.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -th.min(surr1, surr2).mean()

    value_loss = 0.5 * (returns - values).pow(2).mean()

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_Lvalue(policy, buffer_B, optimizer):
    states, returns = zip(*buffer_B)
    states = th.stack(states)
    returns = th.tensor(returns, dtype=th.float32)

    _, values, _ = policy.get_output_for_observation(states)
    value_loss = 0.5 * (returns - values).pow(2).mean()

    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()


# Create the thread and start it
train_reward_predictor_thread = Thread(target=train_reward_predictor)
train_reward_predictor_thread.start()

# Create the preference interface
pref_interface = PreferenceInterface(seg_queue, pref_queue)
pref_interface_thread = Thread(target=pref_interface.run)
pref_interface_thread.start()

# load the VPT model for fine tuning
agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(
    VPT_MODEL_PATH
)

# To create model with the right environment.
# All basalt environments have the same settings, so any of them works here
env = gym.make(ENV)
agent = MineRLAgent(
    env,
    device=DEVICE,
    policy_kwargs=agent_policy_kwargs,
    pi_head_kwargs=agent_pi_head_kwargs,
)
agent.load_weights(VPT_MODEL_WEIGHTS)
policy = agent.policy
show = True

dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

# Create the optimizer
optimizer = th.optim.Adam(policy.parameters(), lr=LR)
N_PHASES = 100
N_POLICY = 100  # Number of policy phase iterations
E_POLICY = 10  # Number of policy phase epochs
E_VALUE = 10  # Number of value phase epochs
E_AUX = 10  # Number of auxiliary epochs
N_ROLLOUTS = 10  # Number of rollouts per policy phase iteration

for phase in range(N_PHASES):
    # Buffers
    buffer_B = []
    # Policy Phase
    for iteration in range(N_POLICY):
        all_episode_states = []
        all_episode_actions = []
        all_episode_log_probs = []
        all_episode_rewards = []
        all_episode_masks = []
        all_episode_advantages = []
        all_episode_returns = []

        for _ in range(N_ROLLOUTS):
            obs = env.reset()
            done = False
            # episode_segment = Segment() dont want double up saved states
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            episode_rewards = []
            episode_masks = []

            while not done:
                if show:
                    env.render()
                action = agent.get_action(obs)
                agent_action = agent._env_action_to_agent(
                    action, to_torch=True, check_if_null=True
                )

                agent_obs = agent._env_obs_to_agent(obs)

                agent_state = policy.initial_state(1)

                (
                    pi_distribution,
                    value_prediction,
                    new_agent_state,
                ) = policy.get_output_for_observation(
                    agent_obs, agent_state, dummy_first
                )

                log_prob = policy.get_logprob_of_action(
                    pi_distribution, agent_action
                )

                # Take the action in the environment
                action["ESC"] = 0
                obs, _, done, _ = env.step(action)

                # Predict the reward for the observation
                input_tensor = agent_obs["img"].permute(
                    0, 3, 1, 2
                )  # convert to NCHW
                reward = r_predict.reward(input_tensor)
                # episode_segment.add_frame(input_tensor, reward)
                # Add the segment to the queue for human preferences
                # seg_queue.put(episode_segment)

                episode_states.append(agent_obs)
                episode_actions.append(agent_action)
                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)
                episode_masks.append(1 - int(done))

            # Compute returns and advantages
            returns, advantages = compute_gae(
                episode_rewards, episode_masks, policy, agent_obs, GAMMA, LAMBDA
            )

            all_episode_states.extend(episode_states)
            all_episode_actions.extend(episode_actions)
            all_episode_log_probs.extend(episode_log_probs)
            all_episode_rewards.extend(episode_rewards)
            all_episode_masks.extend(episode_masks)
            all_episode_advantages.extend(advantages)
            all_episode_returns.extend(returns)

        # Policy Phase: Optimize Lclip + entropy bonus
        for epoch in range(E_POLICY):
            ppo_update(
                all_episode_states,
                all_episode_actions,
                all_episode_log_probs,
                all_episode_rewards,
                all_episode_advantages,
                policy,
                optimizer,
            )

        # Policy Phase: Optimize Lvalue
        for epoch in range(E_VALUE):
            value_update(
                all_episode_states,
                all_episode_returns,
                policy,
                optimizer,
                GAMMA,
            )

        # Add all (state, value_target) pairs to buffer_B
        buffer_B.extend(
            zip(all_episode_states, all_episode_returns, all_episode_actions)
        )

    # Store current policy
    policy_old = deepcopy(policy)

    # Auxiliary Phase
    for epoch in range(E_AUX):
        # Optimize Ljoint wrt policy, on all data in buffer_B
        optimize_Ljoint(policy, policy_old, buffer_B, optimizer)

        # Optimize Lvalue wrt value network, on all data in buffer_B
        optimize_Lvalue(policy, buffer_B, optimizer)

    # Query human for preferences if the collected segments reach the threshold
    pref_interface.query_human()
    # Train the reward predictor network if the collected preferences reach the threshold
