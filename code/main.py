from collections import deque

import gym
import numpy as np
import torch as th
from helpers import exists
from memory import Memory
from openai_vpt.lib.tree_util import tree_map
from ppg import PPG
from reward_predict import RewardPredictorNetwork
from torch.distributions import Categorical
from tqdm import tqdm

VPT_MODEL_PATH = "data/VPT-models/foundation-model-1x.model"
VPT_MODEL_WEIGHTS = "data/VPT-models/foundation-model-1x.weights"
ENV = "MineRLBasaltFindCave-v0"
DEVICE = "cuda"


def main(
    env_name=ENV,
    model=VPT_MODEL_PATH,
    weights=VPT_MODEL_WEIGHTS,
    out_weights="train/rl-fine-tuned-weights.pth",
    reward_predictor=None,
    num_episodes=5000,
    max_timesteps=500,
    actor_hidden_dim=32,
    critic_hidden_dim=256,
    minibatch_size=16,
    lr=0.0005,
    betas=(0.9, 0.999),
    lam=0.95,
    gamma=0.99,
    eps_clip=0.2,
    value_clip=0.4,
    beta_s=0.01,
    update_timesteps=64,
    num_policy_updates_per_aux=32,
    epochs=1,
    epochs_aux=6,
    render=False,
    render_every_eps=250,
    save_every=1000,
    load=True,
):
    """
    :param env_name: OpenAI gym environment name
    :param num_episodes: number of episodes to train
    :param max_timesteps: max timesteps per episode
    :param actor_hidden_dim: actor network hidden layer size
    :param critic_hidden_dim: critic network hidden layer size
    :param minibatch_size: minibatch size for training
    :param lr: learning rate for optimizers
    :param betas: betas for Adam Optimizer
    :param lam: GAE lambda (exponential discount)
    :param gamma: GAE gamma (future discount)
    :param eps_clip: PPO policy loss clip coefficient
    :param value_clip: value loss clip coefficient
    :param beta_s: entropy loss coefficient
    :param update_timesteps: number of timesteps to run before training
    :param epochs: policy phase epochs
    :param epochs_aux: auxiliary phase epochs
    :param render: toggle render environment
    :param render_every_eps: if render, how often to render
    :param save_every: how often to save networks
    :load: toggle load a previously trained network
    """
    device = th.device(DEVICE if th.cuda.is_available() else "cpu")
    env = gym.make(env_name)

    memories = deque([])
    aux_memories = deque([])

    ppg = PPG(
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
    )

    time = 0
    updated = False
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc="episodes"):
        render_eps = render and eps % render_every_eps == 0
        obs = env.reset()
        dummy_first = th.from_numpy(np.array((False,))).to(device)
        dummy_first = dummy_first.unsqueeze(1)
        hidden_state = None
        for timestep in range(max_timesteps):
            time += 1

            if updated and render_eps:
                env.render()

            if hidden_state is None:
                # Get initial state for policy
                hidden_state = ppg.agent.policy.initial_state(1)

            agent_obs = ppg.agent._env_obs_to_agent(obs)

            # Basically just adds a dimension to both camera and button tensors
            agent_obs = tree_map(lambda x: x.unsqueeze(1), agent_obs)

            with th.no_grad():
                (pi_h, v_h), next_hidden_state = ppg.agent.policy.net(
                    agent_obs, hidden_state, context={"first": dummy_first}
                )

                pi_distribution = ppg.agent.policy.pi_head(pi_h)
                v_prediction = ppg.agent.policy.value_head(v_h)

            action = ppg.agent.policy.pi_head.sample(
                pi_distribution, deterministic=False
            )

            log_prob = ppg.agent.policy.get_logprob_of_action(
                pi_distribution, action
            )

            # Take the action in the environment
            minerl_action = ppg.agent._agent_action_to_env(action)
            minerl_action["ESC"] = 0
            obs, _, done, _ = env.step(minerl_action)

            # Preproccess the next observation
            new_agent_obs = ppg.agent._env_obs_to_agent(obs)
            # convert to NCHW
            input_tensor = new_agent_obs["img"].permute(0, 3, 1, 2)
            # Predict the reward for the observation
            reward = reward_predictor.reward(input_tensor)

            memory = Memory(
                agent_obs,
                0,
                0,
                0,
                action,
                log_prob,
                reward,
                done,
                v_prediction,
            )
            memories.append(memory)

            if time % update_timesteps == 0:
                ppg.learn(memories, aux_memories)
                num_policy_updates += 1

                # Get human preferences

                # Train reward predictor

                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    ppg.learn_aux(aux_memories)
                    aux_memories.clear()

                updated = True

            if done:
                if render_eps:
                    updated = False
                break

        if render_eps:
            env.close()

        if eps % save_every == 0:
            ppg.save()


if __name__ == "__main__":
    reward_predictor = RewardPredictorNetwork()
    main(reward_predictor=reward_predictor)
