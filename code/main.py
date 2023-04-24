from collections import deque

import gym
import numpy as np
import torch as th
from helpers import exists
from memory import Memory
from openai_vpt.lib.tree_util import tree_map
from ppg import PPG, AuxParams, PPOParams
from reward_predict import RewardPredictorNetwork
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
):
    """
    :param env_name: MineRL gym environment name
    :param model: Path to VPT model
    :param weights: Path to VPT model weights
    :param out_weights: Path to save fine-tuned weights
    """
    device = th.device(DEVICE if th.cuda.is_available() else "cpu")

    reward_predictor = RewardPredictorNetwork()

    ppo_params = PPOParams(
        epochs=4,
        minibatch_size=8,
        lr=3e-4,
        betas=(0.9, 0.999),
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        value_clip=0.2,
    )

    aux_params = AuxParams(
        epochs=5,
        num_policy_updates_per_aux=2,
    )

    ppg = PPG(
        env_name,
        model,
        weights,
        out_weights,
        ppo_params,
        aux_params,
        device,
        reward_predictor,
    )

    ppg.train(
        num_episodes=100,
        max_timesteps=5000,
        update_timesteps=80,
        num_policy_updates_per_aux=5,
    )


if __name__ == "__main__":
    main()
