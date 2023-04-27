from collections import deque

import gym
import numpy as np
import torch as th
from helpers import exists
from memory import Memory
from openai_vpt.lib.tree_util import tree_map
from ppg import PPG
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

    ppg = PPG(
        env_name,
        model,
        weights,
        out_weights,
        device,
        reward_predictor,
        epochs=1,
        epochs_aux=6,
        minibatch_size=32,
        lr=3e-4,
        betas=(0.9, 0.999),
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        value_clip=0.2,
    )

    ppg.train(
        num_episodes=500000,
        max_timesteps=500,
        render=False,
        update_timesteps=500,
        num_policy_updates_per_aux=32,
        render_every_eps=250,
        save_every=1000,
    )


if __name__ == "__main__":
    main()
