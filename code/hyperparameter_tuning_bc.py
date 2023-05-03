import random

import numpy as np
from behavioural_cloning import random_search

# Define hyperparameter_space for random search
hyperparameter_space = {
    "learning_rate": [2e-5, 2e-6, 1e-5, 1e-6],
    "weight_decay": [0, 1e-2, 1e-3],
    "kl_loss_weight": [0.5, 1.0, 2.0],
    "batch_size": [16, 32],
    "max_grad_norm": [1.0, 2.0, 5.0, 10.0],
}

if __name__ == "__main__":
    # Set the paths and other required variables
    data_dir = "data/MineRLBasaltBuildVillageHouse-v0"
    in_model = "data/VPT-models/foundation-model-1x.model"
    in_weights = "data/VPT-models/foundation-model-1x.weights"
    out_weights = "train/MineRLBasaltBuildVillageHouse.weights"
    env_name = "MineRLBasaltBuildVillageHouse-v0"

    # Run the random search
    best_hyperparameters = random_search(
        data_dir,
        in_model,
        in_weights,
        env_name,
        hyperparameter_space,
        n_iter=15,
        max_batches=5000,
    )

    print("Best hyperparameters found:", best_hyperparameters)
