# Train one model for each task
import copy

from behavioural_cloning import behavior_cloning_train
from helpers import create_agent, load_model_parameters

### Hyperparameter found by random search 5k batches ###
LEARNING_RATE = 1e-6  # or 1e-5 with 1e-3 weight was very close on 5k batches
WEIGHT_DECAY = 0.01
KL_LOSS_WEIGHT = 0.5
BATCH_SIZE = 32
MAX_GRAD_NORM = 5.0
########################################################
IN_MODEL = "data/VPT-models/foundation-model-1x.model"
IN_WEIGHTS = "data/VPT-models/foundation-model-1x.weights"


def main():
    # print("===Training FindCave model===")
    # behavior_cloning_train(
    #     data_dir="data/MineRLBasaltFindCave-v0",
    #     in_model="data/VPT-models/foundation-model-1x.model",
    #     in_weights="data/VPT-models/foundation-model-1x.weights",
    #     out_weights="train/MineRLBasaltFindCave.weights",
    #     env_name="MineRLBasaltFindCave-v0",
    # )

    # # print("===Training MakeWaterfall model===")
    # # behavioural_cloning_train(
    # #     data_dir="data/MineRLBasaltMakeWaterfall-v0",
    # #     in_model="data/VPT-models/foundation-model-1x.model",
    # #     in_weights="data/VPT-models/foundation-model-1x.weights",
    # #     out_weights="train/MineRLBasaltMakeWaterfall.weights"
    # # )

    # print("===Training CreateVillageAnimalPen model===")
    # behavior_cloning_train(
    #     data_dir="data/MineRLBasaltCreateVillageAnimalPen-v0",
    #     in_model="data/VPT-models/foundation-model-1x.model",
    #     in_weights="data/VPT-models/foundation-model-1x.weights",
    #     out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights",
    #     env_name="MineRLBasaltCreateVillageAnimalPen-v0",
    # )

    print("===Training BuildVillageHouse model===")
    # Load model parameters and create agents'
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(IN_MODEL)

    agent = create_agent(
        "MineRLBasaltBuildVillageHouse-v0",
        agent_policy_kwargs,
        agent_pi_head_kwargs,
        IN_WEIGHTS,
    )
    original_agent = copy.deepcopy(agent)

    policy = agent.policy
    original_policy = original_agent.policy

    behavior_cloning_train(
        agent=agent,
        policy=policy,
        original_policy=original_policy,
        data_dir="data/MineRLBasaltBuildVillageHouse-v0",
        out_weights="train/MineRLBasaltBuildVillageHouse.weights",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        kl_loss_weight=KL_LOSS_WEIGHT,
        batch_size=BATCH_SIZE,
        max_grad_norm=MAX_GRAD_NORM,
        save=True,
        save_every=10000,
    )


if __name__ == "__main__":
    main()
