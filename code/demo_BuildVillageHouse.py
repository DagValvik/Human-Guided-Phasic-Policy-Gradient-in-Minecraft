from config import EVAL_EPISODES, EVAL_MAX_STEPS
from run_agent import main as run_agent_main


def main():
    run_agent_main(
        model="data/VPT-models/foundation-model-1x.model",
        weights="train\MineRLBasaltBuildVillageHouse\MineRLBasaltBuildVillageHouse_scheduler_frozen.weights",
        env="MineRLBasaltBuildVillageHouse-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS,
        show=True,
    )


if __name__ == "__main__":
    main()
