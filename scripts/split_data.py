import os
import random
import shutil
from argparse import ArgumentParser
from collections import defaultdict


def split_data_into_train_val(data_folder, val_percentage=0.2):
    env_folders = [
        os.path.join(data_folder, d)
        for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
        and not d.startswith("VPT-models")
    ]

    for env_folder in env_folders:
        train_folder = os.path.join(env_folder, "train")
        val_folder = os.path.join(env_folder, "val")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        all_files = [
            f
            for f in os.listdir(env_folder)
            if os.path.isfile(os.path.join(env_folder, f))
        ]

        # Group files by episode ID
        episode_files = defaultdict(list)
        for file in all_files:
            episode_id = "-".join(
                file.split("-")[:-2]
            )  # Remove timestamp and extension
            episode_files[episode_id].append(file)

        val_episodes_count = int(len(episode_files) * val_percentage)
        val_episodes = random.sample(
            list(episode_files.keys()), val_episodes_count
        )
        train_episodes = [
            ep for ep in episode_files.keys() if ep not in val_episodes
        ]

        def move_files(files, src_folder, dst_folder):
            for file in files:
                src = os.path.join(src_folder, file)
                dst = os.path.join(dst_folder, file)
                shutil.move(src, dst)

        for ep in train_episodes:
            move_files(episode_files[ep], env_folder, train_folder)

        for ep in val_episodes:
            move_files(episode_files[ep], env_folder, val_folder)

        print(
            f"Moved {len(train_episodes)} episodes to the train folder and {val_episodes_count} episodes to the validation folder in {env_folder}."
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing recordings to be split",
    )
    parser.add_argument(
        "--val-percentage",
        type=float,
        default=0.2,
        help="Percentage of the data to be used for validation",
    )
    args = parser.parse_args()
    split_data_into_train_val(args.data_dir, args.val_percentage)
