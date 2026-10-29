import glob
import os

import numpy as np

SHOT_TYPES = [
    "intermediate",
    "first_good",
    "first_short",
    "first_long",
    "final_win",
    "final_lose",
]
DIRECTIONS_OF_PLAY = ["left_to_right", "right_to_left"]
# the split code stored per trajectory in the packed files
TRAIN, DEV, TEST, DROPPED = 0, 1, 2, -1
SPLIT_NAMES = {TRAIN: "train", DEV: "dev", TEST: "test"}


def packed_path(shot_type, direction, cache_dir="cache"):
    return os.path.join(cache_dir, f"synthetic_{shot_type}_{direction}.npz")


def load_packed_folders(cache_dir="cache"):
    # one dict of arrays per shot type and direction, as scripts/prepare_data.py wrote them
    folders = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "synthetic_*.npz"))):
        packed_file = np.load(path)
        folder = {
            array_name: packed_file[array_name] for array_name in packed_file.files
        }
        folders[(str(folder["shot_type"]), str(folder["direction_of_play"]))] = folder
    if not folders:
        raise SystemExit(
            f"No {cache_dir}/synthetic_*.npz: run scripts/prepare_data.py first."
        )
    return folders


def get_one_trajectory(folder, trajectory_number):
    first_sample, one_past_last_sample = (
        folder["trajectory_boundaries"][trajectory_number],
        folder["trajectory_boundaries"][trajectory_number + 1],
    )
    how_many_bounces = folder["how_many_bounces"][trajectory_number]
    return dict(
        sample_times=folder["sample_times"][first_sample:one_past_last_sample],
        positions=folder["positions"][first_sample:one_past_last_sample],
        launch_velocity=folder["launch_velocity"][trajectory_number],
        launch_spin=folder["launch_spin"][trajectory_number],
        bounce_times=folder["bounce_times"][trajectory_number][:how_many_bounces],
        trajectory_id=int(folder["trajectory_id"][trajectory_number]),
    )
