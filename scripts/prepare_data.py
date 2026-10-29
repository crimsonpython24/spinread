"""Pack the released trajectories, clean them and split them 70/10/20.

Reads data/syntheticdata/<shot_type>/<direction>/trajectory_NNNN/ and writes one
cache/synthetic_<shot_type>_<direction>.npz per folder with every trajectory, its labels
and its split code, plus results/data_cleaning.csv, split_sizes.csv and trajectories.csv.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.data import (
    DEV,
    DIRECTIONS_OF_PLAY,
    DROPPED,
    SHOT_TYPES,
    SPLIT_NAMES,
    TEST,
    TRAIN,
    packed_path,
)
from spinread.physics import (
    GRAVITY_VECTOR,
    SAMPLES_PER_SECOND,
    air_acceleration,
    free_flight_end,
    spin_label,
)

SOURCE_ROOT = "data/syntheticdata"
HIGHEST_SPIN_THE_SIMULATOR_DRAWS = 500.0  # rad/s
COLLISION_THRESHOLD = (
    50.0  # m/s^2 off the air model before the first bounce counts as a hit
)
BOUNCES_EXPECTED = {  # table contacts a trajectory of each folder must have
    "intermediate": 1,
    "first_good": 2,
    "first_short": 2,
    "first_long": 1,
    "final_win": 2,
    "final_lose": 0,
}
CHECK_NAMES = [
    "has_nan",
    "wrong_time_step",
    "spin_too_high",
    "bad_bounce_times",
    "wrong_bounce_count",
    "hit_before_first_bounce",
]
SHUFFLE_SEED = 0
TRAIN_SHARE, DEV_SHARE = 0.7, 0.1  # the rest (0.2) is test


def read_one_trajectory(trajectory_folder):
    arrays = {
        name: np.load(os.path.join(trajectory_folder, f"{name}.npy"))
        for name in ["positions", "times", "velocities", "rotations", "bounces"]
    }
    # the first sample is mistimed in the release, so it is dropped
    return dict(
        positions=arrays["positions"][1:],
        sample_times=arrays["times"][1:],
        launch_velocity=arrays["velocities"][0],
        launch_spin=arrays["rotations"][0],
        bounce_times=np.atleast_1d(arrays["bounces"]),
    )


def largest_gap_from_air_model(trajectory):
    # the biggest departure of the pre-bounce acceleration from drag plus Magnus: free
    # flight stays under 1 m/s^2, any contact with the net or the table edge gives hundreds
    flight_end = free_flight_end(
        trajectory["sample_times"], trajectory["positions"], trajectory["bounce_times"]
    )
    flight = trajectory["positions"][:flight_end]
    if len(flight) < 3:
        return 0.0
    time_step = 1 / SAMPLES_PER_SECOND
    velocity = (flight[2:] - flight[:-2]) / (2 * time_step)
    acceleration = (flight[2:] - 2 * flight[1:-1] + flight[:-2]) / time_step**2
    gap = (
        acceleration
        - GRAVITY_VECTOR
        - air_acceleration(velocity, trajectory["launch_spin"])
    )
    return float(np.linalg.norm(gap, axis=1).max())


def failed_checks(trajectory, shot_type):
    times, bounces = trajectory["sample_times"], trajectory["bounce_times"]
    failed = dict(
        has_nan=any(np.isnan(array).any() for array in trajectory.values()),
        wrong_time_step=len(times) < 2
        or not np.allclose(np.diff(times), 1 / SAMPLES_PER_SECOND, atol=1e-6),
        spin_too_high=np.linalg.norm(trajectory["launch_spin"])
        > HIGHEST_SPIN_THE_SIMULATOR_DRAWS + 1e-6,
        bad_bounce_times=len(bounces) > 0
        and (
            np.any(np.diff(bounces) < 0)
            or bounces[0] < times[0]
            or bounces[-1] > times[-1]
        ),
        wrong_bounce_count=len(bounces) != BOUNCES_EXPECTED[shot_type],
    )
    failed["hit_before_first_bounce"] = (
        not (
            failed["has_nan"] or failed["wrong_time_step"] or failed["bad_bounce_times"]
        )
        and largest_gap_from_air_model(trajectory) > COLLISION_THRESHOLD
    )
    return failed


def prepare_one_folder(shot_type, direction):
    shot_folder = os.path.join(SOURCE_ROOT, shot_type, direction)
    folder_names = sorted(
        (name for name in os.listdir(shot_folder) if name.startswith("trajectory_")),
        key=lambda name: int(name.split("_")[1]),
    )
    trajectories = [
        read_one_trajectory(os.path.join(shot_folder, name)) for name in folder_names
    ]
    failed = pd.DataFrame(
        [failed_checks(trajectory, shot_type) for trajectory in trajectories]
    )
    usable = ~failed.any(axis=1).to_numpy()

    # the usable trajectories of the folder are shuffled once and cut 70/10/20
    usable_numbers = np.flatnonzero(usable)
    usable_numbers = usable_numbers[
        np.random.default_rng(SHUFFLE_SEED).permutation(len(usable_numbers))
    ]
    how_many_train = int(round(TRAIN_SHARE * len(usable_numbers)))
    how_many_dev = int(round(DEV_SHARE * len(usable_numbers)))
    split_code = np.full(len(trajectories), DROPPED, dtype=np.int8)
    split_code[usable_numbers[:how_many_train]] = TRAIN
    split_code[usable_numbers[how_many_train : how_many_train + how_many_dev]] = DEV
    split_code[usable_numbers[how_many_train + how_many_dev :]] = TEST

    bounce_rows = np.full((len(trajectories), 2), np.nan)
    for row, trajectory in zip(bounce_rows, trajectories):
        row[: min(2, len(trajectory["bounce_times"]))] = trajectory["bounce_times"][:2]
    np.savez(
        packed_path(shot_type, direction),
        positions=np.concatenate([t["positions"] for t in trajectories]).astype(
            np.float64
        ),
        sample_times=np.concatenate([t["sample_times"] for t in trajectories]).astype(
            np.float64
        ),
        trajectory_boundaries=np.concatenate(
            [[0], np.cumsum([len(t["positions"]) for t in trajectories])]
        ).astype(np.int64),
        launch_velocity=np.array([t["launch_velocity"] for t in trajectories]),
        launch_spin=np.array([t["launch_spin"] for t in trajectories]),
        bounce_times=bounce_rows,
        how_many_bounces=np.array([len(t["bounce_times"]) for t in trajectories]),
        trajectory_id=np.array([int(name.split("_")[1]) for name in folder_names]),
        split_code=split_code,
        shot_type=shot_type,
        direction_of_play=direction,
    )
    cleaning_row = dict(
        shot_type=shot_type,
        direction_of_play=direction,
        total=len(trajectories),
        **failed.sum().astype(int).to_dict(),
        usable=int(usable.sum()),
    )
    split_row = dict(
        shot_type=shot_type,
        direction_of_play=direction,
        **{name: int((split_code == code).sum()) for code, name in SPLIT_NAMES.items()},
        unusable=int((split_code == DROPPED).sum()),
    )
    trajectory_rows = [
        dict(
            shot_type=shot_type,
            direction_of_play=direction,
            trajectory_id=int(folder_names[number].split("_")[1]),
            split=SPLIT_NAMES[int(split_code[number])],
            launch_speed=float(np.linalg.norm(trajectory["launch_velocity"])),
            spin_magnitude=float(
                np.linalg.norm(trajectory["launch_spin"]) / (2 * np.pi)
            ),
            gyrospin=gyrospin,
            topspin=topspin,
            sidespin=sidespin,
            time_to_first_bounce=(
                float(trajectory["bounce_times"][0] - trajectory["sample_times"][0])
                if len(trajectory["bounce_times"])
                else np.nan
            ),
        )
        for number, trajectory in enumerate(trajectories)
        if split_code[number] != DROPPED
        for gyrospin, topspin, sidespin in [
            spin_label(trajectory["launch_spin"], trajectory["launch_velocity"])
        ]
    ]
    return cleaning_row, split_row, trajectory_rows


def main():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    folders = [
        (shot, direction) for shot in SHOT_TYPES for direction in DIRECTIONS_OF_PLAY
    ]
    with ProcessPoolExecutor() as workers:
        per_folder = list(workers.map(prepare_one_folder, *zip(*folders)))
    cleaning = pd.DataFrame([row for row, _, _ in per_folder])
    split_sizes = pd.DataFrame([row for _, row, _ in per_folder])
    trajectories = pd.DataFrame([row for _, _, rows in per_folder for row in rows])
    cleaning.to_csv("results/data_cleaning.csv", index=False)
    split_sizes.to_csv("results/split_sizes.csv", index=False)
    trajectories.to_csv("results/trajectories.csv", index=False)
    print(cleaning.to_string(index=False))
    print("\n", split_sizes.to_string(index=False))
    print(
        f"\n{len(trajectories)} usable trajectories: topspin share "
        f"{(trajectories.topspin > 0).mean():.2f}, label std top {trajectories.topspin.std():.1f} "
        f"side {trajectories.sidespin.std():.1f} rev/s"
    )
    print(
        "Wrote cache/synthetic_*.npz, results/data_cleaning.csv, split_sizes.csv, trajectories.csv"
    )


if __name__ == "__main__":
    main()
