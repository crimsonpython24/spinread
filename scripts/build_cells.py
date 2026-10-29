"""Cut every usable trajectory into the 9 condition cells (windows W1-W3 x levels L1-L3).

Writes cache/cell_<window>_<level>.npz (the 21 flight features, the label and the split code
of every trajectory) and results/fig_windows.png, one rally shot as each level sees it.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.cells import (
    FEATURE_NAMES_PER_WINDOW,
    FLIGHT_FEATURE_NAMES,
    LEVELS,
    WINDOWS,
    cell_path,
    features_of_one_trajectory,
    seed_for,
)
from spinread.data import DROPPED, get_one_trajectory, load_packed_folders, packed_path
from spinread.physics import ball_frame, spin_label

BLUE, ORANGE, GRAY = "#4472C4", "#ED7D31", "#A5A5A5"
LEVEL_DESCRIPTIONS = {
    "L1": "500 Hz, no noise",
    "L2": "145 Hz, 1 mm noise",
    "L3": "50 Hz, 5 mm noise",
}
HOW_MANY_FRAMES = FLIGHT_FEATURE_NAMES.index("how_many_frames")
FIT_RESIDUAL = FLIGHT_FEATURE_NAMES.index("fit_residual_rms")
DURATION = FLIGHT_FEATURE_NAMES.index("window_duration")


def features_of_one_folder(shot_type, direction, level):
    packed_file = np.load(packed_path(shot_type, direction))
    folder = {name: packed_file[name] for name in packed_file.files}
    usable_numbers = np.flatnonzero(folder["split_code"] != DROPPED)
    features = {window: [] for window in WINDOWS}
    labels = []
    for number in usable_numbers:
        trajectory = get_one_trajectory(folder, number)
        per_window = features_of_one_trajectory(
            trajectory["sample_times"],
            trajectory["positions"],
            trajectory["bounce_times"],
            level,
            np.random.default_rng(
                seed_for(level, shot_type, direction, trajectory["trajectory_id"])
            ),
        )
        for window in WINDOWS:
            features[window].append(per_window[window])
        labels.append(
            spin_label(trajectory["launch_spin"], trajectory["launch_velocity"])
        )
    return dict(
        features={window: np.array(rows) for window, rows in features.items()},
        spin_label=np.array(labels),
        split_code=folder["split_code"][usable_numbers],
        shot_type=np.full(len(usable_numbers), shot_type),
        direction_of_play=np.full(len(usable_numbers), direction),
        trajectory_id=folder["trajectory_id"][usable_numbers],
        trajectory_number=usable_numbers,
        launch_speed=np.linalg.norm(folder["launch_velocity"][usable_numbers], axis=1),
        spin_magnitude=np.linalg.norm(folder["launch_spin"][usable_numbers], axis=1)
        / (2 * np.pi),
    )


def write_cells(level, per_folder):
    for window in WINDOWS:
        flight = np.concatenate([result["features"][window] for result in per_folder])
        np.savez(
            cell_path(window, level),
            features=flight.astype(np.float32),
            window_duration=flight[:, DURATION].astype(np.float32),
            feature_names=np.array(FEATURE_NAMES_PER_WINDOW[window]),
            **{
                name: np.concatenate([result[name] for result in per_folder])
                for name in per_folder[0]
                if name != "features"
            },
        )
        print(
            f"  {window}-{level}: {len(flight)} trajectories, "
            f"{int(flight[:, FLIGHT_FEATURE_NAMES.index('too_few_frames')].sum())} with too few frames"
        )


def draw_windows_figure(shot_type, direction, trajectory, path):
    # one trajectory, not an average: the true path, the frames each level records in W2,
    # the subset in W1, and the quadratic fitted to the W2 frames; one panel per level
    positions = trajectory["positions"]
    true_frame = ball_frame(trajectory["launch_velocity"])
    along_launch = (positions - positions[0]) @ true_frame[0]
    figure, panels = plt.subplots(1, len(LEVELS), figsize=(5.6, 2.0), sharey=True)
    for panel, level in zip(panels, LEVELS):
        features, details = features_of_one_trajectory(
            trajectory["sample_times"],
            trajectory["positions"],
            trajectory["bounce_times"],
            level,
            np.random.default_rng(
                seed_for(level, shot_type, direction, trajectory["trajectory_id"])
            ),
            want_details=True,
        )
        frames = details["frame_positions"]
        frames_along_launch = (frames - positions[0]) @ true_frame[0]
        in_w1, in_w2 = details["in_w1"], details["in_w2"]
        panel.plot(
            along_launch,
            positions[:, 2],
            color=GRAY,
            lw=2.5,
            alpha=0.6,
            label="true path (every 2 ms)",
        )
        panel.axhline(0.78, color=GRAY, lw=0.8, ls=":", label="table height")
        if in_w1.sum() > 20:  # dense frames: a band under W1 instead of a ring on each
            panel.plot(
                frames_along_launch[in_w1],
                frames[in_w1, 2],
                "-",
                lw=8,
                alpha=0.3,
                color=ORANGE,
                solid_capstyle="round",
                label="W1 frames (first 100 ms)",
            )
        else:
            panel.plot(
                frames_along_launch[in_w1],
                frames[in_w1, 2],
                "o",
                ms=7,
                mfc="none",
                mew=1.3,
                color=ORANGE,
                label="W1 frames (first 100 ms)",
            )
        panel.plot(
            frames_along_launch[in_w2],
            frames[in_w2, 2],
            "o",
            ms=3,
            color=BLUE,
            label="W2 frames (before the bounce)",
        )
        after = details["after_contact"]
        panel.plot(
            frames_along_launch[after],
            frames[after, 2],
            "s",
            ms=3.5,
            mfc="none",
            mew=1.2,
            color=BLUE,
            label="frames up to 50 ms after the bounce (W3)",
        )
        arc, _ = details["fits"]["W2"]
        if arc is not None:
            since_centre = (
                np.linspace(arc["first_time"], arc["last_time"], 50)
                - arc["centre_time"]
            )
            fitted = (
                arc["position_at_centre"]
                + np.outer(since_centre, arc["velocity_at_centre"])
                + 0.5 * np.outer(since_centre**2, arc["acceleration"])
            )
            panel.plot(
                (fitted - positions[0]) @ true_frame[0],
                fitted[:, 2],
                "-",
                color=ORANGE,
                lw=1.3,
                label="quadratic fitted to W2",
            )
        panel.set_title(
            f"{level}: {LEVEL_DESCRIPTIONS[level]}\n"
            f"frames: W1 {int(features['W1'][HOW_MANY_FRAMES])}, "
            f"W2 {int(features['W2'][HOW_MANY_FRAMES])}, W3 +{int(after.sum())}",
            fontsize=7,
        )
        panel.tick_params(labelsize=6.5)
        panel.grid(alpha=0.15)
    panels[0].set_ylabel("height above the floor (m)", fontsize=7)
    figure.supxlabel("distance along the launch direction (m)", fontsize=7, y=0.2)
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=6.5,
        frameon=False,
    )
    gyro, top, side = spin_label(
        trajectory["launch_spin"], trajectory["launch_velocity"]
    )
    figure.suptitle(
        f"Rally shot {shot_type}/{direction}/{trajectory['trajectory_id']:04d} "
        f"(gyro {gyro:.0f}, top {top:.0f}, side {side:.0f} rev/s) at each sensing level",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.37, wspace=0.08)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    os.makedirs("results", exist_ok=True)
    folders = list(load_packed_folders())
    for level in LEVELS:
        started_at = time.time()
        with ProcessPoolExecutor() as workers:
            per_folder = list(
                workers.map(
                    features_of_one_folder,
                    *zip(*[(shot, direction, level) for shot, direction in folders]),
                )
            )
        print(f"{level}: features built in {time.time() - started_at:.0f} s")
        write_cells(level, per_folder)

    # the figure shows the first usable rally shot whose first 100 ms are all in flight
    folder = load_packed_folders()[("intermediate", "left_to_right")]
    for number in np.flatnonzero(folder["split_code"] != DROPPED):
        trajectory = get_one_trajectory(folder, number)
        flight = features_of_one_trajectory(
            trajectory["sample_times"],
            trajectory["positions"],
            trajectory["bounce_times"],
            "L1",
            np.random.default_rng(0),
        )["W1"]
        if flight[DURATION] >= 0.09:
            break
    draw_windows_figure(
        "intermediate", "left_to_right", trajectory, "results/fig_windows.png"
    )
    print(
        f"Wrote cache/cell_<window>_<level>.npz and results/fig_windows.png (trajectory {trajectory['trajectory_id']})"
    )


if __name__ == "__main__":
    main()
