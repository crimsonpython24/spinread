import os

import numpy as np
from scipy.interpolate import CubicSpline

from spinread.physics import (
    BALL_RADIUS,
    GRAVITY_VECTOR,
    TABLE_HEIGHT,
    ball_frame,
    first_contact_sample,
    fit_polynomial_per_axis,
    free_flight_end,
)
from spinread.data import DEV, DIRECTIONS_OF_PLAY, SHOT_TYPES

WINDOWS = ["W1", "W2"]
LEVELS = {
    "L1": dict(every_nth_sample=1, noise=0.0),
    "L2": dict(frames_per_second=145.0, noise=0.001, timestamp_jitter=0.00045),
    "L3": dict(every_nth_sample=10, noise=0.005),
    "L4": dict(every_nth_sample=20, noise=0.005),
}
W1_LENGTH = 0.1  # s
CONTACT_HEIGHT = (
    TABLE_HEIGHT + BALL_RADIUS
)  # the ball centre sits at 0.78 m when it touches
TINY = 1e-9  # a tolerance for comparing times

AXES = [
    "x",
    "y",
    "z",
]  # x~ along the horizontal flight direction, y~ = z~ cross x~, z~ up
# the order of these names is the order of the columns in the feature arrays.
FLIGHT_FEATURE_NAMES = (
    # fitted velocity at the first frame
    [f"velocity_start_{axis}" for axis in AXES]
    # fitted velocity at the window centre
    + [f"velocity_centre_{axis}" for axis in AXES]
    # fitted acceleration with gravity removed
    + [f"acceleration_{axis}" for axis in AXES]
    # the part of that acceleration perpendicular to the velocity: the Magnus part, drag removed
    + [f"magnus_acceleration_{axis}" for axis in AXES]
    # (velocity x magnus acceleration) / |velocity|^2, which is about k times the spin
    + [f"spin_times_k_{axis}" for axis in AXES]
    + [
        "speed",  # |velocity| at the window centre
        "fit_residual_rms",  # how far the frames sit from the fitted arc (a noise proxy)
        "how_many_frames",
        "window_duration",  # time from the first to the last frame
        "first_frame_time",  # counted from the first kept sample
        "too_few_frames",  # 1 when fewer than 3 frames, so no acceleration could be fitted
    ]
)
# the y~ and z~ parts of (velocity x magnus acceleration) / |velocity|^2: what the physics
# baseline inverts, in the order of the two label components (topspin, sidespin)
SPIN_TIMES_K_COLUMNS = [
    FLIGHT_FEATURE_NAMES.index(f"spin_times_k_{axis}") for axis in ["y", "z"]
]
TOO_FEW_FRAMES_COLUMN = FLIGHT_FEATURE_NAMES.index("too_few_frames")


def seed_for(level, shot_type, direction, trajectory_id):
    # the random draws of the sensor (frame phase, timestamp jitter, noise) depend only on
    # the trajectory and the level, so every script sees the same frames
    return [
        list(LEVELS).index(level),
        SHOT_TYPES.index(shot_type),
        DIRECTIONS_OF_PLAY.index(direction),
        int(trajectory_id),
    ]


def first_sample_off_the_table(positions, contact_sample):
    sample = contact_sample
    while sample < len(positions) and positions[sample, 2] < CONTACT_HEIGHT + 0.001:
        sample += 1
    return sample


def interpolate_positions(sample_times, positions, wanted_times):
    if len(sample_times) >= 4:
        return CubicSpline(sample_times, positions)(wanted_times)
    return np.stack(
        [
            np.interp(wanted_times, sample_times, positions[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )


def resample_around_the_bounce(
    sample_times, positions, contact_sample, off_table_sample, wanted_times
):
    if contact_sample >= len(sample_times):
        return interpolate_positions(sample_times, positions, wanted_times)
    resampled = np.empty((len(wanted_times), 3))
    before_contact = wanted_times <= sample_times[contact_sample]
    after_contact = wanted_times >= sample_times[off_table_sample - 1]
    during_contact = ~before_contact & ~after_contact
    resampled[before_contact] = interpolate_positions(
        sample_times[: contact_sample + 1],
        positions[: contact_sample + 1],
        wanted_times[before_contact],
    )
    if after_contact.any():
        resampled[after_contact] = interpolate_positions(
            sample_times[off_table_sample - 1 :],
            positions[off_table_sample - 1 :],
            wanted_times[after_contact],
        )
    if during_contact.any():
        resampled[during_contact] = np.stack(
            [
                np.interp(
                    wanted_times[during_contact], sample_times, positions[:, axis]
                )
                for axis in range(3)
            ],
            axis=1,
        )
    return resampled


def what_the_sensor_sees(
    sample_times, positions, contact_sample, off_table_sample, level, random_draws
):
    level_spec = LEVELS[level]
    if "frames_per_second" in level_spec:
        frames_per_second = level_spec["frames_per_second"]
        true_frame_times = (
            sample_times[0]
            + random_draws.uniform(0, 1 / frames_per_second)
            + np.arange(len(sample_times)) / frames_per_second
        )
        true_frame_times = true_frame_times[true_frame_times <= sample_times[-1]]
        frame_positions = resample_around_the_bounce(
            sample_times, positions, contact_sample, off_table_sample, true_frame_times
        )
        reported_frame_times = (
            np.round(
                (
                    true_frame_times
                    + random_draws.normal(
                        0, level_spec["timestamp_jitter"], len(true_frame_times)
                    )
                )
                * 1000
            )
            / 1000
        )
    else:
        kept_samples = np.arange(
            random_draws.integers(0, level_spec["every_nth_sample"]),
            len(sample_times),
            level_spec["every_nth_sample"],
        )
        true_frame_times, frame_positions = (
            sample_times[kept_samples],
            positions[kept_samples],
        )
        reported_frame_times = true_frame_times
    if level_spec["noise"]:
        frame_positions = frame_positions + random_draws.normal(
            0, level_spec["noise"], frame_positions.shape
        )
    return reported_frame_times, true_frame_times, frame_positions


def fit_arc(frame_times, frame_positions):
    how_many_frames = len(frame_times)
    if how_many_frames < 2:
        return None
    centre_time = frame_times.mean()
    degree = 2 if how_many_frames >= 3 else 1
    coefficients, design_matrix = fit_polynomial_per_axis(
        frame_times, frame_positions, degree, centre_time
    )
    residuals = frame_positions - design_matrix @ coefficients
    acceleration = 2 * coefficients[2] if degree == 2 else np.zeros(3)
    return dict(
        centre_time=centre_time,
        position_at_centre=coefficients[0],
        velocity_at_centre=coefficients[1],
        acceleration=acceleration,
        degree=degree,
        how_many_frames=how_many_frames,
        first_time=frame_times[0],
        last_time=frame_times[-1],
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
    )


def velocity_of_arc_at(arc, time):
    return arc["velocity_at_centre"] + arc["acceleration"] * (time - arc["centre_time"])


def estimated_ball_frame(arc):
    if arc is None:
        return np.eye(3)
    velocity_at_first_frame = velocity_of_arc_at(arc, arc["first_time"])
    if np.hypot(velocity_at_first_frame[0], velocity_at_first_frame[1]) < 1e-6:
        return np.eye(3)
    return ball_frame(velocity_at_first_frame)


def flight_features(arc, frame, how_many_frames, first_frame_time):
    features = np.zeros(len(FLIGHT_FEATURE_NAMES))
    features[17], features[19], features[20] = (
        how_many_frames,
        first_frame_time,
        float(how_many_frames < 3),
    )
    if arc is None:
        return features
    velocity_at_start, velocity_at_centre = (
        velocity_of_arc_at(arc, arc["first_time"]),
        arc["velocity_at_centre"],
    )
    features[0:3], features[3:6] = frame @ velocity_at_start, frame @ velocity_at_centre
    features[15], features[16], features[18] = (
        np.linalg.norm(velocity_at_centre),
        arc["residual_rms"],
        arc["last_time"] - arc["first_time"],
    )
    if arc["degree"] == 2:  # with too few frames the acceleration features stay at 0
        acceleration_without_gravity = arc["acceleration"] - GRAVITY_VECTOR
        flight_direction = velocity_at_centre / np.linalg.norm(velocity_at_centre)
        magnus_acceleration = (
            acceleration_without_gravity
            - np.dot(acceleration_without_gravity, flight_direction) * flight_direction
        )
        features[6:9], features[9:12] = (
            frame @ acceleration_without_gravity,
            frame @ magnus_acceleration,
        )
        features[12:15] = frame @ (
            np.cross(velocity_at_centre, magnus_acceleration)
            / np.dot(velocity_at_centre, velocity_at_centre)
        )
    return features


def features_of_one_trajectory(
    sample_times, positions, bounce_times, level, random_draws, want_details=False
):
    # the whole recording is passed through the sensor, so that the frames it records before
    # the contact are the ones a camera watching the full rally would have recorded
    contact_sample = first_contact_sample(sample_times, positions, bounce_times)
    off_table_sample = (
        first_sample_off_the_table(positions, contact_sample)
        if len(bounce_times)
        else len(sample_times)
    )
    flight_end = free_flight_end(sample_times, positions, bounce_times)
    last_free_flight_time = sample_times[flight_end - 1] if flight_end > 0 else -np.inf
    reported_times, true_times, frame_positions = what_the_sensor_sees(
        sample_times, positions, contact_sample, off_table_sample, level, random_draws
    )
    reported_times = (
        reported_times - sample_times[0]
    )  # every feature time counts from the first kept sample
    true_times_since_start = true_times - sample_times[0]

    in_w2 = true_times <= last_free_flight_time + TINY
    in_w1 = in_w2 & (true_times_since_start < W1_LENGTH - TINY)

    features_per_window, fits = {}, {}
    for window, in_window in [("W1", in_w1), ("W2", in_w2)]:
        arc = fit_arc(reported_times[in_window], frame_positions[in_window])
        frame = estimated_ball_frame(arc)
        first_frame_time = reported_times[in_window][0] if in_window.any() else 0.0
        features_per_window[window] = flight_features(
            arc, frame, int(in_window.sum()), first_frame_time
        )
        fits[window] = (arc, frame)
    if not want_details:
        return features_per_window
    return features_per_window, dict(
        reported_times=reported_times,
        true_times_since_start=true_times_since_start,
        frame_positions=frame_positions,
        in_w1=in_w1,
        in_w2=in_w2,
        fits=fits,
        contact_time=(
            sample_times[contact_sample] - sample_times[0]
            if contact_sample < len(sample_times)
            else None
        ),
    )


def cell_path(window, level, cache_dir="cache"):
    return os.path.join(cache_dir, f"cell_{window}_{level}.npz")


def load_cell(window, level, cache_dir="cache"):
    cell_file = np.load(cell_path(window, level, cache_dir))
    return {array_name: cell_file[array_name] for array_name in cell_file.files}


def save_dev_predictions(
    method, window, level, cell, dev_predictions, cache_dir="cache"
):
    # one method's predictions on the dev rows of a cell, next to their labels
    is_dev = cell["split_code"] == DEV
    np.savez(
        os.path.join(cache_dir, f"prediction_{method}_{window}_{level}.npz"),
        prediction=np.asarray(dev_predictions, dtype=np.float32),
        spin_label=cell["spin_label"][is_dev].astype(np.float32),
        row_in_cell=np.flatnonzero(is_dev),
    )
