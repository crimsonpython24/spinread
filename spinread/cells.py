import os

import numpy as np
from scipy.interpolate import CubicSpline

from spinread.physics import (
    BALL_RADIUS,
    GRAVITY,
    GRAVITY_VECTOR,
    TABLE_HEIGHT,
    ball_frame,
    first_contact_sample,
    fit_polynomial_per_axis,
    free_flight_end,
)
from spinread.data import DEV, DIRECTIONS_OF_PLAY, SHOT_TYPES

WINDOWS = ["W1", "W2", "W3"]  # first 100 ms; whole flight; flight plus the bounce
LEVELS = {
    "L1": dict(every_nth_sample=1, noise=0.0),
    "L2": dict(frames_per_second=145.0, noise=0.001, timestamp_jitter=0.00045),
    "L3": dict(every_nth_sample=10, noise=0.005),
}
W1_LENGTH = 0.1  # s
AFTER_CONTACT_LENGTH = 0.05  # s of W3 after the first contact
BEFORE_CONTACT_FIT_LENGTH = (
    0.1  # s before contact used to find the impact and the pre-bounce velocity
)
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
# (velocity x magnus acceleration) / |velocity|^2 per axis: what the physics baseline
# inverts, in the order of the three label components (gyro about x~, top about y~, side
# about z~); its x~ part is near zero, so the baseline cannot read gyrospin
SPIN_TIMES_K_COLUMNS = [
    FLIGHT_FEATURE_NAMES.index(f"spin_times_k_{axis}") for axis in AXES
]
TOO_FEW_FRAMES_COLUMN = FLIGHT_FEATURE_NAMES.index("too_few_frames")
BOUNCE_FEATURE_NAMES = (
    # velocity of the pre-bounce arc at the estimated impact time
    [f"velocity_before_{axis}" for axis in AXES]
    # velocity right after the impact, from the gravity-fixed fit
    + [f"velocity_after_{axis}" for axis in AXES]
    # after minus before: the x~ part is the along-track kick, the y~ part the sideways kick
    + [f"velocity_change_{axis}" for axis in AXES]
    + [
        "restitution",  # |vertical speed after / vertical speed before|
        "how_many_frames_after",  # frames in the 50 ms after the contact
        "impact_gap",  # time from the last pre-bounce frame to the estimated impact
        "has_bounce",  # 1 for every shot that bounces at all
        "too_few_frames_after_bounce",  # 1 when fewer than 2 frames show the bounce (kicks left at 0)
    ]
)
# the W3 feature vector: the whole-flight features followed by the bounce features
FEATURE_NAMES_PER_WINDOW = {
    "W1": FLIGHT_FEATURE_NAMES,
    "W2": FLIGHT_FEATURE_NAMES,
    "W3": FLIGHT_FEATURE_NAMES + BOUNCE_FEATURE_NAMES,
}


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


def impact_time(arc_before_contact, first_time_after_contact):
    # when the pre-bounce arc crosses the contact height, after its last frame
    height_above_contact = arc_before_contact["position_at_centre"][2] - CONTACT_HEIGHT
    vertical_velocity, half_vertical_acceleration = (
        arc_before_contact["velocity_at_centre"][2],
        arc_before_contact["acceleration"][2] / 2,
    )
    if abs(half_vertical_acceleration) > 1e-9:
        discriminant = (
            vertical_velocity**2 - 4 * half_vertical_acceleration * height_above_contact
        )
        roots = (
            []
            if discriminant < 0
            else [
                (-vertical_velocity - np.sqrt(discriminant))
                / (2 * half_vertical_acceleration),
                (-vertical_velocity + np.sqrt(discriminant))
                / (2 * half_vertical_acceleration),
            ]
        )
    else:
        roots = (
            [-height_above_contact / vertical_velocity]
            if abs(vertical_velocity) > 1e-9
            else []
        )
    crossings_after_last_frame = [
        arc_before_contact["centre_time"] + root
        for root in roots
        if arc_before_contact["centre_time"] + root
        >= arc_before_contact["last_time"] - TINY
    ]
    when = (
        min(crossings_after_last_frame)
        if crossings_after_last_frame
        else arc_before_contact["last_time"]
    )
    return float(
        np.clip(
            when,
            arc_before_contact["last_time"],
            max(first_time_after_contact, arc_before_contact["last_time"]),
        )
    )


def bounce_features(
    arc_before_contact, frame, times_after_contact, positions_after_contact
):
    # the velocity just before and just after the impact, in the estimated ball frame
    features = np.zeros(len(BOUNCE_FEATURE_NAMES))
    features[10], features[12] = (
        len(times_after_contact),
        1.0,
    )  # frames after, has_bounce
    if arc_before_contact is None or len(times_after_contact) < 2:
        features[13] = 1.0  # too few frames after the bounce: the kicks stay 0
        return features
    when_the_ball_hit = impact_time(arc_before_contact, times_after_contact[0])
    velocity_before = velocity_of_arc_at(arc_before_contact, when_the_ball_hit)
    time_since_impact = times_after_contact - when_the_ball_hit
    positions_without_gravity = positions_after_contact.copy()
    positions_without_gravity[:, 2] += GRAVITY / 2 * time_since_impact**2
    coefficients, _ = fit_polynomial_per_axis(
        time_since_impact, positions_without_gravity, 1, 0.0
    )
    velocity_after = coefficients[1]
    features[0:3], features[3:6], features[6:9] = (
        frame @ velocity_before,
        frame @ velocity_after,
        frame @ (velocity_after - velocity_before),
    )
    features[9] = (
        abs(velocity_after[2] / velocity_before[2])
        if abs(velocity_before[2]) > 1e-6
        else 0.0
    )
    features[11] = when_the_ball_hit - arc_before_contact["last_time"]
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
    if len(bounce_times) and off_table_sample < len(sample_times):
        # W3 adds the frames from the first sample clear of the table to 50 ms after contact
        after_contact = (true_times >= sample_times[off_table_sample] - TINY) & (
            true_times <= sample_times[contact_sample] + AFTER_CONTACT_LENGTH + TINY
        )
    else:
        after_contact = np.zeros(len(true_times), bool)

    features_per_window, fits = {}, {}
    for window, in_window in [("W1", in_w1), ("W2", in_w2)]:
        arc = fit_arc(reported_times[in_window], frame_positions[in_window])
        frame = estimated_ball_frame(arc)
        first_frame_time = reported_times[in_window][0] if in_window.any() else 0.0
        features_per_window[window] = flight_features(
            arc, frame, int(in_window.sum()), first_frame_time
        )
        fits[window] = (arc, frame)
    w2_arc, w2_frame = fits["W2"]
    if len(bounce_times):
        # the pre-bounce velocity comes from an arc fitted to the last 100 ms of flight,
        # which a long flight's single quadratic would get wrong
        near_the_contact = in_w2 & (
            true_times >= last_free_flight_time - BEFORE_CONTACT_FIT_LENGTH - TINY
        )
        arc_before_contact = (
            fit_arc(reported_times[near_the_contact], frame_positions[near_the_contact])
            if near_the_contact.sum() >= 3
            else w2_arc
        )
        fits["pre"] = (arc_before_contact, w2_frame)
        w3_bounce_features = bounce_features(
            arc_before_contact,
            w2_frame,
            reported_times[after_contact],
            frame_positions[after_contact],
        )
    else:
        w3_bounce_features = np.zeros(
            len(BOUNCE_FEATURE_NAMES)
        )  # no bounce: has_bounce = 0
    features_per_window["W3"] = np.concatenate(
        [features_per_window["W2"], w3_bounce_features]
    )
    if not want_details:
        return features_per_window
    return features_per_window, dict(
        reported_times=reported_times,
        true_times_since_start=true_times_since_start,
        frame_positions=frame_positions,
        in_w1=in_w1,
        in_w2=in_w2,
        after_contact=after_contact,
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
