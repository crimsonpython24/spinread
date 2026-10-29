import glob
import os

import numpy as np

from spinread.cells import (
    CONTACT_HEIGHT,
    TINY,
    W1_LENGTH,
    estimated_ball_frame,
    fit_arc,
    flight_features,
)

# SpinDOE (Gossard et al., 2023) releases 232 tracks of a ball thrower, recorded at about
# 145 Hz: index.csv holds the measured spin (rad/s, table frame, z up) and NNN.csv the
# positions (s; m, ';'-separated). Heights are counted from the table top at the ball's
# contact, so the contact plane is at 0; we lift them to the simulator's contact height.
HIGHEST_SPIN_THE_SIMULATOR_DRAWS = (
    500.0  # rad/s; faster balls were never seen in training
)
BOUNCE_HALF_WINDOW = 3  # samples each side in which the contact frame is the lowest
BOUNCE_HEIGHT_LIMIT = (
    0.06  # m above contact: a local minimum higher than this is no contact
)
BOUNCE_RISE = 0.03  # m the ball must climb on both sides of the minimum
TABLE_HALF_X, TABLE_HALF_Y = 0.7625 + 0.05, 1.37 + 0.05  # the table, with a margin
OFF_THE_TABLE_HEIGHT = (
    0.03  # m: without a contact, flight ends where the ball drops below this
)
KINK_THRESHOLD = (
    0.01  # m RMS: a flight whose quadratic misses its frames by more hit something
)
# id ranges in which the flight curvature and the spin label agree (results/spindoe_blocks.csv);
# a post hoc subset, reported next to the full set, never instead of it
CONSISTENT_ID_RANGES = [(0, 60), (100, 220)]


def load_spindoe(data_dir):
    index = np.loadtxt(
        os.path.join(data_dir, "index.csv"), delimiter=",", skiprows=1, ndmin=2
    )
    spins = {int(row[0]): row[1:4] for row in index}
    tracks = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.basename(path)
        if name == "index.csv":
            continue
        tracks[int(name.split(".")[0])] = np.loadtxt(path, delimiter=";", ndmin=2)
    return tracks, spins


def table_contacts(track):
    # local minima of the height near the contact plane, over the table, with a climb on both sides
    heights, across, along = track[:, 3], track[:, 1], track[:, 2]
    how_many, contacts = len(heights), []
    for sample in range(2, how_many - 2):
        first, one_past_last = (
            max(0, sample - BOUNCE_HALF_WINDOW),
            min(how_many, sample + BOUNCE_HALF_WINDOW + 1),
        )
        if (
            heights[sample] != heights[first:one_past_last].min()
            or heights[sample] >= BOUNCE_HEIGHT_LIMIT
            or heights[:sample].max() - heights[sample] < BOUNCE_RISE
            or heights[sample + 1 :].max() - heights[sample] < BOUNCE_RISE
            or abs(across[sample]) > TABLE_HALF_X
            or abs(along[sample]) > TABLE_HALF_Y
            or (contacts and sample - contacts[-1] <= BOUNCE_HALF_WINDOW)
        ):
            continue
        contacts.append(sample)
    return contacts


def free_flight_end(track, contacts):
    # one past the last free-flight sample: the frame before the contact minimum may already touch
    if contacts:
        return contacts[0] - 1
    below = np.flatnonzero(track[:, 3] < OFF_THE_TABLE_HEIGHT)
    return int(below[0]) - 1 if len(below) else len(track)


def features_of_one_track(track):
    """The W1 and W2 features of a SpinDOE track, built the way build_cells.py builds them
    for a simulated trajectory, except that the frames are the ones the camera recorded.
    Returns the features per window, the ball frame of the label, and a few facts about the track.
    """
    contacts = table_contacts(track)
    flight_end = free_flight_end(track, contacts)
    times = track[:, 0] - track[0, 0]
    positions = track[:, 1:4] + np.array([0.0, 0.0, CONTACT_HEIGHT])
    in_w2 = np.arange(len(track)) < flight_end
    in_w1 = in_w2 & (times < W1_LENGTH - TINY)
    arcs, frames = {}, {}
    features_per_window = {}
    for window, in_window in [("W1", in_w1), ("W2", in_w2)]:
        arcs[window] = fit_arc(times[in_window], positions[in_window])
        frames[window] = estimated_ball_frame(arcs[window])
        features_per_window[window] = flight_features(
            arcs[window], frames[window], int(in_window.sum()), 0.0
        )
    # the label frame: the heading at the first frame, from the whole-flight arc, because
    # SpinDOE tracks begin where the camera first sees the ball and record no launch velocity
    facts = dict(
        how_many_frames=len(track),
        fit_residual_mm=(
            1000.0 * arcs["W2"]["residual_rms"] if arcs["W2"] is not None else np.nan
        ),
        clean=bool(
            arcs["W2"] is not None and arcs["W2"]["residual_rms"] <= KINK_THRESHOLD
        ),
        frames_in_flight=int(in_w2.sum()),
        flight_duration=float(times[flight_end - 1]) if flight_end > 0 else 0.0,
        has_bounce=bool(contacts),
        speed=(
            float(np.linalg.norm(arcs["W2"]["velocity_at_centre"]))
            if arcs["W2"]
            else np.nan
        ),
    )
    return features_per_window, frames["W2"], facts
