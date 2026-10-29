"""Manual error analysis of ridge on two cells: a random sample of 100 dev trajectories,
ranked by the ridge error, and the dev R^2 of both methods binned by shot type, spin,
speed and window length. Writes results/errors_<cell>.csv (fill the `cause` column by hand)
and results/error_bins_<cell>.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.cells import FEATURE_NAMES_PER_WINDOW, load_cell
from spinread.metrics import r_squared, root_mean_squared_error
from spinread.physics import SPIN_COMPONENT_NAMES

CELLS = [
    ("W2", "L3", "top"),
    ("W3", "L3", "gyro"),
]  # the cells of the report's Discussion
SHOWN_FEATURES = {
    "W2": ["how_many_frames", "too_few_frames", "fit_residual_rms"],
    "W3": ["how_many_frames", "how_many_frames_after", "too_few_frames_after_bounce"],
}
SAMPLE_SIZE, HOW_MANY_WORST, SAMPLE_SEED = 100, 10, 0
BIN_EDGES = {
    "spin_magnitude": [0, 20, 40, 60, 80],
    "launch_speed": [0, 8, 14, 20, 31],
    "window_duration": [0, 0.1, 0.2, 0.4, 0.6, 2.0],
}
pd.set_option("display.width", 250)


def analyse_one_cell(window, level, component):
    cell = load_cell(window, level)
    ridge = np.load(f"cache/prediction_ridge_{window}_{level}.npz")
    baseline = np.load(f"cache/prediction_baseline_{window}_{level}.npz")
    dev_rows = ridge["row_in_cell"]
    flight = cell["features"][dev_rows]
    rows = pd.DataFrame(
        {
            name: cell[name][dev_rows]
            for name in [
                "shot_type",
                "trajectory_id",
                "launch_speed",
                "window_duration",
                "spin_magnitude",
            ]
        }
        | {
            name: flight[:, FEATURE_NAMES_PER_WINDOW[window].index(name)]
            for name in SHOWN_FEATURES[window]
        }
    )
    for c, name in enumerate(SPIN_COMPONENT_NAMES):
        rows[f"true_{name}"] = ridge["spin_label"][:, c]
        rows[f"ridge_{name}"] = ridge["prediction"][:, c]
        rows[f"baseline_{name}"] = baseline["prediction"][:, c]
    rows["ridge_error"] = rows[f"ridge_{component}"] - rows[f"true_{component}"]
    rows["baseline_error"] = rows[f"baseline_{component}"] - rows[f"true_{component}"]

    sample = rows.sample(n=min(SAMPLE_SIZE, len(rows)), random_state=SAMPLE_SEED)
    sample = sample.loc[
        sample.ridge_error.abs().sort_values(ascending=False).index
    ].assign(cause="")
    tag = f"{window}_{level}_{component}"
    sample.to_csv(f"results/errors_{tag}.csv", index=False)

    groupings = [("shot_type", rows.shot_type)] + [
        (column, pd.cut(rows[column], edges, include_lowest=True).astype(str))
        for column, edges in BIN_EDGES.items()
    ]
    bins = pd.DataFrame(
        dict(
            binned_by=binned_by,
            bin=bin_value,
            how_many=len(group),
            **{
                f"{stat}_{method}": function(
                    group[f"true_{component}"], group[f"{method}_{component}"]
                )
                for method in ["baseline", "ridge"]
                for stat, function in [
                    ("r2", r_squared),
                    ("rmse", root_mean_squared_error),
                ]
            },
        )
        for binned_by, key in groupings
        for bin_value, group in rows.groupby(key, observed=True)
        if len(group) >= 20
    )
    bins.to_csv(f"results/error_bins_{tag}.csv", index=False)

    print(
        f"{window}-{level} {component}: {len(rows)} dev rows, ridge R^2 "
        f"{r_squared(rows[f'true_{component}'], rows[f'ridge_{component}']):.3f}"
    )
    print(
        f"\nThe {HOW_MANY_WORST} largest ridge errors among {SAMPLE_SIZE} random dev trajectories (rev/s):"
    )
    shown = [
        "shot_type",
        "trajectory_id",
        "launch_speed",
        "window_duration",
        *SHOWN_FEATURES[window],
        f"true_{component}",
        f"ridge_{component}",
        f"baseline_{component}",
    ]
    print(sample.head(HOW_MANY_WORST)[shown].round(3).to_string(index=False))
    print("\nAll dev rows of the cell, binned:\n", bins.round(3).to_string(index=False))
    print(f"\nWrote results/errors_{tag}.csv and results/error_bins_{tag}.csv\n")


def main():
    for window, level, component in CELLS:
        analyse_one_cell(window, level, component)


if __name__ == "__main__":
    main()
