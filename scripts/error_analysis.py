import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.cells import FLIGHT_FEATURE_NAMES, load_cell
from spinread.metrics import r_squared, root_mean_squared_error
from spinread.physics import SPIN_COMPONENT_NAMES

WINDOW, LEVEL, COMPONENT = "W2", "L3", "top"  # the cell of the report's Discussion
SAMPLE_SIZE, HOW_MANY_WORST, SAMPLE_SEED = 100, 10, 0
BIN_EDGES = {
    "spin_magnitude": [0, 20, 40, 60, 80],
    "launch_speed": [0, 8, 14, 20, 31],
    "window_duration": [0, 0.1, 0.2, 0.4, 0.6, 2.0],
}
pd.set_option("display.width", 250)


def main():
    cell = load_cell(WINDOW, LEVEL)
    ridge = np.load(f"cache/prediction_ridge_{WINDOW}_{LEVEL}.npz")
    baseline = np.load(f"cache/prediction_baseline_{WINDOW}_{LEVEL}.npz")
    dev_rows = ridge["row_in_cell"]
    flight = cell["flight_features"][dev_rows]
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
            name: flight[:, FLIGHT_FEATURE_NAMES.index(name)]
            for name in ["how_many_frames", "too_few_frames", "fit_residual_rms"]
        }
    )
    for c, name in enumerate(SPIN_COMPONENT_NAMES):
        rows[f"true_{name}"] = ridge["spin_label"][:, c]
        rows[f"ridge_{name}"] = ridge["prediction"][:, c]
        rows[f"baseline_{name}"] = baseline["prediction"][:, c]
    rows["ridge_error"] = rows[f"ridge_{COMPONENT}"] - rows[f"true_{COMPONENT}"]
    rows["baseline_error"] = rows[f"baseline_{COMPONENT}"] - rows[f"true_{COMPONENT}"]

    sample = rows.sample(n=min(SAMPLE_SIZE, len(rows)), random_state=SAMPLE_SEED)
    sample = sample.loc[
        sample.ridge_error.abs().sort_values(ascending=False).index
    ].assign(cause="")
    tag = f"{WINDOW}_{LEVEL}_{COMPONENT}"
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
                    group[f"true_{COMPONENT}"], group[f"{method}_{COMPONENT}"]
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
        f"{WINDOW}-{LEVEL} {COMPONENT}: {len(rows)} dev rows, ridge R^2 "
        f"{r_squared(rows[f'true_{COMPONENT}'], rows[f'ridge_{COMPONENT}']):.3f}"
    )
    print(
        f"\nThe {HOW_MANY_WORST} largest ridge errors among {SAMPLE_SIZE} random dev trajectories (rev/s):"
    )
    shown = [
        "shot_type",
        "trajectory_id",
        "launch_speed",
        "window_duration",
        "how_many_frames",
        "fit_residual_rms",
        f"true_{COMPONENT}",
        f"ridge_{COMPONENT}",
        f"baseline_{COMPONENT}",
    ]
    print(sample.head(HOW_MANY_WORST)[shown].round(3).to_string(index=False))
    print("\nAll dev rows of the cell, binned:\n", bins.round(3).to_string(index=False))
    print(f"\nWrote results/errors_{tag}.csv and results/error_bins_{tag}.csv")


if __name__ == "__main__":
    main()
