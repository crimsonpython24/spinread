import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.cells import (
    LEVELS,
    SPIN_TIMES_K_COLUMNS,
    TOO_FEW_FRAMES_COLUMN,
    WINDOWS,
    load_cell,
    save_dev_predictions,
)
from spinread.data import DEV, TRAIN
from spinread.metrics import r_squared, score
from spinread.physics import SPIN_COMPONENT_NAMES
from spinread import plotstyle as st

# the lambda grid, tuned on dev per cell and component; up to 1e5 because train has 91,500 rows
REGULARIZATION_STRENGTHS = np.logspace(-3, 5, 17)
READABLE_R2 = 0.8  # the proposal's threshold for "the window reveals this component"
KEY_CELLS = ["W1-L1", "W1-L3", "W2-L3", "W3-L1"]  # Table 1 of the report
LAMBDA_TABLE_ROWS = [  # Table 2 of the report: (cell, component)
    ("W1-L1", "top"),
    ("W2-L3", "top"),
    ("W2-L3", "side"),
    ("W3-L2", "gyro"),
]
LAMBDA_TABLE_STRENGTHS = [1e-3, 1.0, 1e3, 1e5]
pd.set_option("display.width", 200)


def fit_one_over_k(spin_times_k, labels):
    # spin is about (1/k) times u, so one least-squares scalar over both components gives 1/k
    labels_in_rad_per_s = labels * 2 * np.pi  # the fit works in rad/s
    return float(np.sum(spin_times_k * labels_in_rad_per_s) / np.sum(spin_times_k**2))


def baseline_on_one_cell(window, level, cell):
    spin_times_k = cell["features"][:, SPIN_TIMES_K_COLUMNS].astype(np.float64)
    labels = cell["spin_label"].astype(np.float64)
    is_train, is_dev = cell["split_code"] == TRAIN, cell["split_code"] == DEV
    # a window with too few frames has u = 0: it is left out of the fit, but it still
    # predicts 0 on dev and counts against the cell. W3 shares W2's u, so the baseline
    # gives the same numbers there and about 0 for gyrospin everywhere
    has_u = cell["features"][:, TOO_FEW_FRAMES_COLUMN] == 0
    one_over_k = fit_one_over_k(
        spin_times_k[is_train & has_u], labels[is_train & has_u]
    )
    predictions = one_over_k * spin_times_k[is_dev] / (2 * np.pi)  # back to rev/s
    save_dev_predictions("baseline", window, level, cell, predictions)
    return dict(
        window=window,
        level=level,
        k=1 / one_over_k,
        too_few_frames_dev=int((is_dev & ~has_u).sum()),
        **score(labels[is_dev], predictions),
    )


def ridge_on_one_cell(window, level, cell):
    features = cell["features"].astype(np.float64)  # W3: flight plus bounce features
    labels = cell["spin_label"].astype(np.float64)
    is_train, is_dev = cell["split_code"] == TRAIN, cell["split_code"] == DEV
    scaler = StandardScaler().fit(features[is_train])
    train_features, dev_features = (
        scaler.transform(features[is_train]),
        scaler.transform(features[is_dev]),
    )
    # one fit per lambda predicts both components; ridge treats the outputs independently,
    # so this equals two separate models, and lambda is still chosen per component
    dev_predictions = np.stack(
        [
            Ridge(alpha=strength)
            .fit(train_features, labels[is_train])
            .predict(dev_features)
            for strength in REGULARIZATION_STRENGTHS
        ]
    )  # lambda x dev row x component
    r2_per_lambda = np.array(
        [
            [
                r_squared(labels[is_dev, c], prediction[:, c])
                for c in range(len(SPIN_COMPONENT_NAMES))
            ]
            for prediction in dev_predictions
        ]
    )  # lambda x component
    best = r2_per_lambda.argmax(axis=0)
    predictions = np.column_stack(
        [dev_predictions[best[c], :, c] for c in range(len(SPIN_COMPONENT_NAMES))]
    )
    save_dev_predictions("ridge", window, level, cell, predictions)
    cell_row = dict(
        window=window,
        level=level,
        **{
            f"lambda_{name}": float(REGULARIZATION_STRENGTHS[best[c]])
            for c, name in enumerate(SPIN_COMPONENT_NAMES)
        },
        **score(labels[is_dev], predictions),
    )
    lambda_rows = [
        dict(cell=f"{window}-{level}", component=name)
        | {
            f"r2_at_{strength:g}": r2_per_lambda[
                np.argmin(
                    np.abs(np.log10(REGULARIZATION_STRENGTHS) - np.log10(strength))
                ),
                c,
            ]
            for strength in LAMBDA_TABLE_STRENGTHS
        }
        | dict(
            best_lambda=float(REGULARIZATION_STRENGTHS[best[c]]),
            r2_at_best=r2_per_lambda[best[c], c],
        )
        for c, name in enumerate(SPIN_COMPONENT_NAMES)
    ]
    return cell_row, lambda_rows


COMPONENT_TITLES = {
    "gyro": "gyrospin (x~)",
    "top": "topspin/backspin (y~)",
    "side": "sidespin (z~)",
}


def draw_heatmap(ridge_rows, path):
    # three grids, one per component (rows are the windows, columns the levels), the dev
    # R^2 written in each cell; a negative R^2 is colored as 0 but printed as it is.
    # Two rows of panels: gyrospin and topspin above, sidespin below with the level key.
    st.apply()
    colormap = "Blues"
    figure = plt.figure(figsize=(st.COLUMN_WIDTH * 5 / 7, 2.9 * 5 / 7))
    grid = figure.add_gridspec(2, 2, hspace=0.55, wspace=0.35)
    slots = [grid[0, 0], grid[0, 1], grid[1, 0]]
    long_names = {"gyro": "gyrospin", "top": "topspin", "side": "sidespin"}
    window_labels = {"W1": "100 ms", "W2": "flight", "W3": "bounce"}
    by_cell = ridge_rows.set_index(["window", "level"])
    for column_index, (slot, name) in enumerate(zip(slots, SPIN_COMPONENT_NAMES)):
        panel = figure.add_subplot(slot)
        values = np.array(
            [
                [by_cell.loc[(window, level), f"r2_{name}"] for level in LEVELS]
                for window in WINDOWS
            ]
        )
        clipped = np.clip(values, 0, 1)
        panel.imshow(clipped, vmin=0, vmax=1, cmap=colormap, aspect="auto")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                panel.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=st.FONT - 1,
                    color="white" if clipped[row, column] > 0.6 else "black",
                )
        panel.set_xticks(range(len(LEVELS)), list(LEVELS))
        panel.set_yticks(
            range(len(WINDOWS)),
            [f"{w} {window_labels[w]}" for w in WINDOWS] if column_index != 1 else [],
        )
        panel.set_title(f"{long_names[name]} $\\omega_{{\\tilde {name_axis(name)}}}$")
        panel.tick_params(length=0)
        for spine in panel.spines.values():
            spine.set_visible(False)
    key = figure.add_subplot(grid[1, 1])
    key.axis("off")
    key.text(0, 1.0, "ridge dev $R^2$", fontsize=st.FONT, va="top")
    key.text(0, 0.74, "levels:", fontsize=st.FONT, va="top")
    key.text(
        0,
        0.5,
        "L1 500 Hz, clean\nL2 145 Hz, 1 mm\nL3 50 Hz, 5 mm",
        fontsize=st.FONT - 1,
        va="top",
        linespacing=1.6,
    )
    st.finish(
        figure, path, left=0.3, right=0.98, top=0.9, bottom=0.1, hspace=0.65, wspace=0.3
    )
    plt.close(figure)


def name_axis(name):
    return {"gyro": "x", "top": "y", "side": "z"}[name]


def main():
    os.makedirs("results", exist_ok=True)
    baseline_rows, ridge_rows, lambda_rows = [], [], []
    for window in WINDOWS:
        for level in LEVELS:
            cell = load_cell(window, level)
            baseline_rows.append(baseline_on_one_cell(window, level, cell))
            ridge_row, cell_lambda_rows = ridge_on_one_cell(window, level, cell)
            ridge_rows.append(ridge_row)
            lambda_rows += cell_lambda_rows
    baseline, ridge = pd.DataFrame(baseline_rows), pd.DataFrame(ridge_rows)
    baseline.to_csv("results/baseline_dev.csv", index=False)
    ridge.to_csv("results/ridge_dev.csv", index=False)

    score_columns = [f"r2_{name}" for name in SPIN_COMPONENT_NAMES] + ["sign_top"]
    per_method = {
        name: rows.assign(cell=rows.window + "-" + rows.level).set_index("cell")
        for name, rows in [("baseline", baseline), ("ridge", ridge)]
    }
    main_table = pd.DataFrame(
        dict(
            method=method,
            cell=cell,
            **rows.loc[
                cell, score_columns + [f"rmse_{n}" for n in SPIN_COMPONENT_NAMES]
            ],
        )
        for cell in KEY_CELLS
        for method, rows in per_method.items()
    )
    main_table.to_csv("results/main_table.csv", index=False)
    lambda_table = (
        pd.DataFrame(lambda_rows)
        .set_index(["cell", "component"])
        .loc[LAMBDA_TABLE_ROWS]
        .reset_index()
    )
    lambda_table.to_csv("results/ridge_lambda_table.csv", index=False)

    # reading horizon: per level and component, the shortest window whose better method
    # reaches R^2 >= 0.8 on dev, or "none"
    best = pd.concat(per_method.values()).groupby("cell").max()
    horizon = pd.DataFrame(
        dict(
            level=level,
            **{
                name: next(
                    (
                        w
                        for w in WINDOWS
                        if best.loc[f"{w}-{level}", f"r2_{name}"] >= READABLE_R2
                    ),
                    "none",
                )
                for name in SPIN_COMPONENT_NAMES
            },
        )
        for level in LEVELS
    )
    horizon.to_csv("results/reading_horizon.csv", index=False)
    draw_heatmap(ridge, "results/fig_ridge_heatmap.png")

    print("Baseline, dev:\n", baseline.round(3).to_string(index=False))
    print("\nRidge, dev:\n", ridge.round(3).to_string(index=False))
    print("\nKey cells:\n", main_table.round(3).to_string(index=False))
    print("\nLambda grid:\n", lambda_table.round(3).to_string(index=False))
    print("\nReading horizon:\n", horizon.to_string(index=False))
    print(
        "\nWrote results/baseline_dev.csv, ridge_dev.csv, main_table.csv, ridge_lambda_table.csv,"
        " reading_horizon.csv, fig_ridge_heatmap.png and cache/prediction_<method>_<cell>.npz"
    )


if __name__ == "__main__":
    main()
