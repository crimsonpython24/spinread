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
    SPIN_TIMES_K_COLUMNS,
    TOO_FEW_FRAMES_COLUMN,
    WINDOWS,
    load_cell,
)
from spinread.data import TRAIN
from spinread.metrics import r_squared, score
from spinread.physics import MAGNUS_COEFFICIENT, SPIN_COMPONENT_NAMES
from spinread.spindoe import (
    CONSISTENT_ID_RANGES,
    HIGHEST_SPIN_THE_SIMULATOR_DRAWS,
    features_of_one_track,
    load_spindoe,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_models import fit_one_over_k

DATA_DIR = "data/spindoe"  # index.csv and NNN.csv
LEVEL = "L2"
HOW_MANY_FOLDS, FOLD_SEED = 5, 0
HOW_MANY_RESAMPLES, BOOTSTRAP_SEED = 2000, 0
BLOCK_LENGTH = 20  # consecutive track ids per row of the agreement table
BLUE, GRAY = "#4472C4", "#A5A5A5"
pd.set_option("display.width", 250)


def models_trained_on_the_simulator(window):
    # both methods refitted on the train rows of the simulated cell, ridge with the lambda
    # that run_models.py chose on dev
    cell = load_cell(window, LEVEL)
    features = cell["flight_features"].astype(np.float64)
    labels = cell["spin_label"].astype(np.float64)
    is_train = cell["split_code"] == TRAIN
    chosen = pd.read_csv("results/ridge_dev.csv").set_index(["window", "level"])
    scaler = StandardScaler().fit(features[is_train])
    ridges = [
        Ridge(alpha=float(chosen.loc[(window, LEVEL), f"lambda_{name}"])).fit(
            scaler.transform(features[is_train]), labels[is_train, c]
        )
        for c, name in enumerate(SPIN_COMPONENT_NAMES)
    ]
    has_u = is_train & (features[:, TOO_FEW_FRAMES_COLUMN] == 0)
    one_over_k = fit_one_over_k(features[has_u][:, SPIN_TIMES_K_COLUMNS], labels[has_u])
    return dict(
        baseline=lambda real: one_over_k * real[:, SPIN_TIMES_K_COLUMNS] / (2 * np.pi),
        ridge=lambda real: np.column_stack(
            [ridge.predict(scaler.transform(real)) for ridge in ridges]
        ),
    )


def cross_validated_scale(predictions, labels):
    # one output scale, fitted by least squares on four folds and applied to the fifth, so
    # no track is scored with its own scale; returns the calibrated predictions and the
    # mean scale
    fold_of = np.random.default_rng(FOLD_SEED).permutation(len(labels)) % HOW_MANY_FOLDS
    calibrated, scales = predictions.copy(), []
    for fold in range(HOW_MANY_FOLDS):
        fit_on = fold_of != fold
        scales.append(
            np.sum(predictions[fit_on] * labels[fit_on])
            / np.sum(predictions[fit_on] ** 2)
        )
        calibrated[fold_of == fold] *= scales[-1]
    return calibrated, float(np.mean(scales))


def score_with_intervals(labels, predictions):
    scores = score(labels, predictions)
    draws = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = pd.DataFrame(
        score(labels[chosen], predictions[chosen])
        for chosen in draws.integers(0, len(labels), (HOW_MANY_RESAMPLES, len(labels)))
    )
    for column in resampled.columns:
        scores[f"{column}_lo"], scores[f"{column}_hi"] = np.percentile(
            resampled[column], [2.5, 97.5]
        )
    return scores


def draw_scatter(rows, path):
    # measured against the calibrated ridge prediction from the whole flight (W2)
    figure, panels = plt.subplots(1, 2, figsize=(5.6, 2.9))
    kept = rows.in_range & rows.clean
    consistent = rows.in_consistent_block
    for panel, name in zip(panels, SPIN_COMPONENT_NAMES):
        measured, predicted = (
            rows.loc[kept, f"true_{name}"],
            rows.loc[kept, f"ridge_calibrated_W2_{name}"],
        )
        limit = 1.05 * max(abs(measured).max(), abs(predicted).max())
        panel.plot([-limit, limit], [-limit, limit], color=GRAY, lw=1)
        panel.scatter(
            measured[~consistent],
            predicted[~consistent],
            s=10,
            color=GRAY,
            label="other tracks",
        )
        panel.scatter(
            measured[consistent],
            predicted[consistent],
            s=10,
            color=BLUE,
            label="tracks 0-59, 100-219",
        )
        panel.set_xlim(-limit, limit)
        panel.set_ylim(-limit, limit)
        panel.set_xlabel(
            f"measured {'topspin/backspin (y~)' if name == 'top' else 'sidespin (z~)'} (rev/s)"
        )
        panel.set_title(
            f"R² {r_squared(measured, predicted):.2f} (blocks {r_squared(measured[consistent], predicted[consistent]):.2f})",
            fontsize=9,
        )
        panel.grid(alpha=0.3)
    panels[0].set_ylabel("predicted (rev/s)")
    panels[0].legend(fontsize=7, frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    tracks, spins = load_spindoe(DATA_DIR)
    rows, features, labels = [], {window: [] for window in WINDOWS}, []
    for track_number, track in tracks.items():
        per_window, label_frame, facts = features_of_one_track(track)
        spin_world = np.asarray(spins[track_number])
        labels.append((label_frame @ spin_world / (2 * np.pi))[1:])
        for window in WINDOWS:
            features[window].append(per_window[window])
        rows.append(
            dict(
                track=track_number,
                spin_magnitude=float(np.linalg.norm(spin_world)) / (2 * np.pi),
                in_range=bool(
                    np.linalg.norm(spin_world) <= HIGHEST_SPIN_THE_SIMULATOR_DRAWS
                ),
                in_consistent_block=any(
                    first <= track_number < one_past_last
                    for first, one_past_last in CONSISTENT_ID_RANGES
                ),
                **facts,
            )
        )
    rows, labels = pd.DataFrame(rows), np.array(labels)
    features = {window: np.array(features[window]) for window in WINDOWS}
    for c, name in enumerate(SPIN_COMPONENT_NAMES):
        rows[f"true_{name}"] = labels[:, c]
    subsets = {
        "scored": (rows.in_range & rows.clean).values,
        "consistent_blocks": (
            rows.in_range & rows.clean & rows.in_consistent_block
        ).values,
        "faster_than_simulator": (~rows.in_range & rows.clean).values,
    }
    print(
        f"SpinDOE: {len(rows)} tracks; {int((~rows.clean).sum())} dropped because the flight is not "
        f"one arc, {int((~rows.in_range).sum())} set aside because the spin exceeds 500 rad/s; "
        f"{int(subsets['scored'].sum())} scored. Label std (rev/s): "
        + ", ".join(
            f"{name} {labels[subsets['scored'], c].std():.0f}"
            for c, name in enumerate(SPIN_COMPONENT_NAMES)
        )
    )

    score_rows = []
    for window in WINDOWS:
        for method, predict in models_trained_on_the_simulator(window).items():
            predictions = predict(features[window])
            for c, name in enumerate(SPIN_COMPONENT_NAMES):
                rows[f"{method}_{window}_{name}"] = predictions[:, c]
            for subset, kept in subsets.items():
                if kept.sum() < 2 * HOW_MANY_FOLDS:
                    continue
                score_rows.append(
                    dict(
                        window=window,
                        subset=subset,
                        method=method,
                        calibration="none",
                        magnus_scale=np.nan,
                    )
                    | score_with_intervals(labels[kept], predictions[kept])
                )
                if subset == "faster_than_simulator":
                    continue
                calibrated, scale = cross_validated_scale(
                    predictions[kept], labels[kept]
                )
                score_rows.append(
                    dict(
                        window=window,
                        subset=subset,
                        method=method,
                        calibration="5-fold",
                        magnus_scale=scale,
                    )
                    | score_with_intervals(labels[kept], calibrated)
                )
                if subset == "scored":
                    for c, name in enumerate(SPIN_COMPONENT_NAMES):
                        rows.loc[kept, f"{method}_calibrated_{window}_{name}"] = (
                            calibrated[:, c]
                        )
    scores = pd.DataFrame(score_rows)

    # how well the flight curvature agrees with the label, block by block of track ids:
    # the share of clearly spun tracks on which the uncalibrated W2 baseline has the sign of
    # the label, and their correlation
    clear = rows[rows.in_range & rows.clean & (rows.true_top.abs() >= 20)]
    blocks = pd.DataFrame(
        dict(
            first_track=first,
            last_track=first + BLOCK_LENGTH - 1,
            how_many=len(block),
            same_sign_share=float(
                np.mean(np.sign(block.baseline_W2_top) == np.sign(block.true_top))
            ),
            correlation=float(np.corrcoef(block.baseline_W2_top, block.true_top)[0, 1]),
        )
        for first in range(0, int(rows.track.max()) + 1, BLOCK_LENGTH)
        for block in [
            clear[(clear.track >= first) & (clear.track < first + BLOCK_LENGTH)]
        ]
        if len(block) >= 3
    )

    scores.to_csv("results/spindoe_scores.csv", index=False)
    blocks.to_csv("results/spindoe_blocks.csv", index=False)
    rows.to_csv("results/spindoe_tracks.csv", index=False)
    draw_scatter(rows, "results/fig_spindoe.png")

    shown = scores[
        ["window", "subset", "method", "calibration", "how_many", "magnus_scale"]
    ].copy()
    for column in [f"r2_{name}" for name in SPIN_COMPONENT_NAMES] + ["sign_top"]:
        shown[column] = [
            f"{value:.2f} [{lo:.2f}, {hi:.2f}]"
            for value, lo, hi in zip(
                scores[column], scores[f"{column}_lo"], scores[f"{column}_hi"]
            )
        ]
    print(
        "\nSpinDOE scores with 95% bootstrap intervals ('consistent_blocks' is post hoc):"
    )
    print(shown.round(3).to_string(index=False))
    scale = scores.query(
        "window == 'W2' and subset == 'consistent_blocks' and method == 'baseline' and calibration == '5-fold'"
    ).magnus_scale.iloc[0]
    print(
        f"\nMagnus coefficient of the real ball from the W2 baseline scale on the consistent blocks: "
        f"k = {MAGNUS_COEFFICIENT:.4f} / {scale:.2f} = {MAGNUS_COEFFICIENT / scale:.4f}"
    )
    print(
        "\nFlight-versus-label agreement per block of track ids (W2 baseline, |topspin| >= 20 rev/s):"
    )
    print(blocks.round(3).to_string(index=False))
    print(
        "\nWrote results/spindoe_scores.csv, spindoe_blocks.csv, spindoe_tracks.csv and fig_spindoe.png"
    )


if __name__ == "__main__":
    main()
