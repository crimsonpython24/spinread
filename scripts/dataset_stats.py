import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.io import load_packed  # noqa: E402
from spinread.physics import spin_label  # noqa: E402
from spinread.splits import load_splits  # noqa: E402

SPLIT_NAMES = {0: "train", 1: "dev", 2: "test"}
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def build_table():
    data, splits = load_packed("cache"), load_splits("cache")
    rows = []
    for (mode, direction), d in sorted(data.items()):
        split = splits[(mode, direction)]
        for i in np.flatnonzero(split >= 0):
            a, b = d["offsets"][i], d["offsets"][i + 1]
            t = d["times"][a:b]
            gyro, top, side = spin_label(d["w0"][i], d["v0"][i])
            nb = d["n_bounces"][i]
            rows.append(
                dict(
                    mode=mode,
                    direction=direction,
                    i=int(i),
                    traj_id=int(d["traj_id"][i]),
                    split=SPLIT_NAMES[int(split[i])],
                    speed=float(np.linalg.norm(d["v0"][i])),
                    spin=float(np.linalg.norm(d["w0"][i]) / (2 * np.pi)),
                    gyro=gyro,
                    top=top,
                    side=side,
                    flight_to_bounce=float(d["bounces"][i][0] - t[0]) if nb else np.nan,
                    duration=float(t[-1] - t[0]),
                    n_bounces=int(nb),
                )
            )
    return pd.DataFrame(rows)


def main(plot=True):
    os.makedirs("results", exist_ok=True)
    df = build_table()
    df.to_csv("results/trajectories.csv", index=False)

    counts = df.pivot_table(
        index="mode", columns="split", values="i", aggfunc="count", fill_value=0
    )
    counts = counts[["train", "dev", "test"]]
    counts.loc["TOTAL"] = counts.sum()
    print("Trajectories per shot type and split\n", counts, "\n")

    stats = df.groupby("mode").agg(
        n=("i", "count"),
        speed_median=("speed", "median"),
        speed_min=("speed", "min"),
        speed_max=("speed", "max"),
        spin_median=("spin", "median"),
        spin_max=("spin", "max"),
        topspin_share=("top", lambda x: float((x > 0).mean())),
        flight_median=("flight_to_bounce", "median"),
        flight_max=("flight_to_bounce", "max"),
    )
    print(
        "Per shot type (speed m/s, spin rev/s, flight to first bounce in s)\n",
        stats.round(3),
        "\n",
    )
    pd.concat([counts.add_prefix("n_"), stats], axis=1).to_csv(
        "results/dataset_stats.csv"
    )

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
        ax[0].hist(df["spin"], bins=50)
        ax[0].set_xlabel("spin magnitude (rev/s)")
        ax[1].hist(df["speed"], bins=50)
        ax[1].set_xlabel("launch speed (m/s)")
        ax[2].hist(df["flight_to_bounce"].dropna(), bins=50)
        ax[2].set_xlabel("flight until first bounce (s)")
        ax[0].set_ylabel("trajectories")
        fig.tight_layout()
        fig.savefig("results/fig_dataset.png", dpi=200)
    print(
        "Wrote results/trajectories.csv, results/dataset_stats.csv"
        + (", results/fig_dataset.png" if plot else "")
    )


if __name__ == "__main__":
    main(plot="--no-plot" not in sys.argv)
