import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.io import load_packed  # noqa: E402
from spinread.metrics import r2, rmse, sign_accuracy  # noqa: E402
from spinread.physics import (
    ball_frame,
    physics_u,
    preflight_end,
    spin_label,
)  # noqa: E402
from spinread.splits import load_splits  # noqa: E402

WINDOW = 0.1  # s: the first 100 ms (window W1)
MODES = ["intermediate"]


def collect(data, splits):
    U, F, Y, S = [], [], [], []
    for (mode, direction), d in sorted(data.items()):
        if mode not in MODES:
            continue
        split = splits[(mode, direction)]
        for i in np.flatnonzero(split >= 0):
            a, b = d["offsets"][i], d["offsets"][i + 1]
            t, P = d["times"][a:b], d["positions"][a:b]
            bounces = d["bounces"][i][: d["n_bounces"][i]]
            end = preflight_end(t, P, bounces)
            m = (t - t[0] < WINDOW) & (np.arange(len(t)) < end)
            if m.sum() < 5:
                continue
            u, _, _ = physics_u(t[m], P[m], "center")
            U.append(u)  # world frame, = k * omega_perp
            F.append(ball_frame(d["v0"][i]))
            Y.append(spin_label(d["w0"][i], d["v0"][i]))  # rev/s, ball frame
            S.append(split[i])
    return np.array(U), np.array(F), np.array(Y), np.array(S)


def main():
    U, F, Y, S = collect(load_packed("cache"), load_splits("cache"))
    tr, dev = S == 0, S == 1
    Ub = np.einsum("nij,nj->ni", F, U)  # u in the ball frame
    Yr = Y * 2 * np.pi  # back to rad/s for the fit
    c = np.sum(Ub[tr] * Yr[tr]) / np.sum(Ub[tr] ** 2)  # omega ~ c * u, so k = 1/c
    pred = c * Ub / (2 * np.pi)  # rev/s
    print(
        f"rally shots: {tr.sum()} train, {dev.sum()} dev;  fitted k = {1 / c:.4f} (proposal check: 0.0076)"
    )
    for j, name in enumerate(["gyro", "top", "side"]):
        print(
            f"  {name:<5} R^2 {r2(Y[dev, j], pred[dev, j]):6.3f}   RMSE {rmse(Y[dev, j], pred[dev, j]):5.2f} rev/s"
        )
    print(f"  topspin sign accuracy {sign_accuracy(Y[dev, 1], pred[dev, 1]):.3f}")
    print(
        "Expected: top about 0.99, side about 0.97, gyro near 0 (the baseline cannot see spin along v)."
    )


if __name__ == "__main__":
    main()
