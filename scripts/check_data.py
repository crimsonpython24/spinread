import csv
import os
import sys


import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # so `spinread` imports
from spinread.io import load_packed  # noqa: E402
from spinread.physics import FPS, GVEC, air_accel, preflight_end  # noqa: E402

MAX_SPIN = 500.0  # rad/s
EXPECTED_BOUNCES = {
    "intermediate": 1,
    "first_good": 2,
    "first_short": 2,
    "first_long": 1,
    "final_win": 2,
    "final_lose": 0,
}
CHECKS = ["nan", "dt", "spin", "bounces", "n_bounces", "collision"]
COLLISION_THRESHOLD = 50.0  # m/s^2


def max_air_mismatch(t, P, w0, bounces):
    """Largest gap between measured and modelled acceleration before the first bounce."""
    end = preflight_end(t, P, bounces)
    Q = P[:end]
    if len(Q) < 3:
        return 0.0
    dt = t[1] - t[0]
    v = (Q[2:] - Q[:-2]) / (2 * dt)
    a = (Q[2:] - 2 * Q[1:-1] + Q[:-2]) / dt**2 - GVEC
    return float(np.linalg.norm(a - air_accel(v, w0), axis=1).max())


def check_folder(d, mode):
    n = len(d["traj_id"])
    off = d["offsets"]
    fail = {c: np.zeros(n, bool) for c in CHECKS}
    short_flight = np.zeros(n, bool)
    for i in range(n):
        t = d["times"][off[i] : off[i + 1]]
        P = d["positions"][off[i] : off[i + 1]]
        nb = d["n_bounces"][i]
        b = d["bounces"][i][: min(nb, 2)]
        w0 = d["w0"][i]
        fail["nan"][i] = (
            np.isnan(P).any()
            or np.isnan(t).any()
            or np.isnan(d["v0"][i]).any()
            or np.isnan(w0).any()
        )
        fail["dt"][i] = len(t) < 2 or not np.allclose(np.diff(t), 1 / FPS, atol=1e-6)
        fail["spin"][i] = np.linalg.norm(w0) > MAX_SPIN + 1e-6
        fail["bounces"][i] = len(b) > 0 and (
            np.any(np.diff(b) < 0) or b[0] < t[0] or b[-1] > t[-1]
        )
        fail["n_bounces"][i] = nb != EXPECTED_BOUNCES[mode]
        if not (fail["nan"][i] or fail["dt"][i] or fail["bounces"][i]):
            fail["collision"][i] = max_air_mismatch(t, P, w0, b) > COLLISION_THRESHOLD
        short_flight[i] = nb > 0 and len(b) > 0 and b[0] - t[0] < 0.1
    keep = ~np.any([fail[c] for c in CHECKS], axis=0)
    return keep, fail, short_flight


def main():
    data = load_packed("cache")
    if not data:
        raise SystemExit(
            "No cache/synthetic_*.npz files found. Run scripts/pack_data.py first."
        )
    os.makedirs("results", exist_ok=True)
    keeps, rows = {}, []
    for (mode, direction), d in sorted(data.items()):
        keep, fail, short_flight = check_folder(d, mode)
        keeps[f"{mode}/{direction}"] = keep
        rows.append(
            dict(
                mode=mode,
                direction=direction,
                total=len(keep),
                **{c: int(fail[c].sum()) for c in CHECKS},
                kept=int(keep.sum()),
                short_flight_kept=int((short_flight & keep).sum()),
            )
        )
    np.savez("cache/keep.npz", **keeps)
    with open("results/data_cleaning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["total"] + CHECKS + ["kept", "short_flight_kept"]
    print(
        f'{"mode":<13}{"direction":<15}'
        + "".join(f"{c:>{max(11, len(c) + 2)}}" for c in cols)
    )
    for r in rows:
        print(
            f'{r["mode"]:<13}{r["direction"]:<15}'
            + "".join(f"{r[c]:>{max(11, len(c) + 2)}}" for c in cols)
        )
    tot = {c: sum(r[c] for r in rows) for c in cols}
    print(f'{"TOTAL":<28}' + "".join(f"{tot[c]:>{max(11, len(c) + 2)}}" for c in cols))
    print(
        "\nA trajectory can fail several checks, so the check columns can add up to more than total - kept."
    )
    print("Wrote cache/keep.npz and results/data_cleaning.csv")


if __name__ == "__main__":
    main()
