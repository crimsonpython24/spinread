import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.cells import (  # noqa: E402
    BOUNCE_NAMES,
    DIRECTIONS,
    FLIGHT_NAMES,
    LAUNCH_NAMES,
    LEVELS,
    MODES,
    WINDOWS,
    cell_path,
    process_trajectory,
)
from spinread.physics import spin_label  # noqa: E402
from spinread.splits import load_splits  # noqa: E402

I_TOO_FEW = FLIGHT_NAMES.index("too_few_frames")
I_DURATION = FLIGHT_NAMES.index("duration")
I_HAS_BOUNCE = BOUNCE_NAMES.index("has_bounce")
I_TOO_FEW_POST = BOUNCE_NAMES.index("too_few_post")


def seed_for(level, mode, direction, traj_id):
    return [
        list(LEVELS).index(level),
        MODES.index(mode),
        DIRECTIONS.index(direction),
        int(traj_id),
    ]


def process_folder(cache_dir, mode, direction, level, limit):
    z = np.load(os.path.join(cache_dir, f"synthetic_{mode}_{direction}.npz"))
    d = {k: z[k] for k in z.files}
    split = load_splits(cache_dir)[(mode, direction)]
    idx = np.flatnonzero(split >= 0)
    if limit:
        idx = idx[:limit]
    out = {w: dict(flight=[], bounce=[], launch=[]) for w in WINDOWS}
    y, speed, spin = [], [], []
    for i in idx:
        a, b = d["offsets"][i], d["offsets"][i + 1]
        t, P = d["times"][a:b], d["positions"][a:b]
        bounces = d["bounces"][i][: d["n_bounces"][i]]
        rng = np.random.default_rng(seed_for(level, mode, direction, d["traj_id"][i]))
        feats = process_trajectory(t, P, bounces, level, rng)
        for w in WINDOWS:
            xf, xb, xl = feats[w]
            out[w]["flight"].append(xf)
            out[w]["launch"].append(xl)
            if xb is not None:
                out[w]["bounce"].append(xb)
        y.append(spin_label(d["w0"][i], d["v0"][i]))
        speed.append(np.linalg.norm(d["v0"][i]))
        spin.append(np.linalg.norm(d["w0"][i]) / (2 * np.pi))
    return dict(
        mode=mode,
        direction=direction,
        idx=idx,
        split=split[idx],
        traj_id=d["traj_id"][idx],
        y=np.array(y),
        speed=np.array(speed),
        spin=np.array(spin),
        windows={w: {k: np.array(v) for k, v in out[w].items()} for w in WINDOWS},
    )


def write_cell(window, level, parts, cache_dir):
    cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
    X_flight = np.concatenate([p["windows"][window]["flight"] for p in parts])
    X_launch = np.concatenate([p["windows"][window]["launch"] for p in parts])
    if window == "W3":
        X_bounce = np.concatenate([p["windows"][window]["bounce"] for p in parts])
    else:
        X_bounce = np.zeros((len(X_flight), 0))
    np.savez(
        cell_path(window, level, cache_dir),
        X_flight=X_flight.astype(np.float32),
        X_bounce=X_bounce.astype(np.float32),
        X_launch=X_launch.astype(np.float32),
        y=cat("y").astype(np.float32),
        split=cat("split").astype(np.int8),
        mode=np.concatenate([[p["mode"]] * len(p["idx"]) for p in parts]),
        direction=np.concatenate([[p["direction"]] * len(p["idx"]) for p in parts]),
        traj_id=cat("traj_id").astype(np.int64),
        index=cat("idx").astype(np.int64),
        speed=cat("speed").astype(np.float32),
        duration=X_flight[:, I_DURATION].astype(np.float32),
        spin_mag=cat("spin").astype(np.float32),
        flight_names=np.array(FLIGHT_NAMES),
        bounce_names=np.array(BOUNCE_NAMES if window == "W3" else []),
        launch_names=np.array(LAUNCH_NAMES),
    )
    row = dict(
        window=window,
        level=level,
        n=len(X_flight),
        too_few_frames=int(X_flight[:, I_TOO_FEW].sum()),
        median_frames=float(np.median(X_flight[:, FLIGHT_NAMES.index("n_frames")])),
        has_bounce=int(X_bounce[:, I_HAS_BOUNCE].sum()) if window == "W3" else "",
        too_few_post=int(X_bounce[:, I_TOO_FEW_POST].sum()) if window == "W3" else "",
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument(
        "--levels",
        nargs="+",
        default=list(LEVELS),
        help="e.g. --levels L1 to time one level first",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N kept trajectories per folder (testing)",
    )
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    rows, jobs = [], [(m, d) for m in MODES for d in DIRECTIONS]
    for level in args.levels:
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            parts = list(
                ex.map(
                    process_folder,
                    *zip(*[(args.cache, m, d, level, args.limit) for m, d in jobs]),
                )
            )
        for window in WINDOWS:
            rows.append(write_cell(window, level, parts, args.cache))
        n = sum(len(p["idx"]) for p in parts)
        print(f"{level}: {n} trajectories, 3 cells written in {time.time() - t0:.0f} s")

    path = "results/cell_summary.csv"
    existing = []
    if os.path.exists(path) and set(args.levels) != set(LEVELS):
        with open(path) as f:
            existing = [r for r in csv.DictReader(f) if r["level"] not in args.levels]
    rows = existing + [{k: str(v) for k, v in r.items()} for r in rows]
    rows.sort(key=lambda r: (r["window"], r["level"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(
        f'\n{"cell":<8}{"n":>8}{"too_few_frames":>16}{"median_frames":>15}{"has_bounce":>12}{"too_few_post":>14}'
    )
    for r in rows:
        print(
            f'{r["window"]}_{r["level"]:<5}{r["n"]:>8}{r["too_few_frames"]:>16}{r["median_frames"]:>15}'
            f'{r["has_bounce"]:>12}{r["too_few_post"]:>14}'
        )
    print(f"\nWrote cache/cell_<W>_<L>.npz and {path}")


if __name__ == "__main__":
    main()
