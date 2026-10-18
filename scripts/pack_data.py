import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

MODES = [
    "intermediate",
    "first_good",
    "first_short",
    "first_long",
    "final_win",
    "final_lose",
]
DIRECTIONS = ["left_to_right", "right_to_left"]
MAX_BOUNCES = 2


def list_trajectories(folder):
    """trajectory_XXXX folders sorted by number (names go past 9999, so plain sorting is wrong)."""
    names = [n for n in os.listdir(folder) if n.startswith("trajectory_")]
    return sorted(names, key=lambda n: int(n.split("_")[1]))


def pack_folder(src, out_dir, mode, direction, limit=None):
    """Pack one mode/direction folder into a single npz. Each trajectory's first sample is dropped:
    it is stamped 2 ms before the second, but the ball moves only about half a step between them,
    which shows up as a huge spurious acceleration in finite differences."""
    folder = os.path.join(src, mode, direction)
    names = list_trajectories(folder)
    if limit:
        names = names[:limit]
    pos, tim, lengths, v0, w0, bnc, nb, ids, skipped = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for name in names:
        d = os.path.join(folder, name)
        try:
            p = np.load(os.path.join(d, "positions.npy"))
            t = np.load(os.path.join(d, "times.npy"))
            v = np.load(os.path.join(d, "velocities.npy"))
            w = np.load(os.path.join(d, "rotations.npy"))
            b = np.atleast_1d(np.load(os.path.join(d, "bounces.npy")))
        except (OSError, ValueError) as e:
            skipped.append((name, str(e)))
            continue
        if p.ndim != 2 or p.shape[1] != 3 or len(t) != len(p) or len(p) < 2:
            skipped.append((name, f"bad shapes positions {p.shape}, times {t.shape}"))
            continue
        p, t = p[1:], t[1:]  # drop the mistimed first sample (see the docstring)
        pos.append(p)
        tim.append(t)
        lengths.append(len(p))
        v0.append(v[0])
        w0.append(w[0])
        row = np.full(MAX_BOUNCES, np.nan)
        row[: min(len(b), MAX_BOUNCES)] = b[:MAX_BOUNCES]
        bnc.append(row)
        nb.append(len(b))
        ids.append(int(name.split("_")[1]))

    out = os.path.join(out_dir, f"synthetic_{mode}_{direction}.npz")
    np.savez(
        out,
        positions=np.concatenate(pos).astype(np.float64),
        times=np.concatenate(tim).astype(np.float64),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        v0=np.array(v0),
        w0=np.array(w0),
        bounces=np.array(bnc),
        n_bounces=np.array(nb, dtype=np.int64),
        traj_id=np.array(ids, dtype=np.int64),
        mode=mode,
        direction=direction,
    )
    lengths = np.array(lengths)
    return dict(
        mode=mode,
        direction=direction,
        n=len(ids),
        skipped=skipped,
        out=out,
        len_min=int(lengths.min()),
        len_med=int(np.median(lengths)),
        len_max=int(lengths.max()),
        bounce_counts=np.bincount(nb, minlength=MAX_BOUNCES + 1).tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/syntheticdata")
    ap.add_argument("--out", default="cache")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N trajectories per folder (for testing)",
    )
    ap.add_argument("--workers", type=int, default=4, help="folders packed in parallel")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(m, d) for m in MODES for d in DIRECTIONS]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [
            ex.submit(pack_folder, args.src, args.out, m, d, args.limit)
            for m, d in jobs
        ]
        results = [f.result() for f in futures]

    print(f"\nPacked in {time.time() - t0:.0f} s\n")
    print(
        f'{"mode":<13}{"direction":<15}{"n":>7}{"skipped":>9}   {"length min/median/max":<21}   bounces 0/1/2'
    )
    for r in results:
        length = f'{r["len_min"]}/{r["len_med"]}/{r["len_max"]}'
        print(
            f'{r["mode"]:<13}{r["direction"]:<15}{r["n"]:>7}{len(r["skipped"]):>9}'
            f'   {length:<21}   {r["bounce_counts"]}'
        )
    print(f'\nTotal: {sum(r["n"] for r in results)} trajectories')
    for r in results:
        for name, why in r["skipped"]:
            print(f'skipped {r["mode"]}/{r["direction"]}/{name}: {why}')


if __name__ == "__main__":
    main()
