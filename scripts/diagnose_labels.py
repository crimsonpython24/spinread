import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.io import load_keep, load_packed  # noqa: E402
from spinread.physics import FPS, GVEC, air_accel, polyfit, preflight_end  # noqa: E402

WINDOW = 0.1


def mismatch(t, P, w0):
    t0 = t.mean()
    c, _ = polyfit(t, P, 2, t0)
    v, a = c[1], 2 * c[2]
    air = a - GVEC
    return np.linalg.norm(air - air_accel(v, w0)) / max(np.linalg.norm(air), 1.0)


def main():
    data, keep = load_packed("cache"), load_keep("cache")
    rows = []
    for (mode, direction), d in sorted(data.items()):
        for i in np.flatnonzero(keep[(mode, direction)]):
            a, b = d["offsets"][i], d["offsets"][i + 1]
            t, P = d["times"][a:b], d["positions"][a:b]
            bounces = d["bounces"][i][: d["n_bounces"][i]]
            end = preflight_end(t, P, bounces)
            m = (t - t[0] < WINDOW) & (np.arange(len(t)) < end)
            if end / FPS < WINDOW or m.sum() < 5:
                continue
            rows.append(
                (
                    mode,
                    direction,
                    int(d["traj_id"][i]),
                    mismatch(t[m], P[m], d["w0"][i]),
                    float(np.linalg.norm(d["v0"][i])),
                )
            )
    mode = np.array([r[0] for r in rows])
    dirn = np.array([r[1] for r in rows])
    tid = np.array([r[2] for r in rows])
    mis = np.array([r[3] for r in rows])
    bad = mis > 0.2
    print(
        f"{len(rows)} trajectories tested; flagged (mismatch > 20%): {bad.sum()} ({bad.mean():.1%}); "
        f"median mismatch {np.median(mis):.3f}\n"
    )
    print("Flagged share by shot type and direction:")
    for m_ in sorted(set(mode)):
        for d_ in sorted(set(dirn)):
            s = (mode == m_) & (dirn == d_)
            if s.any():
                print(f"  {m_:<13}{d_:<15}{bad[s].mean():7.1%}  of {s.sum()}")
    print("\nFlagged share by trajectory number, rally shots only:")
    for d_ in sorted(set(dirn)):
        s = (mode == "intermediate") & (dirn == d_)
        edges = np.arange(0, tid[s].max() + 5000, 5000)
        line = []
        for lo in edges[:-1]:
            r = s & (tid >= lo) & (tid < lo + 5000)
            if r.any():
                line.append(f"{lo // 1000}k:{bad[r].mean():.0%}")
        print(f"  {d_:<15}" + " ".join(line))
    print("\n15 worst (shot type/direction/trajectory, mismatch, launch speed):")
    for i in np.argsort(-mis)[:15]:
        print(
            f"  {mode[i]}/{dirn[i]}/trajectory_{tid[i]:04d}   {mis[i]:6.2f}   {rows[i][4]:5.1f} m/s"
        )


if __name__ == "__main__":
    main()
