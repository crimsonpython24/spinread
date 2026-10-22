import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.cells import (  # noqa: E402
    BOUNCE_NAMES,
    FLIGHT_NAMES,
    LAUNCH_NAMES,
    LEVELS,
    WINDOWS,
    process_trajectory,
    rebound_index,
)
from spinread.io import get_trajectory, load_packed  # noqa: E402
from spinread.metrics import r2  # noqa: E402
from spinread.physics import (
    K_MAGNUS,
    ball_frame,
    contact_index,
    spin_label,
)  # noqa: E402
from spinread.splits import load_splits  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cells import seed_for  # noqa: E402

N = 20
IU = [FLIGHT_NAMES.index(f"u_{a}") for a in "xyz"]
I_FLIGHT = FLIGHT_NAMES.index("duration")
I_REST = BOUNCE_NAMES.index("restitution")
I_HAS = BOUNCE_NAMES.index("has_bounce")
I_FEW = FLIGHT_NAMES.index("too_few_frames")
I_FEW_POST = BOUNCE_NAMES.index("too_few_post")


def raw_restitution(tr):
    t, P, b = tr["t"], tr["P"], tr["bounces"]
    kc = contact_index(t, P, b)
    kr = rebound_index(P, kc)
    dt = t[1] - t[0]
    vz_pre = (P[kc - 2, 2] - P[kc - 4, 2]) / (2 * dt)
    vz_post = (P[kr + 2, 2] - P[kr, 2]) / (2 * dt) if kr + 2 < len(P) else float("nan")
    r = abs(vz_post / vz_pre) if abs(vz_pre) > 1e-6 else float("nan")
    info = (
        f"contact at x={P[kc, 0]:.2f} y={P[kc, 1]:.2f} z={P[kc, 2]:.3f} after {t[kc] - t[0]:.3f} s, "
        f"{kr - kc} samples on the table, vz {vz_pre:.2f} -> {vz_post:.2f} m/s, "
        f"speed {np.linalg.norm(tr['v0']):.1f} m/s, spin {np.linalg.norm(tr['w0']) / (2 * np.pi):.0f} rev/s, "
        f"{len(b)} bounces"
    )
    return r, info


def first_kept(data, splits, mode, direction, n):
    d = data[(mode, direction)]
    idx = np.flatnonzero(splits[(mode, direction)] >= 0)[:n]
    return [(mode, direction, get_trajectory(d, i)) for i in idx]


def run(trajs, level):
    out = []
    for mode, direction, tr in trajs:
        rng = np.random.default_rng(seed_for(level, mode, direction, tr["traj_id"]))
        feats, dbg = process_trajectory(
            tr["t"], tr["P"], tr["bounces"], level, rng, debug=True
        )
        out.append((feats, dbg, spin_label(tr["w0"], tr["v0"]), ball_frame(tr["v0"])))
    return out


def check(trajs, label, bounce_expected):
    ok = True
    for level in LEVELS:
        res = run(trajs, level)
        res2 = run(trajs, level)
        for (f1, _, _, _), (f2, _, _, _) in zip(res, res2):
            for w in WINDOWS:
                for a, b in zip(f1[w], f2[w]):
                    if a is not None and not np.array_equal(a, b):
                        print(f"  FAIL {label} {level} {w}: not reproducible")
                        ok = False
        for feats, dbg, y, F in res:
            for w in WINDOWS:
                xf, xb, xl = feats[w]
                assert xf.shape == (len(FLIGHT_NAMES),) and xl.shape == (
                    len(LAUNCH_NAMES),
                )
                assert (xb is None) == (w != "W3") and (
                    xb is None or xb.shape == (len(BOUNCE_NAMES),)
                )
                for x in (xf, xb, xl):
                    if x is not None and not np.all(np.isfinite(x)):
                        print(f"  FAIL {label} {level} {w}: non-finite feature")
                        ok = False
        # the baseline on W1 (clean) and W2 (all levels), only for shots with enough frames
        for w in ["W1", "W2"]:
            rows = [
                (f[w][0], y)
                for f, _, y, _ in res
                if f[w][0][I_FEW] == 0 and f[w][0][I_FLIGHT] >= 0.06
            ]
            if len(rows) >= 5:
                U = np.array([r[0][IU] for r in rows]) / K_MAGNUS / (2 * np.pi)
                Y = np.array([r[1] for r in rows])
                print(
                    f"  {label} {level} {w}: n={len(rows):>2}  topspin R^2 {r2(Y[:, 1], U[:, 1]):6.3f}"
                    f"  sidespin R^2 {r2(Y[:, 2], U[:, 2]):6.3f}  (u/k in the estimated frame)"
                )
                if level == "L1" and w == "W1" and r2(Y[:, 1], U[:, 1]) < 0.9:
                    print(f"  FAIL {label}: clean W1 topspin R^2 should be about 0.99")
                    ok = False
        xb = np.array([f["W3"][1] for f, _, _, _ in res])
        has = xb[:, I_HAS]
        few = np.array([f["W1"][0][I_FEW] for f, _, _, _ in res])
        seen = (has == 1) & (xb[:, I_FEW_POST] == 0)
        print(
            f"  {label} {level} W3: has_bounce {int(has.sum())}/{len(has)}, bounce seen in frames {int(seen.sum())}, "
            f"restitution median {np.median(xb[seen, I_REST]) if seen.any() else float('nan'):.2f}, "
            f"W1 too_few_frames {int(few.sum())}/{len(few)}"
        )
        if bounce_expected and not has.all():
            print(
                f"  FAIL {label} {level}: every rally shot should have has_bounce = 1"
            )
            ok = False
        if not bounce_expected and has.any():
            print(f"  FAIL {label} {level}: final_lose must have has_bounce = 0")
            ok = False
        if bounce_expected and level == "L1":
            # An odd restitution is a bug only if the fit disagrees with the raw 500 Hz samples.
            for (feats, dbg, y, F), (mode, direction, tr) in zip(res, trajs):
                xb = feats["W3"][1]
                r_fit = xb[I_REST]
                if xb[I_FEW_POST] or 0.5 < r_fit < 1.1:
                    continue
                r_raw, info = raw_restitution(tr)
                agree = abs(r_fit - r_raw) < max(0.15, 0.25 * r_raw)
                print(
                    f"  {'note' if agree else 'FAIL'} {label} trajectory {tr['traj_id']}: fitted restitution {r_fit:.2f}, "
                    f"raw 500 Hz {r_raw:.2f}; {info}"
                )
                ok &= agree
    return ok


def plot(trajs, which, path):
    mode, direction, tr = trajs[which]
    t, P = tr["t"] - tr["t"][0], tr["P"]
    F = ball_frame(tr["v0"])
    along = (P - P[0]) @ F[0]  # distance along the launch direction
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, level in zip(axes.ravel(), LEVELS):
        rng = np.random.default_rng(seed_for(level, mode, direction, tr["traj_id"]))
        feats, dbg = process_trajectory(
            tr["t"], tr["P"], tr["bounces"], level, rng, debug=True
        )
        ax.plot(along, P[:, 2], color="0.75", lw=3, label="true 500 Hz path")
        obs_along = (dbg["P"] - P[0]) @ F[0]
        ax.plot(
            obs_along[dbg["in_w2"]],
            dbg["P"][dbg["in_w2"], 2],
            "o",
            ms=4,
            color="C0",
            label="frames, pre-bounce",
        )
        ax.plot(
            obs_along[dbg["in_w1"]],
            dbg["P"][dbg["in_w1"], 2],
            "o",
            ms=7,
            mfc="none",
            color="C0",
            label="frames in W1 (<100 ms)",
        )
        ax.plot(
            obs_along[dbg["in_post"]],
            dbg["P"][dbg["in_post"], 2],
            "s",
            ms=4,
            color="C3",
            label="frames, post-bounce (W3)",
        )
        fit, _ = dbg["fits"]["W2"]
        if fit is not None:
            s = np.linspace(fit["t_first"], fit["t_last"], 50)
            Q = (
                fit["p"]
                + np.outer(s - fit["tc"], fit["v"])
                + 0.5 * np.outer((s - fit["tc"]) ** 2, fit["a"])
            )
            ax.plot(
                (Q - P[0]) @ F[0],
                Q[:, 2],
                "-",
                color="C0",
                lw=1.5,
                label="fitted pre-bounce arc",
            )
        xb = feats["W3"][1]
        if xb[I_HAS] and xb[I_FEW_POST] == 0:
            ts_post = dbg["ts"][dbg["in_post"]]
            pre, Fe = dbg["fits"]["pre"]
            t_imp = pre["t_last"] + xb[BOUNCE_NAMES.index("impact_gap")]
            v_post = (
                Fe.T @ xb[3:6]
            )  # back from the estimated ball frame to the world frame
            p_imp = (
                pre["p"]
                + (t_imp - pre["tc"]) * pre["v"]
                + 0.5 * (t_imp - pre["tc"]) ** 2 * pre["a"]
            )
            s = np.linspace(0, ts_post[-1] - t_imp, 30)
            Q = p_imp + np.outer(s, v_post) - 0.5 * 9.81 * np.outer(s**2, [0, 0, 1])
            ax.plot(
                (Q - P[0]) @ F[0],
                Q[:, 2],
                "-",
                color="C3",
                lw=1.5,
                label="fitted post-bounce path",
            )
        xf = feats["W2"][0]
        ax.set_title(
            f'{level}: {int(feats["W1"][0][FLIGHT_NAMES.index("n_frames")])} frames in W1, '
            f'{int(xf[FLIGHT_NAMES.index("n_frames")])} in W2, fit RMS {xf[FLIGHT_NAMES.index("resid_rms")] * 1000:.1f} mm'
        )
        ax.axhline(0.78, color="k", lw=0.5, ls=":")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("distance along launch direction (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("height (m)")
    y = spin_label(tr["w0"], tr["v0"])
    fig.suptitle(
        f"{mode}/{direction} trajectory {tr['traj_id']}: true spin gyro {y[0]:.0f}, top {y[1]:.0f}, side {y[2]:.0f} rev/s"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plot-index",
        type=int,
        default=0,
        help="which of the 20 rally trajectories to draw",
    )
    args = ap.parse_args()
    data, splits = load_packed("cache"), load_splits("cache")
    rally = first_kept(data, splits, "intermediate", "left_to_right", N)
    lose = first_kept(data, splits, "final_lose", "left_to_right", N)
    ok = check(rally, "intermediate", True)
    ok &= check(lose, "final_lose", False)
    os.makedirs("results", exist_ok=True)
    plot(rally, args.plot_index, "results/fig_windows.png")
    print(
        "\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED (see FAIL lines above)"
    )


if __name__ == "__main__":
    main()
