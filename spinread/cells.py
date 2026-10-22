import os

import numpy as np
from scipy.interpolate import CubicSpline

from spinread.physics import (
    G,
    GVEC,
    R_BALL,
    TABLE_HEIGHT,
    ball_frame,
    contact_index,
    polyfit,
    preflight_end,
)

WINDOWS = ["W1", "W2", "W3"]
LEVELS = {
    "L1": dict(step=1, noise=0.0),
    "L2": dict(rate=145.0, noise=0.001, jitter=0.00045),
    "L3": dict(step=10, noise=0.005),
    "L4": dict(step=20, noise=0.005),
}
MODES = [
    "intermediate",
    "first_good",
    "first_short",
    "first_long",
    "final_win",
    "final_lose",
]
DIRECTIONS = ["left_to_right", "right_to_left"]

W1_LENGTH = 0.1  # s
POST_LENGTH = 0.05  # s of W3 after the first contact
PRE_LOCAL = 0.1  # s before contact used for the impact-time and v_pre fit
Z_CONTACT = TABLE_HEIGHT + R_BALL  # ball centre height at contact, 0.78 m
EPS = 1e-9

AXES = [
    "x",
    "y",
    "z",
]  # x~ along the horizontal flight direction, y~ = z~ cross x~, z~ up
FLIGHT_NAMES = (
    [f"v_start_{a}" for a in AXES]  # fitted velocity at the first frame
    + [f"v_centre_{a}" for a in AXES]  # fitted velocity at the window centre
    + [f"a_{a}" for a in AXES]  # fitted acceleration with gravity removed
    + [
        f"a_perp_{a}" for a in AXES
    ]  # Magnus residual: the part of a_ng perpendicular to v
    + [f"u_{a}" for a in AXES]  # baseline output (v x a_perp)/|v|^2, about k * omega
    + ["speed", "resid_rms", "n_frames", "duration", "t_first", "too_few_frames"]
)
BOUNCE_NAMES = (
    [f"v_pre_{a}" for a in AXES]  # pre-bounce arc velocity at the estimated impact time
    + [
        f"v_post_{a}" for a in AXES
    ]  # post-bounce velocity at the impact time (gravity-fixed fit)
    + [
        f"dv_{a}" for a in AXES
    ]  # v_post - v_pre: dv_x is the along-track kick, dv_y the sideways kick
    + ["restitution", "n_post", "impact_gap", "has_bounce", "too_few_post"]
)  # has_bounce = 1 for every shot that bounces; too_few_post = 1 when < 2 frames show it (kicks left at 0)
LAUNCH_NAMES = (
    [f"p0_world_{a}" for a in AXES]  # first observed position, world frame
    + [
        f"p0_{a}" for a in AXES
    ]  # the same position rotated into the estimated ball frame
    + [f"v_start_{a}" for a in AXES]
)


def rebound_index(P, kc):
    j = kc
    while j < len(P) and P[j, 2] < Z_CONTACT + 0.001:
        j += 1
    return j


def _interp(t, P, tq):
    if len(t) >= 4:
        return CubicSpline(t, P)(tq)
    return np.stack([np.interp(tq, t, P[:, j]) for j in range(3)], axis=1)


def resample(t, P, kc, kr, tq):
    if kc >= len(t):
        return _interp(t, P, tq)
    out = np.empty((len(tq), 3))
    pre, post = tq <= t[kc], tq >= t[kr - 1]
    mid = ~pre & ~post
    out[pre] = _interp(t[: kc + 1], P[: kc + 1], tq[pre])
    if post.any():
        out[post] = _interp(t[kr - 1 :], P[kr - 1 :], tq[post])
    if mid.any():
        out[mid] = np.stack([np.interp(tq[mid], t, P[:, j]) for j in range(3)], axis=1)
    return out


def observe(t, P, kc, kr, level, rng):
    spec = LEVELS[level]
    if "rate" in spec:
        tt = t[0] + rng.uniform(0, 1 / spec["rate"]) + np.arange(len(t)) / spec["rate"]
        tt = tt[tt <= t[-1]]
        Ps = resample(t, P, kc, kr, tt)
        ts = np.round((tt + rng.normal(0, spec["jitter"], len(tt))) * 1000) / 1000
    else:
        idx = np.arange(rng.integers(0, spec["step"]), len(t), spec["step"])
        tt, Ps = t[idx], P[idx]
        ts = tt
    if spec["noise"]:
        Ps = Ps + rng.normal(0, spec["noise"], Ps.shape)
    return ts, tt, Ps


def fit_arc(ts, Ps):
    n = len(ts)
    if n < 2:
        return None
    tc = ts.mean()
    deg = 2 if n >= 3 else 1
    c, A = polyfit(ts, Ps, deg, tc)
    resid = Ps - A @ c
    a = 2 * c[2] if deg == 2 else np.zeros(3)
    return dict(
        tc=tc,
        p=c[0],
        v=c[1],
        a=a,
        deg=deg,
        n=n,
        t_first=ts[0],
        t_last=ts[-1],
        rms=float(np.sqrt(np.mean(resid**2))),
    )


def velocity_at(fit, s):
    return fit["v"] + fit["a"] * (s - fit["tc"])


def estimated_frame(fit):
    if fit is None:
        return np.eye(3)
    v = velocity_at(fit, fit["t_first"])
    if np.hypot(v[0], v[1]) < 1e-6:
        return np.eye(3)
    return ball_frame(v)


def flight_features(fit, F, n_frames, t_first):
    x = np.zeros(len(FLIGHT_NAMES))
    x[17], x[19], x[20] = n_frames, t_first, float(n_frames < 3)
    if fit is None:
        return x
    v_start, v = velocity_at(fit, fit["t_first"]), fit["v"]
    x[0:3], x[3:6] = F @ v_start, F @ v
    x[15], x[16], x[18] = np.linalg.norm(v), fit["rms"], fit["t_last"] - fit["t_first"]
    if fit["deg"] == 2:  # too_few_frames leaves the acceleration features at 0
        a_ng = fit["a"] - GVEC
        vh = v / np.linalg.norm(v)
        a_perp = a_ng - np.dot(a_ng, vh) * vh
        x[6:9], x[9:12] = F @ a_ng, F @ a_perp
        x[12:15] = F @ (np.cross(v, a_perp) / np.dot(v, v))
    return x


def impact_time(pre, t_hi):
    c0 = pre["p"][2] - Z_CONTACT
    c1, c2 = pre["v"][2], pre["a"][2] / 2
    if abs(c2) > 1e-9:
        disc = c1**2 - 4 * c2 * c0
        roots = (
            []
            if disc < 0
            else [(-c1 - np.sqrt(disc)) / (2 * c2), (-c1 + np.sqrt(disc)) / (2 * c2)]
        )
    else:
        roots = [-c0 / c1] if abs(c1) > 1e-9 else []
    after = [pre["tc"] + s for s in roots if pre["tc"] + s >= pre["t_last"] - EPS]
    t_imp = min(after) if after else pre["t_last"]
    return float(np.clip(t_imp, pre["t_last"], max(t_hi, pre["t_last"])))


def bounce_features(pre, F, ts_post, Ps_post):
    x = np.zeros(len(BOUNCE_NAMES))
    x[10], x[12] = len(ts_post), 1.0  # n_post, has_bounce: the trajectory does bounce
    if pre is None or len(ts_post) < 2:
        x[13] = (
            1.0  # too_few_post: the frames do not show the bounce, so the kick features stay 0
        )
        return x
    t_imp = impact_time(pre, ts_post[0])
    v_pre = velocity_at(pre, t_imp)
    s = ts_post - t_imp
    Y = Ps_post.copy()
    Y[:, 2] += G / 2 * s**2
    c, _ = polyfit(s, Y, 1, 0.0)
    v_post = c[1]
    x[0:3], x[3:6], x[6:9] = F @ v_pre, F @ v_post, F @ (v_post - v_pre)
    x[9] = abs(v_post[2] / v_pre[2]) if abs(v_pre[2]) > 1e-6 else 0.0
    x[11] = t_imp - pre["t_last"]
    return x


def launch_features(F, Ps_window, fit):
    x = np.zeros(len(LAUNCH_NAMES))
    if len(Ps_window):
        x[0:3] = Ps_window[0]
        x[3:6] = F @ Ps_window[0]
    if fit is not None:
        x[6:9] = F @ velocity_at(fit, fit["t_first"])
    return x


def process_trajectory(t, P, bounces, level, rng, debug=False):
    kc = contact_index(t, P, bounces)
    kr = rebound_index(P, kc) if len(bounces) else len(t)
    pre_end = preflight_end(t, P, bounces)
    t_pre = t[pre_end - 1] if pre_end > 0 else -np.inf
    ts, tt, Ps = observe(t, P, kc, kr, level, rng)
    ts = ts - t[0]  # all feature times are relative to the first kept sample
    rel = tt - t[0]

    in_w2 = tt <= t_pre + EPS
    in_w1 = in_w2 & (rel < W1_LENGTH - EPS)
    if len(bounces) and kr < len(
        t
    ):  # frames from the first sample clear of the table to contact + 50 ms
        in_post = (tt >= t[kr] - EPS) & (tt <= t[kc] + POST_LENGTH + EPS)
    else:
        in_post = np.zeros(len(tt), bool)

    out, fits = {}, {}
    for w, m in [("W1", in_w1), ("W2", in_w2)]:
        fit = fit_arc(ts[m], Ps[m])
        F = estimated_frame(fit)
        t_first = ts[m][0] if m.any() else 0.0
        out[w] = (
            flight_features(fit, F, int(m.sum()), t_first),
            None,
            launch_features(F, Ps[m], fit),
        )
        fits[w] = (fit, F)
    fit2, F2 = fits["W2"]
    if len(bounces):
        local = in_w2 & (tt >= t_pre - PRE_LOCAL - EPS)
        pre = fit_arc(ts[local], Ps[local]) if local.sum() >= 3 else fit2
        fits["pre"] = (pre, F2)
        xb = bounce_features(pre, F2, ts[in_post], Ps[in_post])
    else:
        xb = np.zeros(len(BOUNCE_NAMES))  # no bounce: zeros and has_bounce = 0
    out["W3"] = (out["W2"][0], xb, out["W2"][2])
    if not debug:
        return out
    return out, dict(
        ts=ts,
        rel=rel,
        P=Ps,
        in_w1=in_w1,
        in_w2=in_w2,
        in_post=in_post,
        fits=fits,
        t_contact=t[kc] - t[0] if kc < len(t) else None,
    )


def cell_path(window, level, cache_dir="cache"):
    return os.path.join(cache_dir, f"cell_{window}_{level}.npz")


def load_cell(window, level, cache_dir="cache"):
    z = np.load(cell_path(window, level, cache_dir))
    return {k: z[k] for k in z.files}
