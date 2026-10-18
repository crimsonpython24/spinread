import glob
import os

import numpy as np


def load_packed(cache_dir="cache"):
    out = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "synthetic_*.npz"))):
        z = np.load(path)
        d = {k: z[k] for k in z.files}
        out[(str(d["mode"]), str(d["direction"]))] = d
    return out


def get_trajectory(d, i):
    a, b = d["offsets"][i], d["offsets"][i + 1]
    return dict(
        t=d["times"][a:b],
        P=d["positions"][a:b],
        v0=d["v0"][i],
        w0=d["w0"][i],
        bounces=d["bounces"][i][: d["n_bounces"][i]],
        traj_id=int(d["traj_id"][i]),
    )


def load_keep(cache_dir="cache"):
    z = np.load(os.path.join(cache_dir, "keep.npz"))
    return {tuple(k.split("/")): z[k] for k in z.files}
