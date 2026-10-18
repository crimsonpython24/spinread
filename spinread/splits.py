import os

import numpy as np

TRAIN, DEV, TEST, DROPPED = 0, 1, 2, -1


def load_splits(cache_dir="cache"):
    z = np.load(os.path.join(cache_dir, "splits.npz"))
    return {tuple(k.split("/")): z[k] for k in z.files}
