import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.io import load_keep  # noqa: E402

SEED = 0
FRACTIONS = (0.7, 0.1, 0.2)


def main():
    keep = load_keep("cache")
    out, rows = {}, []
    rng = np.random.default_rng(SEED)
    for key in sorted(keep):
        k = keep[key]
        idx = np.flatnonzero(k)
        idx = idx[rng.permutation(len(idx))]
        n_tr = int(round(FRACTIONS[0] * len(idx)))
        n_dev = int(round(FRACTIONS[1] * len(idx)))
        split = np.full(len(k), -1, dtype=np.int8)
        split[idx[:n_tr]] = 0
        split[idx[n_tr : n_tr + n_dev]] = 1
        split[idx[n_tr + n_dev :]] = 2
        out["/".join(key)] = split
        rows.append(
            dict(
                mode=key[0],
                direction=key[1],
                train=int((split == 0).sum()),
                dev=int((split == 1).sum()),
                test=int((split == 2).sum()),
                dropped=int((split == -1).sum()),
            )
        )
    np.savez("cache/splits.npz", **out)
    os.makedirs("results", exist_ok=True)
    with open("results/split_sizes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        f'{"mode":<13}{"direction":<15}{"train":>8}{"dev":>8}{"test":>8}{"dropped":>9}'
    )
    for r in rows:
        print(
            f'{r["mode"]:<13}{r["direction"]:<15}{r["train"]:>8}{r["dev"]:>8}{r["test"]:>8}{r["dropped"]:>9}'
        )
    tot = {c: sum(r[c] for r in rows) for c in ["train", "dev", "test", "dropped"]}
    print(
        f'{"TOTAL":<28}{tot["train"]:>8}{tot["dev"]:>8}{tot["test"]:>8}{tot["dropped"]:>9}'
    )
    print("Wrote cache/splits.npz and results/split_sizes.csv")


if __name__ == "__main__":
    main()
