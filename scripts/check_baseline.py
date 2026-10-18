import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread.io import load_packed  # noqa: E402
from spinread.metrics import r2, rmse, sign_accuracy  # noqa: E402
from spinread.physics import FPS, ball_frame, physics_u, preflight_end, spin_label  # noqa: E402
from spinread.splits import load_splits  # noqa: E402

WINDOW = 0.1          # s: the first 100 ms (window W1)
MODES = ['intermediate']
NAMES = ['gyro', 'top', 'side']


def collect(data, splits):
    rows = []
    for (mode, direction), d in sorted(data.items()):
        if mode not in MODES:
            continue
        split = splits[(mode, direction)]
        for i in np.flatnonzero(split >= 0):
            a, b = d['offsets'][i], d['offsets'][i + 1]
            t, P = d['times'][a:b], d['positions'][a:b]
            bounces = d['bounces'][i][:d['n_bounces'][i]]
            end = preflight_end(t, P, bounces)
            m = (t - t[0] < WINDOW) & (np.arange(len(t)) < end)
            if m.sum() < 5:
                continue
            u, _, _ = physics_u(t[m], P[m], 'center')
            F = ball_frame(d['v0'][i])
            rows.append(dict(u=F @ u, y=spin_label(d['w0'][i], d['v0'][i]), split=split[i],
                             flight=end / FPS, speed=np.linalg.norm(d['v0'][i]),
                             key=f'{direction}/{int(d["traj_id"][i])}'))
    return rows


def report(label, Y, pred):
    out = [f'{label:<34}n={len(Y):>6}']
    for j, name in enumerate(NAMES):
        out.append(f'{name} {r2(Y[:, j], pred[:, j]):6.3f}')
    out.append(f'sign {sign_accuracy(Y[:, 1], pred[:, 1]):.3f}')
    print('  '.join(out))


def main():
    rows = collect(load_packed('cache'), load_splits('cache'))
    U = np.array([r['u'] for r in rows])
    Y = np.array([r['y'] for r in rows])
    S = np.array([r['split'] for r in rows])
    flight = np.array([r['flight'] for r in rows])
    speed = np.array([r['speed'] for r in rows])
    full = flight >= WINDOW
    tr, dev = S == 0, S == 1

    Yr = Y * 2 * np.pi                                     # rad/s for the fit
    sel = tr & full
    c = np.sum(U[sel] * Yr[sel]) / np.sum(U[sel] ** 2)     # omega ~ c * u, so k = 1/c
    pred = c * U / (2 * np.pi)                             # rev/s
    print(f'Rally shots: {tr.sum()} train, {dev.sum()} dev. '
          f'k fit on the {sel.sum()} train shots with >= 100 ms flight: k = {1 / c:.4f} (proposal check: 0.0076)\n')
    print('Dev R^2 per component, and topspin sign accuracy:')
    report('HEADLINE: flight >= 100 ms', Y[dev & full], pred[dev & full])
    report('all rally shots', Y[dev], pred[dev])
    print('\nBy pre-bounce flight time (window = min(100 ms, flight)):')
    for lo, hi in [(0, 0.03), (0.03, 0.06), (0.06, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 9)]:
        m = dev & (flight >= lo) & (flight < hi)
        if m.sum() > 20:
            report(f'  {lo * 1000:.0f}-{hi * 1000:.0f} ms', Y[m], pred[m])
    print('\nBy launch speed, flight >= 100 ms only:')
    for lo, hi in [(0, 8), (8, 14), (14, 20), (20, 31)]:
        m = dev & full & (speed >= lo) & (speed < hi)
        if m.sum() > 20:
            report(f'  {lo}-{hi} m/s', Y[m], pred[m])

    err = np.abs(pred[:, 1] - Y[:, 1])
    worst = np.argsort(-np.where(dev, err, -1))[:5]
    print('\nWorst 5 dev topspin errors (direction/traj_id: true vs predicted rev/s, flight, speed):')
    for i in worst:
        print(f'  {rows[i]["key"]:<22} {Y[i, 1]:7.1f} vs {pred[i, 1]:7.1f}   '
              f'{flight[i] * 1000:5.0f} ms   {speed[i]:5.1f} m/s')
    print('\nExpected for the HEADLINE row: top about 0.99, side about 0.97, gyro near 0.')


if __name__ == '__main__':
    main()
