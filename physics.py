import numpy as np

FPS = 500  # simulator sampling rate
G = 9.81
GVEC = np.array([0.0, 0.0, -G])
R_BALL = 0.02  # m
TABLE_HEIGHT = 0.76  # simulator table top; ball centre at contact = 0.78 m


def polyfit(t, y, deg, t0):
    """Least-squares polynomial per column of y, in powers of (t - t0). Returns coefficients
    c with c[j] the coefficient of (t - t0)**j, shape (deg + 1, n_columns)."""
    A = np.vander(t - t0, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, A


def ball_frame(v):
    xh = np.array([v[0], v[1], 0.0])
    xh /= np.linalg.norm(xh)
    zh = np.array([0.0, 0.0, 1.0])
    yh = np.cross(zh, xh)
    return np.stack([xh, yh, zh])


def spin_label(w0, v0):
    return ball_frame(v0) @ np.asarray(w0) / (2 * np.pi)


def physics_u(t, P, where="center"):
    t0 = t.mean() if where in ("center", "centre") else t[0]
    c, _ = polyfit(t, P, 2, t0)
    v, a = c[1], 2 * c[2]
    a_ng = a - GVEC
    vh = v / np.linalg.norm(v)
    a_perp = a_ng - np.dot(a_ng, vh) * vh
    return np.cross(v, a_perp) / np.dot(v, v), v, a_perp


def contact_index(t, P, bounces):
    if len(bounces) == 0:
        return len(t)
    z = P[:, 2]
    i0 = int(round((bounces[0] - t[0]) * FPS))
    lo, hi = max(0, i0 - 25), min(len(z), i0 + 25)
    j = lo + int(np.argmin(z[lo:hi]))
    while j > 0 and z[j - 1] < TABLE_HEIGHT + R_BALL + 0.001:
        j -= 1
    return j


def preflight_end(t, P, bounces):
    return max(0, contact_index(t, P, bounces) - 2) if len(bounces) else len(t)
