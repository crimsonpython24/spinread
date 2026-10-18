import numpy as np

FPS = 500  # simulator sampling rate
G = 9.81
GVEC = np.array([0.0, 0.0, -G])
R_BALL = 0.02  # m
TABLE_HEIGHT = 0.76  # simulator table top; ball center at contact = 0.78 m
K_MAGNUS, K_DRAG = 0.0075, 0.133  # fitted on a sample of the released data


def polyfit(t, y, deg, t0):
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


def air_accel(v, w):
    """Drag + Magnus acceleration for velocity v, shape (3,) or (N, 3), and spin w."""
    return -K_DRAG * np.linalg.norm(
        v, axis=-1, keepdims=True
    ) * v + K_MAGNUS * np.cross(w, v)


def physics_u(t, P, where="center"):
    t0 = t.mean() if where == "center" else t[0]
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
