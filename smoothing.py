"""Trajectory smoothing.

The camera "trajectory" is the running sum of frame-to-frame motions (x, y, angle).
Shake = high-frequency part of it; intended motion (pans) = low-frequency part.
We low-pass filter the trajectory and move every frame from where it *is* (raw)
to where it *should be* (smoothed).

- Offline: we know the whole future -> symmetric Gaussian filter (best quality).
- Live:    we only know `lookahead` frames of future (= output delay).
           A local *linear* fit removes the lag a one-sided average would cause during pans.
"""
import numpy as np


def _gauss(offsets, sigma):
    return np.exp(-0.5 * (np.asarray(offsets, float) / sigma) ** 2)


def smooth_offline(traj, radius):
    traj = np.asarray(traj, dtype=np.float64)
    if radius <= 0 or len(traj) < 2:
        return traj.copy()
    k = _gauss(np.arange(-radius, radius + 1), radius / 2.0)
    k /= k.sum()
    padded = np.pad(traj, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], k, mode="valid") for j in range(traj.shape[1])], axis=1)


class LiveSmoother:
    def __init__(self, radius=30, lookahead=10, trend=True):
        self.radius = max(1, int(radius))
        self.lookahead = max(0, int(lookahead))
        self.trend = trend

    def smooth_at(self, traj, i):
        """Smoothed position of frame i, using traj[i-radius .. i+lookahead] (whatever exists)."""
        n = len(traj)
        lo, hi = max(0, i - self.radius), min(n - 1, i + self.lookahead)
        offs = np.arange(lo, hi + 1, dtype=np.float64) - i
        ys = np.asarray(traj[lo:hi + 1], dtype=np.float64)
        w = _gauss(offs, self.radius / 2.0)
        if self.trend and len(offs) >= 3 and (hi - i) < (i - lo):
            # weighted least-squares line y = a + b*offset, evaluated at offset 0
            sw, st, stt = w.sum(), (w * offs).sum(), (w * offs * offs).sum()
            sy, sty = (w[:, None] * ys).sum(0), ((w * offs)[:, None] * ys).sum(0)
            den = sw * stt - st * st
            if abs(den) > 1e-9:
                return (stt * sy - st * sty) / den
        return (w[:, None] * ys).sum(0) / w.sum()


# --------------------------------------------------------------------------- "lock" (tripod) mode
# Instead of following a smoothed path, the virtual camera stays completely still
# while the real camera wobbles inside a dead-zone. Only when the real camera moves
# further than the dead-zone (an intentional re-aim) does the view follow - and that
# follow movement is itself smoothed. Result: vibrations are removed entirely, not
# just reduced, at the cost of a larger crop (correction can reach the dead-zone).

def _deadband(traj, dz):
    out = np.empty_like(traj)
    s = traj[0].copy()
    for i, p in enumerate(traj):
        e = p - s
        s = s + np.sign(e) * np.maximum(np.abs(e) - dz, 0.0)
        out[i] = s
    return out


def lock_offline(traj, deadzone, radius):
    """deadzone = (x_px, y_px, angle_rad). Forward+backward pass so re-aims are centred in time."""
    traj = np.asarray(traj, dtype=np.float64)
    dz = np.asarray(deadzone, dtype=np.float64)
    fwd = _deadband(traj, dz)
    bwd = _deadband(traj[::-1], dz)[::-1]
    return smooth_offline(0.5 * (fwd + bwd), radius)


class LiveLock:
    """Causal version: dead-zone follower + exponential smoothing of re-aim moves."""

    def __init__(self, deadzone, radius=30):
        self.dz = np.asarray(deadzone, dtype=np.float64)
        self.alpha = 2.0 / (max(1, radius) + 1)       # EMA, similar time constant to `radius`
        self.anchor = None
        self.state = None
        self.lookahead = 0

    def smooth_at(self, traj, i):
        p = np.asarray(traj[i], dtype=np.float64)
        if self.anchor is None:
            self.anchor, self.state = p.copy(), p.copy()
        e = p - self.anchor
        self.anchor = self.anchor + np.sign(e) * np.maximum(np.abs(e) - self.dz, 0.0)
        self.state = self.state + self.alpha * (self.anchor - self.state)
        return self.state.copy()
