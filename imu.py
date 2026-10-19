"""Optional IMU (gyroscope) support.

Disabled by default (--imu none). When enabled, gyro rates are integrated between
two frame timestamps, converted to image motion, and fused with the vision estimate:

    image shift x  ~= focal_px * yaw      (small angles)
    image shift y  ~= focal_px * pitch
    image rotation  =  roll

Why bother: during a sudden hit the frames are blurred and feature tracking fails,
but a gyro still measures the motion perfectly well.

Sign convention used here (camera axes, after --imu-axes mapping):
    +yaw   -> image content moves right
    +pitch -> image content moves down
    +roll  -> image content rotates clockwise
If your sensor disagrees, flip signs / swap axes with --imu-axes (e.g. --imu-axes=-y,x,z).
The program prints a vision-vs-IMU correlation at the end to help you find the mapping.

Sources:
    none                    vision only
    csv:path.csv            replay; columns t,gx,gy,gz (t in seconds from video start)
    serial:/dev/ttyUSB0     live; one line per sample "gx,gy,gz" (extra leading fields ignored)
    serial:/dev/ttyUSB0@921600
"""
import csv
import threading
import time
from collections import deque

import numpy as np

from motion import Motion


class AxisMap:
    """'x,y,z' -> camera (pitch, yaw, roll) = (+x, +y, +z) of the sensor. '-y,x,z' etc. allowed."""

    def __init__(self, spec="x,y,z"):
        names = {"x": 0, "y": 1, "z": 2}
        parts = [p.strip().lower() for p in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--imu-axes needs 3 comma-separated entries, got {spec!r}")
        self.index = np.array([names[p.lstrip("+-")] for p in parts])
        self.sign = np.array([-1.0 if p.startswith("-") else 1.0 for p in parts])

    def __call__(self, g):
        g = np.asarray(g, dtype=np.float64)
        return g[..., self.index] * self.sign


def _integrate(t, g, t0, t1, tolerance=0.05):
    """Integrate gyro rate g (Nx3, rad/s) over [t0, t1]. Returns rotation (3,) or None."""
    if len(t) < 2 or t1 <= t0:
        return None
    if t0 < t[0] - tolerance or t1 > t[-1] + tolerance:
        return None                              # no IMU data covering this interval
    t0c, t1c = np.clip([t0, t1], t[0], t[-1])
    inner = (t > t0c) & (t < t1c)
    ts = np.concatenate([[t0c], t[inner], [t1c]])
    interp = lambda x: np.array([np.interp(x, t, g[:, k]) for k in range(3)])
    gs = np.vstack([interp(t0c), g[inner], interp(t1c)])
    rot = (0.5 * (gs[1:] + gs[:-1]) * np.diff(ts)[:, None]).sum(axis=0)
    # if we had to clamp (data slightly behind), extend with the last rate
    rot += interp(t1c) * max(0.0, t1 - t1c) + interp(t0c) * max(0.0, t0c - t0)
    return rot


class CSVIMU:
    def __init__(self, path, axis_map, units="rad", time_offset=0.0):
        rows = []
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append([float(r["t"]), float(r["gx"]), float(r["gy"]), float(r["gz"])])
        data = np.array(rows)
        self.t = data[:, 0] + time_offset
        g = data[:, 1:]
        if units == "deg":
            g = np.deg2rad(g)
        self.g = axis_map(g)

    def integrate(self, t0, t1):
        return _integrate(self.t, self.g, t0, t1)

    def close(self):
        pass


class SerialIMU:
    """Reads 'gx,gy,gz' lines from a microcontroller in a background thread.
    Timestamps are taken on the host (time.monotonic) when each line arrives."""

    def __init__(self, port, baud, axis_map, units="rad", time_offset=0.0, maxlen=4000):
        import serial  # pip install pyserial
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.axis_map, self.units, self.offset = axis_map, units, time_offset
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            line = self.ser.readline()
            if not line:
                continue
            try:
                vals = [float(v) for v in line.decode(errors="ignore").replace(";", ",").split(",") if v.strip()]
            except ValueError:
                continue
            if len(vals) < 3:
                continue
            g = np.array(vals[-3:])
            if self.units == "deg":
                g = np.deg2rad(g)
            with self.lock:
                self.buf.append((time.monotonic() + self.offset, *self.axis_map(g)))

    def integrate(self, t0, t1):
        with self.lock:
            if len(self.buf) < 2:
                return None
            data = np.array(self.buf)
        return _integrate(data[:, 0], data[:, 1:], t0, t1)

    def close(self):
        self.running = False
        self.thread.join(timeout=1)
        self.ser.close()


def make_imu(spec, axes="x,y,z", units="rad", time_offset=0.0):
    if spec is None or spec.lower() == "none":
        return None
    kind, _, target = spec.partition(":")
    amap = AxisMap(axes)
    if kind == "csv":
        return CSVIMU(target, amap, units, time_offset)
    if kind == "serial":
        port, _, baud = target.partition("@")
        return SerialIMU(port, int(baud or 115200), amap, units, time_offset)
    raise ValueError(f"Unknown IMU source {spec!r} (use none, csv:FILE or serial:PORT[@BAUD])")


def focal_from_hfov(width_px, hfov_deg):
    return (width_px / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)


class MotionFusion:
    """Complementary blend of vision and gyro motion, with fallback to gyro on tracking loss / hits."""

    def __init__(self, focal_px, w_trans=0.3, w_rot=0.7, hit_px=25.0):
        self.f = focal_px
        self.w = np.array([w_trans, w_trans, w_rot])
        self.hit_px = hit_px

    def imu_motion(self, rot):
        pitch, yaw, roll = rot
        return np.array([self.f * yaw, self.f * pitch, roll])

    def fuse(self, vis: Motion, imu_m):
        """Returns (motion array, source label)."""
        if imu_m is None:
            return (vis.as_array(), "vision") if vis.ok else (np.zeros(3), "lost")
        if not vis.ok:
            return imu_m, "imu"
        v = vis.as_array()
        if np.hypot(*(v[:2] - imu_m[:2])) > self.hit_px:   # vision and gyro disagree a lot -> trust gyro
            return imu_m, "imu"
        return self.w * imu_m + (1 - self.w) * v, "fused"
