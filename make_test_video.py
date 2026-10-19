#!/usr/bin/env python3
"""Create a synthetic shaky video (+ matching gyro log + ground truth) for testing.

The "camera" does a slow intended pan, plus:
  - mechanical vibration (sinusoids at 6-15 Hz),
  - random jitter,
  - a few sudden hits (damped oscillation + motion blur).

Outputs (in --out-dir):
  shaky.mp4          the video to stabilize
  imu.csv            simulated gyroscope (t,gx,gy,gz in rad/s, with noise and bias)
  ground_truth.csv   true per-frame motion, incl. the intended pan

  python make_test_video.py                         # procedural test scene
  python make_test_video.py --image my_photo.jpg    # use your own picture as the scene
"""
import argparse
import csv
import os

import cv2
import numpy as np


def procedural_scene(w, h, rng):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.dstack([60 + 120 * xx / w, 60 + 120 * yy / h, np.full((h, w), 90.0)]).astype(np.uint8)
    for _ in range(160):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 80)), y + int(rng.integers(10, 80))), color, -1)
        elif kind == 1:
            cv2.circle(img, (x, y), int(rng.integers(5, 40)), color, -1)
        else:
            cv2.line(img, (x, y), (int(rng.integers(0, w)), int(rng.integers(0, h))), color, 2)
    for k in range(8):
        cv2.putText(img, f"TEST {k}", (int(rng.integers(0, w - 150)), int(rng.integers(30, h))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    noise = rng.normal(0, 10, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def damped(t, t0, f, tau):
    s = np.clip(t - t0, 0, None)
    return np.where(t >= t0, np.exp(-s / tau) * np.sin(2 * np.pi * f * s), 0.0)


def motion_blur(img, vx, vy):
    L = np.hypot(vx, vy)
    if L < 1.5:
        return img
    k = int(np.ceil(L)) | 1
    ker = np.zeros((k, k), np.float32)
    c = k // 2
    cv2.line(ker, (int(round(c - vx / 2)), int(round(c - vy / 2))),
             (int(round(c + vx / 2)), int(round(c + vy / 2))), 1.0, 1, cv2.LINE_AA)
    ker /= ker.sum()
    return cv2.filter2D(img, -1, ker)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="test_data")
    p.add_argument("--image", help="optional picture to use as the scene")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--size", default="640x480")
    p.add_argument("--vibration", type=float, default=6.0, help="vibration amplitude [px]")
    p.add_argument("--pan", type=float, default=60.0, help="intended pan amplitude [px]")
    p.add_argument("--hits", type=int, default=3, help="number of sudden hits")
    p.add_argument("--hit-strength", type=float, default=30.0, help="sudden hit amplitude [px]")
    p.add_argument("--hfov", type=float, default=62.0, help="camera horizontal FOV [deg] (for gyro scaling)")
    p.add_argument("--imu-rate", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    W, H = (int(v) for v in a.size.lower().split("x"))
    CW, CH = int(W * 1.6), int(H * 1.6)
    if a.image:
        scene = cv2.resize(cv2.imread(a.image), (CW, CH))
    else:
        scene = procedural_scene(CW, CH, rng)

    # ---- motion on a fine time grid (also the IMU sample times) ----
    dt = 1.0 / a.imu_rate
    t = np.arange(0, a.seconds + 1, dt)
    pan_x = a.pan * np.sin(2 * np.pi * 0.05 * t)
    pan_y = 0.3 * a.pan * np.sin(2 * np.pi * 0.035 * t + 1.0)
    A = a.vibration
    vib_x = A * (np.sin(2 * np.pi * 7.0 * t) + 0.5 * np.sin(2 * np.pi * 13.3 * t + 0.5))
    vib_y = A * (0.8 * np.sin(2 * np.pi * 8.1 * t + 1.0) + 0.4 * np.sin(2 * np.pi * 15.0 * t))
    vib_a = np.radians(0.5) * np.sin(2 * np.pi * 6.3 * t + 2.0)
    smooth_noise = lambda s: np.convolve(rng.normal(0, s, t.size), np.ones(15) / 15, mode="same") * np.sqrt(15)
    jit_x, jit_y, jit_a = smooth_noise(1.5), smooth_noise(1.5), smooth_noise(np.radians(0.15))
    hit_x = np.zeros_like(t); hit_y = np.zeros_like(t); hit_a = np.zeros_like(t)
    hit_times = np.sort(rng.uniform(1.5, a.seconds - 1.5, a.hits))
    for th in hit_times:
        ang = rng.uniform(0, 2 * np.pi)
        d = damped(t, th, 4.0, 0.25)
        hit_x += a.hit_strength * np.cos(ang) * d
        hit_y += a.hit_strength * np.sin(ang) * d
        hit_a += np.radians(3.0 * a.hit_strength / 30) * rng.choice([-1, 1]) * d

    X = pan_x + vib_x + jit_x + hit_x
    Y = pan_y + vib_y + jit_y + hit_y
    ANG = vib_a + jit_a + hit_a

    # ---- render frames ----
    os.makedirs(a.out_dir, exist_ok=True)
    n = int(a.seconds * a.fps)
    tf = np.arange(n) / a.fps
    fx, fy, fa = (np.interp(tf, t, v) for v in (X, Y, ANG))
    vx, vy = (np.interp(tf, t, np.gradient(v, dt)) for v in (X, Y))
    exposure = 0.5 / a.fps
    out = cv2.VideoWriter(os.path.join(a.out_dir, "shaky.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    Ccx, Ccy, Cox, Coy = CW / 2, CH / 2, W / 2, H / 2
    for i in range(n):
        c, s = np.cos(fa[i]), np.sin(fa[i])
        M = np.array([[c, -s, Cox + fx[i] - (c * Ccx - s * Ccy)],
                      [s, c, Coy + fy[i] - (s * Ccx + c * Ccy)]])
        frame = cv2.warpAffine(scene, M, (W, H), flags=cv2.INTER_LINEAR)
        frame = motion_blur(frame, vx[i] * exposure, vy[i] * exposure)
        frame = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()

    with open(os.path.join(a.out_dir, "ground_truth.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "t", "x", "y", "a", "pan_x", "pan_y"])
        px, py = np.interp(tf, t, pan_x), np.interp(tf, t, pan_y)
        for i in range(n):
            wr.writerow([i, f"{tf[i]:.4f}", f"{fx[i]:.4f}", f"{fy[i]:.4f}", f"{fa[i]:.6f}",
                         f"{px[i]:.4f}", f"{py[i]:.4f}"])

    # ---- simulated gyro: camera axes x=pitch, y=yaw, z=roll (matches --imu-axes x,y,z) ----
    focal = (W / 2) / np.tan(np.radians(a.hfov) / 2)
    yaw_rate = np.gradient(X, dt) / focal
    pitch_rate = np.gradient(Y, dt) / focal
    roll_rate = np.gradient(ANG, dt)
    gyro = np.stack([pitch_rate, yaw_rate, roll_rate], 1)
    gyro += rng.normal(0, 0.004, gyro.shape) + np.array([0.003, -0.002, 0.001])   # noise + bias
    with open(os.path.join(a.out_dir, "imu.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "gx", "gy", "gz"])
        for k in range(t.size):
            wr.writerow([f"{t[k]:.5f}", *(f"{v:.6f}" for v in gyro[k])])

    print(f"Wrote {n} frames ({W}x{H} @ {a.fps} fps) to {a.out_dir}/shaky.mp4")
    print(f"Hits at t = {', '.join(f'{x:.2f}s' for x in hit_times)}")
    print(f"Also: {a.out_dir}/imu.csv, {a.out_dir}/ground_truth.csv")


if __name__ == "__main__":
    main()
