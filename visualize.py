"""Side-by-side comparison frames and motion plots."""
import os

import cv2
import numpy as np

SOURCE_COLORS = {"vision": (80, 200, 80), "fused": (230, 180, 60), "imu": (0, 165, 255),
                 "lost": (0, 0, 255), "start": (200, 200, 200)}


def _grid(img, color=(0, 255, 255)):
    h, w = img.shape[:2]
    for k in (1, 2, 3):
        cv2.line(img, (w * k // 4, 0), (w * k // 4, h), color, 1, cv2.LINE_AA)
        cv2.line(img, (0, h * k // 4), (w, h * k // 4), color, 1, cv2.LINE_AA)
    return img


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def side_by_side(original, stabilized, source="vision", info="", max_width=640, grid=True):
    """Original | stabilized, with a reference grid so shake is easy to see."""
    h, w = original.shape[:2]
    s = min(1.0, max_width / w)
    a = cv2.resize(original, None, fx=s, fy=s) if s < 1 else original.copy()
    b = cv2.resize(stabilized, None, fx=s, fy=s) if s < 1 else stabilized.copy()
    if grid:
        _grid(a), _grid(b)
    _label(a, "ORIGINAL")
    _label(b, "STABILIZED")
    out = np.hstack([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b])
    bar = np.zeros((26, out.shape[1], 3), np.uint8)
    color = SOURCE_COLORS.get(source, (255, 255, 255))
    cv2.circle(bar, (14, 13), 7, color, -1)
    cv2.putText(bar, f"motion source: {source}   {info}", (28, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([out, bar])


def _shade(ax, t, sources):
    """Shade frames where vision tracking failed (IMU fallback or lost)."""
    bad = np.array([s in ("imu", "lost") for s in sources])
    if not bad.any():
        return
    color = {True: "red"}
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    dt = (t[1] - t[0]) if len(t) > 1 else 0.03
    for s, e in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[s] - dt / 2, t[e - 1] + dt / 2, color=color[True], alpha=0.15, lw=0)


def save_plots(out_dir, t, raw, smooth, corrections, sources, imu_traj=None, gt=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(t, float)
    raw, smooth, corr = (np.asarray(a, float) for a in (raw, smooth, corrections))
    n = min(len(t), len(raw), len(smooth), len(corr))
    t, raw, smooth, corr, sources = t[:n], raw[:n], smooth[:n], corr[:n], list(sources[:n])
    stab = raw + corr                       # where the content ends up after stabilization

    conv = [1.0, 1.0, np.degrees(1.0)]
    names = ["x shift [px]", "y shift [px]", "rotation [deg]"]

    # ---- 1. trajectories ---------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for k, ax in enumerate(axes):
        if gt is not None:
            m = min(n, len(gt["total"]))
            ax.plot(t[:m], gt["total"][:m, k] * conv[k], "k--", lw=1, alpha=0.6, label="true camera motion")
            ax.plot(t[:m], gt["intended"][:m, k] * conv[k], color="green", lw=2, alpha=0.5,
                    label="true intended motion (no shake)")
        ax.plot(t, raw[:, k] * conv[k], color="0.55", lw=1, label="estimated raw trajectory")
        if imu_traj is not None:
            ax.plot(t, np.asarray(imu_traj)[:n, k] * conv[k], color="orange", lw=1, alpha=0.8,
                    label="gyro-only trajectory")
        ax.plot(t, stab[:, k] * conv[k], color="tab:blue", lw=2, label="stabilized trajectory")
        _shade(ax, t, sources)
        ax.set_ylabel(names[k])
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time [s]   (red bands = vision tracking failed)")
    fig.suptitle(f"Camera trajectory {title}")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "trajectory.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # ---- 2. frame-to-frame jitter -----------------------------------------
    d_raw = np.abs(np.diff(raw, axis=0))
    d_stab = np.abs(np.diff(stab, axis=0))
    tt = t[1:]
    d_true = None
    if gt is not None:                       # honest check: true motion + applied correction
        m = min(n, len(gt["total"]))
        d_true = np.abs(np.diff(gt["total"][:m] + corr[:m], axis=0))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for k, ax in enumerate(axes):
        r_rms = np.sqrt(np.mean(d_raw[:, k] ** 2)) * conv[k]
        s_rms = np.sqrt(np.mean(d_stab[:, k] ** 2)) * conv[k]
        ax.plot(tt, d_raw[:, k] * conv[k], color="tab:red", lw=1, alpha=0.8, label=f"original (RMS {r_rms:.2f})")
        ax.plot(tt, d_stab[:, k] * conv[k], color="tab:blue", lw=1.2, label=f"stabilized (RMS {s_rms:.2f})")
        if d_true is not None:
            g_rms = np.sqrt(np.mean(d_true[:, k] ** 2)) * conv[k]
            ax.plot(tt[:len(d_true)], d_true[:, k] * conv[k], color="black", lw=0.8, alpha=0.6,
                    label=f"stabilized, true residual from ground truth (RMS {g_rms:.2f})")
        _shade(ax, t, sources)
        ax.set_ylabel("|Δ " + names[k].split(" [")[0] + "| per frame")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Frame-to-frame motion (what your eye sees as shake) {title}")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "jitter.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    rms_raw = np.sqrt(np.mean(d_raw[:, :2] ** 2))
    rms_stab = np.sqrt(np.mean(d_stab[:, :2] ** 2))
    rms_true = None if d_true is None else np.sqrt(np.mean(d_true[:, :2] ** 2))
    return p1, p2, rms_raw, rms_stab, rms_true


def load_ground_truth(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    total = np.array([[float(r["x"]), float(r["y"]), float(r["a"])] for r in rows])
    intended = np.array([[float(r["pan_x"]), float(r["pan_y"]), 0.0] for r in rows])
    return {"total": total - total[0], "intended": intended - total[0]}
