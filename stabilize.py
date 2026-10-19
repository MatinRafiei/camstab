#!/usr/bin/env python3
"""Software video stabilizer: offline (recorded video) and live (camera) modes.

Examples
  python stabilize.py offline test_data/shaky.mp4
  python stabilize.py offline test_data/shaky.mp4 --imu csv:test_data/imu.csv --ground-truth test_data/ground_truth.csv
  python stabilize.py live --source 0                      # USB webcam
  python stabilize.py live --source csi-jetson             # Jetson CSI camera
  python stabilize.py live --source libcamera              # Raspberry Pi camera (GStreamer libcamerasrc)
  python stabilize.py live --source 0 --imu serial:/dev/ttyUSB0@115200 --imu-units deg
"""
import argparse
import os
import time
from collections import deque

import cv2
import numpy as np

from imu import MotionFusion, focal_from_hfov, make_imu
from motion import MotionEstimator
from smoothing import LiveLock, LiveSmoother, lock_offline, smooth_offline
from visualize import load_ground_truth, save_plots, side_by_side


# --------------------------------------------------------------------------- core
class TrajectoryTracker:
    """Accumulates per-frame motion (vision, optionally fused with gyro) into a trajectory."""

    def __init__(self, estimator, fusion, imu=None):
        self.est, self.fusion, self.imu = estimator, fusion, imu
        self.traj, self.sources, self.vis, self.imu_m = [], [], [], []
        self.t_prev = None

    def add(self, frame, t):
        vis = self.est.estimate(frame)
        imu_m = None
        if self.imu is not None and self.t_prev is not None:
            rot = self.imu.integrate(self.t_prev, t)
            if rot is not None:
                imu_m = self.fusion.imu_motion(rot)
        if not self.traj:
            m, src = np.zeros(3), "start"
        else:
            m, src = self.fusion.fuse(vis, imu_m)
        prev = self.traj[-1] if self.traj else np.zeros(3)
        self.traj.append(prev + m)
        self.sources.append(src)
        self.vis.append(vis)
        self.imu_m.append(imu_m)
        self.t_prev = t
        return src

    def imu_trajectory(self):
        if self.imu is None:
            return None
        steps = [m if m is not None else np.zeros(3) for m in self.imu_m]
        return np.cumsum(steps, axis=0)


class Warper:
    """Moves a frame from its raw position to the smoothed one; zooms a bit to hide black borders."""

    def __init__(self, w, h, zoom=1.08, max_shift_frac=0.12, max_angle_deg=6.0):
        self.w, self.h, self.zoom = w, h, zoom
        self.lim = np.array([max_shift_frac * w, max_shift_frac * h, np.radians(max_angle_deg)])

    def matrix(self, dx, dy, da):
        cx, cy = self.w / 2.0, self.h / 2.0
        c, s = np.cos(da), np.sin(da)
        R = np.array([[c, -s, cx - c * cx + s * cy + dx],
                      [s, c, cy - s * cx - c * cy + dy],
                      [0, 0, 1.0]])
        z = self.zoom
        Z = np.array([[z, 0, (1 - z) * cx], [0, z, (1 - z) * cy], [0, 0, 1.0]])
        return (Z @ R)[:2]

    def needed_zoom(self, corr, max_zoom=1.5):
        """Smallest zoom that hides all black borders for this correction."""
        dx, dy, da = corr
        c, sn = np.cos(da), np.sin(da)
        cx, cy = self.w / 2.0, self.h / 2.0
        corners = np.array([[0, 0], [self.w, 0], [0, self.h], [self.w, self.h]], float) - [cx, cy]
        lo, hi = 1.0, max_zoom
        for _ in range(25):
            z = 0.5 * (lo + hi)
            q = corners / z - [dx, dy]                     # undo zoom, then translation
            p = np.c_[c * q[:, 0] + sn * q[:, 1], -sn * q[:, 0] + c * q[:, 1]] + [cx, cy]  # undo rotation
            inside = (p[:, 0] >= 0) & (p[:, 0] <= self.w) & (p[:, 1] >= 0) & (p[:, 1] <= self.h)
            lo, hi = (lo, z) if inside.all() else (z, hi)
        return hi

    def apply(self, frame, raw, smooth):
        corr = np.clip(np.asarray(smooth) - np.asarray(raw), -self.lim, self.lim)
        out = cv2.warpAffine(frame, self.matrix(*corr), (self.w, self.h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out, corr


def parse_deadzone(spec):
    x, y, a = (float(v) for v in spec.split(","))
    return np.array([x, y, np.radians(a)])


def build_tracker(args, width):
    imu = make_imu(args.imu, args.imu_axes, args.imu_units, args.imu_time_offset)
    fusion = MotionFusion(focal_from_hfov(width, args.hfov), args.imu_weight_trans,
                          args.imu_weight_rot, args.hit_px)
    est = MotionEstimator(process_width=args.process_width, enhance=not args.no_enhance,
                          keyframe=args.keyframe)
    return TrajectoryTracker(est, fusion, imu)


def imu_sign_check(tracker):
    """Correlate vision and gyro motion; a negative value means the axis sign is flipped."""
    if tracker.imu is None:
        return
    pairs = [(v.as_array(), m) for v, m in zip(tracker.vis, tracker.imu_m) if v.ok and m is not None]
    if len(pairs) < 30:
        print("IMU check: not enough overlapping vision/IMU samples.")
        return
    V = np.array([p[0] for p in pairs])
    I = np.array([p[1] for p in pairs])
    print("IMU vs vision correlation (should be clearly positive):")
    for k, name in enumerate(["x  <- yaw  ", "y  <- pitch", "rot <- roll "]):
        if V[:, k].std() < 1e-9 or I[:, k].std() < 1e-9:
            print(f"   {name}: n/a (no motion)")
            continue
        r = np.corrcoef(V[:, k], I[:, k])[0, 1]
        hint = "" if r > 0.3 else ("  <-- sign looks flipped, negate this axis in --imu-axes" if r < -0.3
                                   else "  <-- weak, check axis mapping / --hfov / time offset")
        print(f"   {name}: {r:+.2f}{hint}")


def report(args, out_dir, t, tracker, smooth, corrections):
    gt = load_ground_truth(args.ground_truth) if getattr(args, "ground_truth", None) else None
    title = "(vision + IMU)" if tracker.imu is not None else "(vision only)"
    p1, p2, r0, r1, rt = save_plots(out_dir, t, tracker.traj, smooth, corrections, tracker.sources,
                                tracker.imu_trajectory(), gt, title)
    m = len(corrections)
    data = np.c_[np.asarray(t)[:m], np.asarray(tracker.traj)[:m], np.asarray(smooth)[:m], np.asarray(corrections)[:m]]
    np.savetxt(os.path.join(out_dir, "trajectory.csv"), data, delimiter=",", fmt="%.5f",
               header="t,raw_x,raw_y,raw_a,smooth_x,smooth_y,smooth_a,corr_x,corr_y,corr_a", comments="")
    n = len(tracker.sources)
    counts = {s: tracker.sources.count(s) for s in sorted(set(tracker.sources))}
    print(f"\nFrames: {n}   motion sources: {counts}")
    print(f"Frame-to-frame shake (RMS px): original {r0:.2f} -> stabilized {r1:.2f}")
    if rt is not None:
        print(f"  ...checked against ground truth: true residual shake {rt:.2f} px")
    print(f"Plots: {p1}\n       {p2}")
    imu_sign_check(tracker)


class Display:
    def __init__(self, enabled):
        self.enabled = enabled

    def show(self, img):
        if not self.enabled:
            return True
        try:
            cv2.imshow("camstab  (q = quit)", img)
            return (cv2.waitKey(1) & 0xFF) != ord("q")
        except cv2.error:
            print("No display available - continuing headless (use --no-display to silence).")
            self.enabled = False
            return True

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()


def writer(path, fps, size):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


# --------------------------------------------------------------------------- offline
def run_offline(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = build_tracker(args, w)
    first = int(round(args.start * fps))
    last = int(round(args.end * fps)) if args.end else None

    def frames():
        """Yield (video frame index, frame) for the selected time range."""
        c = cv2.VideoCapture(args.input)
        k = 0
        while True:
            ok, f = c.read()
            if not ok or (last is not None and k >= last):
                break
            if k >= first:
                yield k, f
            k += 1
        c.release()

    # pass 1: estimate motion
    t0 = time.time()
    i = 0
    for k, frame in frames():
        tracker.add(frame, k / fps)        # IMU timestamps stay relative to the video start
        i += 1
        if i % 100 == 0:
            print(f"  pass 1: {i} frames ({i / (time.time() - t0):.0f} fps)")
    cap.release()
    if i < 2:
        raise SystemExit("Video too short (check --start / --end).")
    print(f"Pass 1 (motion estimation): {i} frames at {i / (time.time() - t0):.1f} fps")

    raw = np.array(tracker.traj)
    if args.mode == "lock":
        smooth = lock_offline(raw, parse_deadzone(args.deadzone), args.radius)
    else:
        smooth = smooth_offline(raw, args.radius)
    warper = Warper(w, h, 1.0, args.max_shift, args.max_angle)
    if args.zoom == "auto":
        corr = np.clip(smooth - raw, -warper.lim, warper.lim)
        need = np.array([warper.needed_zoom(c, args.max_zoom) for c in corr])
        zoom = float(min(np.percentile(need, args.auto_zoom_pct), args.max_zoom))
        print(f"Auto zoom: {zoom:.3f} (black borders hidden in {args.auto_zoom_pct:g}% of frames)")
    else:
        zoom = float(args.zoom)
    warper.zoom = zoom

    # pass 2: warp and write
    os.makedirs(args.out, exist_ok=True)
    src = frames()
    w_stab = writer(os.path.join(args.out, "stabilized.mp4"), fps, (w, h))
    w_cmp = None
    disp = Display(args.show)
    corrections = []
    for j, (k, frame) in zip(range(len(raw)), src):
        stab, corr = warper.apply(frame, raw[j], smooth[j])
        corrections.append(corr)
        w_stab.write(stab)
        sbs = side_by_side(frame, stab, tracker.sources[j], f"t={k / fps:6.2f}s")
        if w_cmp is None:
            w_cmp = writer(os.path.join(args.out, "comparison.mp4"), fps, (sbs.shape[1], sbs.shape[0]))
        w_cmp.write(sbs)
        if not disp.show(sbs):
            break
    w_stab.release(), w_cmp.release(), disp.close()
    print(f"Videos: {args.out}/stabilized.mp4, {args.out}/comparison.mp4")

    t = (first + np.arange(len(corrections))) / fps
    report(args, args.out, t, tracker, smooth[:len(corrections)], corrections)
    if tracker.imu:
        tracker.imu.close()


# --------------------------------------------------------------------------- live
def open_source(src, width, height, fps):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, False
    if src == "csi-jetson":
        src = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
               f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
    elif src == "libcamera":
        src = (f"libcamerasrc ! video/x-raw,width={width},height={height},framerate={fps}/1 ! "
               f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
    if "!" in src:
        return cv2.VideoCapture(src, cv2.CAP_GSTREAMER), False
    return cv2.VideoCapture(src), True          # a video file, used to simulate a camera


def run_live(args):
    cap, is_file = open_source(args.source, args.width, args.height, args.fps)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        raise SystemExit(f"Cannot read from source {args.source!r}")
    h, w = frame.shape[:2]
    file_fps = (cap.get(cv2.CAP_PROP_FPS) or args.fps) if is_file else args.fps
    tracker = build_tracker(args, w)
    if args.mode == "lock":
        smoother = LiveLock(parse_deadzone(args.deadzone), args.radius)
    else:
        smoother = LiveSmoother(args.radius, args.lookahead, trend=not args.no_trend)
    if args.zoom == "auto":
        print("Note: --zoom auto only works offline; using 1.15 for live.")
        args.zoom = 1.15
    warper = Warper(w, h, float(args.zoom), args.max_shift, args.max_angle)
    disp = Display(not args.no_display)
    os.makedirs(args.out, exist_ok=True)

    rec = writer(args.record, file_fps, (w, h)) if args.record else None
    w_cmp = None
    buffer = deque()
    times, smooth_out, corrections = [], [], []
    t_start, n_out, t_fps = time.monotonic(), 0, time.monotonic()
    fps_meas = 0.0
    print(f"Live: {w}x{h}, output delay = {args.lookahead} frames. Press q (or Ctrl+C) to stop.")

    def emit(j, f):
        nonlocal w_cmp, n_out, t_fps, fps_meas
        s = smoother.smooth_at(tracker.traj, j)
        stab, corr = warper.apply(f, tracker.traj[j], s)
        smooth_out.append(s)
        corrections.append(corr)
        n_out += 1
        if n_out % 15 == 0:
            now = time.monotonic()
            fps_meas, t_fps = 15 / (now - t_fps), now
        sbs = side_by_side(f, stab, tracker.sources[j], f"{fps_meas:.1f} fps")
        if args.save:
            if w_cmp is None:
                w_cmp = writer(os.path.join(args.out, "live_comparison.mp4"), file_fps, (sbs.shape[1], sbs.shape[0]))
            w_cmp.write(sbs)
        return disp.show(sbs)

    i, running = 0, True
    try:
        while running:
            t = i / file_fps if is_file else time.monotonic() - t_start
            if rec:
                rec.write(frame)
            tracker.add(frame, t)
            times.append(t)
            buffer.append((i, frame))
            i += 1
            while len(buffer) > args.lookahead and running:
                running = emit(*buffer.popleft())
            ok, frame = cap.read()
            if not ok:
                break
    except KeyboardInterrupt:
        pass
    while buffer and running:
        running = emit(*buffer.popleft())

    cap.release()
    if rec:
        rec.release()
        print(f"Raw recording: {args.record}  (re-run it with `offline` for the best-quality result)")
    if w_cmp:
        w_cmp.release()
        print(f"Comparison video: {args.out}/live_comparison.mp4")
    disp.close()
    report(args, args.out, times[:len(corrections)], tracker, smooth_out, corrections)
    if tracker.imu:
        tracker.imu.close()


# --------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        g = sp.add_argument_group("stabilization")
        g.add_argument("--mode", choices=["smooth", "lock"], default="smooth",
                       help="smooth = follow a smoothed path; lock = tripod-like, hold the view still")
        g.add_argument("--deadzone", default="40,40,2",
                       help="lock mode: how far the camera may wobble (x_px,y_px,deg) before the view follows")
        g.add_argument("--radius", type=int, default=30, help="smoothing radius in frames (bigger = smoother, more crop)")
        g.add_argument("--zoom", default="1.08", help="zoom to hide black borders (1.0 = off, 'auto' = offline only)")
        g.add_argument("--max-zoom", type=float, default=1.25, help="upper limit for --zoom auto")
        g.add_argument("--auto-zoom-pct", type=float, default=98.0, help="--zoom auto hides borders in this %% of frames")
        g.add_argument("--no-enhance", action="store_true", help="disable CLAHE contrast boost for tracking")
        g.add_argument("--keyframe", action="store_true",
                       help="track against a reference frame (less slow drift, but noisier during hits)")
        g.add_argument("--max-shift", type=float, default=0.12, help="max correction as fraction of frame size")
        g.add_argument("--max-angle", type=float, default=6.0, help="max rotation correction [deg]")
        g.add_argument("--process-width", type=int, default=480, help="width used for feature tracking (speed)")
        g.add_argument("--out", default="output", help="output folder")
        i = sp.add_argument_group("IMU (optional)")
        i.add_argument("--imu", default="none", help="none | csv:FILE | serial:PORT[@BAUD]")
        i.add_argument("--imu-axes", default="x,y,z", help="sensor axes for camera pitch,yaw,roll, e.g. --imu-axes=-y,x,z (use '=' when it starts with '-')")
        i.add_argument("--imu-units", choices=["rad", "deg"], default="rad", help="gyro units (per second)")
        i.add_argument("--imu-time-offset", type=float, default=0.0, help="seconds added to IMU timestamps")
        i.add_argument("--hfov", type=float, default=62.0, help="camera horizontal field of view [deg]")
        i.add_argument("--imu-weight-trans", type=float, default=0.3, help="gyro weight for x/y (0..1)")
        i.add_argument("--imu-weight-rot", type=float, default=0.7, help="gyro weight for rotation (0..1)")
        i.add_argument("--hit-px", type=float, default=25.0, help="vision/gyro disagreement that triggers gyro-only")
        sp.add_argument("--ground-truth", help="ground_truth.csv from make_test_video.py (for plots)")

    po = sub.add_parser("offline", help="stabilize a recorded video (best quality)")
    po.add_argument("input")
    po.add_argument("--start", type=float, default=0.0, help="skip the first N seconds")
    po.add_argument("--end", type=float, default=None, help="stop at N seconds")
    po.add_argument("--show", action="store_true", help="show a window while writing")
    common(po)

    pl = sub.add_parser("live", help="stabilize a live camera feed")
    pl.add_argument("--source", default="0", help="0,1.. | csi-jetson | libcamera | GStreamer pipeline | video file")
    pl.add_argument("--width", type=int, default=640)
    pl.add_argument("--height", type=int, default=480)
    pl.add_argument("--fps", type=int, default=30)
    pl.add_argument("--lookahead", type=int, default=10, help="future frames used = output delay (0 = no delay)")
    pl.add_argument("--no-trend", action="store_true", help="disable lag compensation in live smoothing")
    pl.add_argument("--record", help="also save the raw camera video to this file")
    pl.add_argument("--save", action="store_true", help="save the side-by-side video")
    pl.add_argument("--no-display", action="store_true", help="headless (no window)")
    common(pl)

    args = p.parse_args()
    run_offline(args) if args.command == "offline" else run_live(args)


if __name__ == "__main__":
    main()
