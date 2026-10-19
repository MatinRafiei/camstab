"""Vision-based frame-to-frame motion estimation.

Pipeline per frame (classic "feature tracking" stabilization):
  1. Find good corners in the previous frame       (cv2.goodFeaturesToTrack)
  2. Track them into the current frame             (cv2.calcOpticalFlowPyrLK)
  3. Fit a rigid motion (shift + rotation) robustly (cv2.estimateAffinePartial2D + RANSAC)

Everything runs on a down-scaled grayscale copy to stay fast on a Jetson / Raspberry Pi.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Motion:
    """How the image content moved from the previous frame to this one.

    dx, dy : shift of the image centre, in full-resolution pixels (+x right, +y down)
    da     : in-plane rotation in radians (positive = clockwise on screen)
    ok     : False when tracking failed (motion blur, no texture, sudden hit)
    """
    dx: float = 0.0
    dy: float = 0.0
    da: float = 0.0
    ok: bool = True
    inliers: int = 0

    def as_array(self):
        return np.array([self.dx, self.dy, self.da], dtype=np.float64)


class MotionEstimator:
    def __init__(self, process_width=480, max_corners=400, quality=0.001,
                 min_distance=10, min_inliers=15, ransac_thresh=3.0, enhance=True, keyframe=False):
        self.process_width = process_width
        self.keyframe = keyframe
        self.key_gray = None
        # CLAHE boosts local contrast so low-texture images (IR/thermal, fog, walls)
        # still give enough corners to track.
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if enhance else None
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.prev_gray = None
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def _prepare(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        scale = min(1.0, self.process_width / gray.shape[1])
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if self.clahe is not None:
            gray = self.clahe.apply(gray)
        return gray, scale

    def reset(self):
        self.prev_gray = None
        self.key_gray = None

    # ------------------------------------------------------------------ helpers
    def _detect(self, gray):
        return cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners, qualityLevel=self.quality,
                                       minDistance=self.min_distance, blockSize=5)

    def _fit(self, src, dst, pts, init=None):
        """Track pts from src to dst and fit a rigid motion. Returns (2x3 matrix, inliers) or (None, 0)."""
        if pts is None or len(pts) < self.min_inliers:
            return None, 0
        if init is not None:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, init.copy(),
                                                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params)
        else:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, **self.lk_params)
        good = status.reshape(-1).astype(bool)
        p0, p1 = pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
        if len(p0) < self.min_inliers:
            return None, 0
        M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                             ransacReprojThreshold=self.ransac_thresh,
                                             maxIters=2000, confidence=0.99)
        if M is None or inl is None or int(inl.sum()) < self.min_inliers:
            return None, 0
        return M, int(inl.sum())

    def _pose(self, M, shape, scale):
        """Matrix -> (dx, dy, da): rotation about the image centre + shift of the centre, full-res px."""
        h, w = shape
        cx, cy = w / 2.0, h / 2.0
        da = float(np.arctan2(M[1, 0], M[0, 0]))
        dx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
        dy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
        return np.array([dx / scale, dy / scale, da])

    def _new_key(self, gray):
        self.key_gray, self.key_pts = gray, self._detect(gray)
        self.key_M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        self.key_pose = np.zeros(3)
        self.key_ref = None

    # ------------------------------------------------------------------ main
    def estimate(self, frame) -> Motion:
        gray, scale = self._prepare(frame)
        prev, self.prev_gray = self.prev_gray, gray
        if prev is None:                      # first frame: nothing to compare with
            if self.keyframe:
                self._new_key(gray)
            return Motion()

        # 1) Keyframe tracking: measure this frame against a fixed reference frame.
        #    Errors then don't add up from frame to frame (no slow drift).
        if self.keyframe and self.key_gray is not None and self.key_pts is not None:
            pts = self.key_pts
            init = cv2.transform(pts, self.key_M).astype(np.float32)   # predict from last frame
            M, n = self._fit(self.key_gray, gray, pts, init)
            ref = self.key_ref or n
            if M is not None and n >= max(self.min_inliers, 0.3 * ref):
                pose = self._pose(M, gray.shape, scale)
                step = pose - self.key_pose
                self.key_M, self.key_pose = M, pose
                if self.key_ref is None:
                    self.key_ref = n
                if n < 0.6 * ref:             # reference getting stale (scene moved away) -> refresh
                    self._new_key(gray)
                return Motion(*step, True, n)

        # 2) Fallback: classic frame-to-frame tracking, then start a new keyframe here.
        M, n = self._fit(prev, gray, self._detect(prev))
        if self.keyframe:
            self._new_key(gray)
        if M is None:
            return Motion(ok=False)
        return Motion(*self._pose(M, gray.shape, scale), True, n)
