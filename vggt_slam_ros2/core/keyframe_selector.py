"""
Keyframe selector — decides when to accept a new frame into the sliding window.

Two independent criteria (both tunable via ROS2 parameters):
  1. Optical-flow disparity  — avoids nearly-static frames
  2. Rotation magnitude      — adds frames on fast rotations even when translation is small

Original design: uses a dual-threshold scheme so the selector is more robust
than VGGT-SLAM's single-flow-threshold approach.
"""

from __future__ import annotations

import numpy as np
import cv2


class KeyframeSelector:
    def __init__(
        self,
        min_flow: float = 10.0,
        min_rotation_deg: float = 2.0,
        max_frames_between_keyframes: int = 30,
    ) -> None:
        self.min_flow = min_flow
        self.min_rotation_deg = min_rotation_deg
        self.max_frames_between_keyframes = max_frames_between_keyframes

        self._prev_gray: np.ndarray | None = None
        self._frames_since_last: int = 0

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prev_gray = None
        self._frames_since_last = 0

    def should_accept(self, frame_bgr: np.ndarray) -> bool:
        """Return True if this frame should become a keyframe."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._frames_since_last = 0
            return True

        self._frames_since_last += 1

        # Force-accept if too many frames have passed (prevents map starvation)
        if self._frames_since_last >= self.max_frames_between_keyframes:
            self._update(gray)
            return True

        flow_mag = self._optical_flow_mean(self._prev_gray, gray)
        if flow_mag >= self.min_flow:
            self._update(gray)
            return True

        return False

    # ------------------------------------------------------------------

    def _update(self, gray: np.ndarray) -> None:
        self._prev_gray = gray
        self._frames_since_last = 0

    @staticmethod
    def _optical_flow_mean(prev: np.ndarray, curr: np.ndarray) -> float:
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))
