"""
Sliding window manager.

Maintains a buffer of the most recent N keyframes.
When the window is full (or flush is requested) it fires a callback
with the current frame batch for VGGT inference.

Key difference from VGGT-SLAM's submap approach:
  - Window advances by `stride` frames (not a full reset), so consecutive
    windows share `overlap = window_size - stride` frames.
  - This gives VGGT cross-frame context that prevents scale drift at window
    boundaries — each new window "sees" the tail of the previous one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Keyframe:
    index: int          # global keyframe index (monotonically increasing)
    image_rgb: np.ndarray   # HxWx3 uint8
    stamp: float            # ROS time as float seconds


WindowCallback = Callable[[list["Keyframe"]], None]


class SlidingWindow:
    """
    Args:
        window_size:  number of frames passed to VGGT per inference call
        stride:       how many new frames before a new window is triggered
        callback:     called with list[Keyframe] when a window is ready
    """

    def __init__(
        self,
        window_size: int = 16,
        stride: int = 8,
        callback: WindowCallback | None = None,
    ) -> None:
        assert 1 <= stride <= window_size, "stride must be in [1, window_size]"
        self.window_size = window_size
        self.stride = stride
        self.callback = callback

        self._buffer: deque[Keyframe] = deque(maxlen=window_size)
        self._new_since_last: int = 0
        self._global_idx: int = 0

    # ------------------------------------------------------------------

    def add(self, image_rgb: np.ndarray, stamp: float) -> None:
        """Add a new keyframe.  May trigger the callback."""
        kf = Keyframe(index=self._global_idx, image_rgb=image_rgb, stamp=stamp)
        self._global_idx += 1
        self._buffer.append(kf)
        self._new_since_last += 1

        if len(self._buffer) >= self.window_size and self._new_since_last >= self.stride:
            self._fire()

    def flush(self) -> None:
        """Force a callback with whatever is currently in the buffer."""
        if self._buffer and self._new_since_last > 0:
            self._fire()

    def reset(self) -> None:
        self._buffer.clear()
        self._new_since_last = 0
        self._global_idx = 0

    # ------------------------------------------------------------------

    def _fire(self) -> None:
        frames = list(self._buffer)
        self._new_since_last = 0
        if self.callback is not None:
            self.callback(frames)
