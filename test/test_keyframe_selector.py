"""Tests for KeyframeSelector."""

import numpy as np
import pytest

from vggt_slam_ros2.core.keyframe_selector import KeyframeSelector


def _solid_bgr(h: int = 64, w: int = 64, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _shifted_frame(source: np.ndarray, offset: int = 10) -> np.ndarray:
    """Shift a frame horizontally — guarantees measurable Farneback flow."""
    return np.roll(source, offset, axis=1).copy()


class TestFirstFrame:
    def test_always_accepted(self):
        sel = KeyframeSelector(min_flow=5.0)
        assert sel.should_accept(_solid_bgr()) is True

    def test_state_after_first(self):
        sel = KeyframeSelector(min_flow=5.0)
        sel.should_accept(_solid_bgr())
        assert sel._prev_gray is not None
        assert sel._frames_since_last == 0


class TestFlowThreshold:
    def test_identical_frame_rejected(self):
        sel = KeyframeSelector(min_flow=5.0, max_frames_between_keyframes=100)
        frame = _solid_bgr()
        sel.should_accept(frame)          # first frame — accepted
        assert sel.should_accept(frame) is False

    def test_high_flow_frame_accepted(self):
        rng = np.random.default_rng(99)
        frame1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        frame2 = _shifted_frame(frame1, offset=10)   # 10 px shift → clear motion
        sel = KeyframeSelector(min_flow=0.1, max_frames_between_keyframes=100)
        sel.should_accept(frame1)
        assert sel.should_accept(frame2) is True

    def test_counter_increments_on_rejection(self):
        sel = KeyframeSelector(min_flow=999.0, max_frames_between_keyframes=100)
        frame = _solid_bgr()
        sel.should_accept(frame)
        sel.should_accept(frame)   # rejected
        assert sel._frames_since_last == 1
        sel.should_accept(frame)   # rejected again
        assert sel._frames_since_last == 2

    def test_counter_resets_on_acceptance(self):
        rng = np.random.default_rng(99)
        frame1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        frame2 = _shifted_frame(frame1, offset=10)
        sel = KeyframeSelector(min_flow=0.1, max_frames_between_keyframes=100)
        sel.should_accept(frame1)
        sel.should_accept(frame2)   # accepted via flow
        assert sel._frames_since_last == 0


class TestMaxGap:
    def test_force_accept_at_max_gap(self):
        max_gap = 5
        sel = KeyframeSelector(min_flow=999.0, max_frames_between_keyframes=max_gap)
        frame = _solid_bgr()
        sel.should_accept(frame)                  # first — accepted

        for _ in range(max_gap - 1):              # below threshold
            assert sel.should_accept(frame) is False

        assert sel.should_accept(frame) is True   # max_gap reached

    def test_counter_resets_after_forced_accept(self):
        sel = KeyframeSelector(min_flow=999.0, max_frames_between_keyframes=3)
        frame = _solid_bgr()
        sel.should_accept(frame)
        sel.should_accept(frame)
        sel.should_accept(frame)
        sel.should_accept(frame)   # forced accept
        assert sel._frames_since_last == 0


class TestReset:
    def test_reset_accepts_next_as_first(self):
        sel = KeyframeSelector(min_flow=999.0, max_frames_between_keyframes=100)
        frame = _solid_bgr()
        sel.should_accept(frame)
        sel.should_accept(frame)   # rejected
        sel.reset()
        assert sel.should_accept(frame) is True

    def test_reset_clears_internal_state(self):
        sel = KeyframeSelector()
        sel.should_accept(_solid_bgr())
        sel.reset()
        assert sel._prev_gray is None
        assert sel._frames_since_last == 0
