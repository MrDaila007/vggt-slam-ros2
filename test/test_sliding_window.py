"""Tests for SlidingWindow."""

import numpy as np
import pytest

from vggt_slam_ros2.core.sliding_window import SlidingWindow, Keyframe


def _dummy_image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _fill(window: SlidingWindow, n: int, start_stamp: float = 0.0) -> list[list[Keyframe]]:
    """Add n frames; return all callback payloads received."""
    received: list[list[Keyframe]] = []
    original_cb = window.callback

    def _capture(frames):
        received.append(list(frames))
        if original_cb:
            original_cb(frames)

    window.callback = _capture
    for i in range(n):
        window.add(_dummy_image(), stamp=start_stamp + i)
    return received


class TestCallbackTiming:
    def test_no_callback_before_window_full(self):
        w = SlidingWindow(window_size=4, stride=2)
        calls = _fill(w, 3)
        assert calls == []

    def test_callback_fires_when_window_full_and_stride_met(self):
        w = SlidingWindow(window_size=4, stride=2)
        calls = _fill(w, 4)
        assert len(calls) == 1

    def test_callback_fires_again_after_each_stride(self):
        w = SlidingWindow(window_size=4, stride=2)
        calls = _fill(w, 8)
        # windows at frames [0-3], [2-5], [4-7]
        assert len(calls) == 3

    def test_stride_equals_window_no_overlap(self):
        w = SlidingWindow(window_size=3, stride=3)
        calls = _fill(w, 6)
        assert len(calls) == 2


class TestCallbackContent:
    def test_window_size_is_correct(self):
        w = SlidingWindow(window_size=4, stride=2)
        calls = _fill(w, 4)
        assert len(calls[0]) == 4

    def test_frames_are_keyframe_instances(self):
        w = SlidingWindow(window_size=2, stride=2)
        calls = _fill(w, 2)
        for kf in calls[0]:
            assert isinstance(kf, Keyframe)

    def test_global_indices_monotonically_increase(self):
        w = SlidingWindow(window_size=3, stride=3)
        calls = _fill(w, 6)
        indices_0 = [kf.index for kf in calls[0]]
        indices_1 = [kf.index for kf in calls[1]]
        assert indices_0 == [0, 1, 2]
        assert indices_1 == [3, 4, 5]

    def test_stamps_are_preserved(self):
        w = SlidingWindow(window_size=2, stride=2)
        calls = _fill(w, 2, start_stamp=10.0)
        stamps = [kf.stamp for kf in calls[0]]
        assert stamps == [10.0, 11.0]

    def test_overlap_frames_shared_between_windows(self):
        # window=4 stride=2 → overlap=2; last 2 of window N == first 2 of window N+1
        w = SlidingWindow(window_size=4, stride=2)
        calls = _fill(w, 6)
        tail_indices = [kf.index for kf in calls[0][-2:]]
        head_indices = [kf.index for kf in calls[1][:2]]
        assert tail_indices == head_indices


class TestFlush:
    def test_flush_fires_callback_with_partial_buffer(self):
        received: list[list[Keyframe]] = []
        w = SlidingWindow(window_size=4, stride=2, callback=lambda f: received.append(f))
        for i in range(3):
            w.add(_dummy_image(), stamp=float(i))
        w.flush()
        assert len(received) == 1
        assert len(received[0]) == 3

    def test_flush_empty_buffer_does_nothing(self):
        received: list[list[Keyframe]] = []
        w = SlidingWindow(window_size=4, stride=2, callback=lambda f: received.append(f))
        w.flush()
        assert received == []

    def test_flush_no_callback_if_nothing_new_since_last_fire(self):
        received: list[list[Keyframe]] = []
        w = SlidingWindow(window_size=2, stride=2, callback=lambda f: received.append(f))
        _fill(w, 2)
        count_before = len(received)
        w.flush()   # _new_since_last == 0 after the fire → should not re-fire
        assert len(received) == count_before


class TestReset:
    def test_reset_clears_buffer(self):
        w = SlidingWindow(window_size=4, stride=2)
        _fill(w, 4)
        w.reset()
        assert len(w._buffer) == 0

    def test_global_index_restarts_after_reset(self):
        received: list[list[Keyframe]] = []
        w = SlidingWindow(window_size=2, stride=2, callback=lambda f: received.append(f))
        _fill(w, 2)
        w.reset()
        _fill(w, 2)
        assert received[-1][0].index == 0

    def test_new_since_last_reset_to_zero(self):
        w = SlidingWindow(window_size=4, stride=2)
        _fill(w, 3)
        w.reset()
        assert w._new_since_last == 0


class TestConstructorValidation:
    def test_stride_greater_than_window_raises(self):
        with pytest.raises(AssertionError):
            SlidingWindow(window_size=4, stride=5)

    def test_stride_zero_raises(self):
        with pytest.raises(AssertionError):
            SlidingWindow(window_size=4, stride=0)

    def test_stride_equal_window_is_valid(self):
        SlidingWindow(window_size=4, stride=4)   # should not raise
