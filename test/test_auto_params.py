"""Unit tests for utils/auto_params.py (no GPU required)."""

import pytest

from vggt_slam_ros2.utils.auto_params import (
    WindowParams,
    select_window_params,
    _ACTIVATION_PER_FRAME_GB,
    _MODEL_WEIGHT_GB,
)


class TestSelectWindowParams:
    def test_returns_window_params(self):
        p = select_window_params(memory_budget_gb=8.0)
        assert isinstance(p, WindowParams)

    def test_window_within_bounds(self):
        p = select_window_params(memory_budget_gb=8.0, min_window=4, max_window=32)
        assert 4 <= p.window_size <= 32

    def test_stride_positive(self):
        p = select_window_params(memory_budget_gb=8.0)
        assert p.stride >= 1

    def test_overlap_equals_window_minus_stride(self):
        p = select_window_params(memory_budget_gb=8.0)
        assert p.overlap == p.window_size - p.stride

    def test_min_window_respected_on_small_memory(self):
        p = select_window_params(memory_budget_gb=1.0, min_window=4)
        assert p.window_size >= 4

    def test_max_window_respected_on_large_memory(self):
        p = select_window_params(memory_budget_gb=100.0, max_window=32)
        assert p.window_size <= 32

    def test_peak_estimate_reasonable(self):
        p = select_window_params(memory_budget_gb=16.0)
        expected_peak = _MODEL_WEIGHT_GB + p.window_size * _ACTIVATION_PER_FRAME_GB
        assert abs(p.estimated_peak_gb - expected_peak) < 0.01

    def test_larger_memory_gives_larger_window(self):
        p_small = select_window_params(memory_budget_gb=4.0)
        p_large = select_window_params(memory_budget_gb=16.0)
        assert p_large.window_size >= p_small.window_size

    def test_overlap_ratio_respected(self):
        ratio = 0.5
        p = select_window_params(memory_budget_gb=8.0, target_overlap_ratio=ratio)
        actual_ratio = p.overlap / p.window_size
        assert abs(actual_ratio - ratio) <= 0.3  # approximate due to rounding

    def test_safety_margin_applied(self):
        budget = 8.0
        p_safe = select_window_params(memory_budget_gb=budget, safety_margin=0.5)
        p_full = select_window_params(memory_budget_gb=budget, safety_margin=1.0)
        assert p_safe.window_size <= p_full.window_size
