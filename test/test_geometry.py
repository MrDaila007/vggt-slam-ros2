"""Tests for geometry utilities."""

import numpy as np
import pytest

from vggt_slam_ros2.utils.geometry import (
    se3_inverse,
    transform_points,
    filter_points_by_confidence,
    sliding_window_indices,
    relative_rotation_angle,
    normalize_scale,
)


# ── SE3 inverse ───────────────────────────────────────────────────────────────

class TestSE3Inverse:
    def test_identity_inverse_is_identity(self):
        T = np.eye(4, dtype=np.float64)
        T_inv = se3_inverse(T)
        np.testing.assert_allclose(T_inv, np.eye(4), atol=1e-12)

    def test_roundtrip_is_identity(self):
        rng = np.random.default_rng(1)
        # Build a valid SE3: random rotation via QR decomposition
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Q
        T[:3, 3] = rng.standard_normal(3)

        result = T @ se3_inverse(T)
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)

    def test_translation_only(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [1.0, 2.0, 3.0]
        T_inv = se3_inverse(T)
        np.testing.assert_allclose(T_inv[:3, 3], [-1.0, -2.0, -3.0], atol=1e-12)
        np.testing.assert_allclose(T_inv[:3, :3], np.eye(3), atol=1e-12)


# ── transform_points ──────────────────────────────────────────────────────────

class TestTransformPoints:
    def test_identity_preserves_points(self):
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = transform_points(pts, np.eye(4))
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_pure_translation(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        out = transform_points(pts, T)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0], atol=1e-12)
        np.testing.assert_allclose(out[1], [2.0, 2.0, 3.0], atol=1e-12)

    def test_output_shape(self):
        pts = np.random.rand(100, 3)
        out = transform_points(pts, np.eye(4))
        assert out.shape == (100, 3)

    def test_90deg_rotation_around_z(self):
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        pts = np.array([[1.0, 0.0, 0.0]])
        out = transform_points(pts, T)
        np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-12)


# ── filter_points_by_confidence ───────────────────────────────────────────────

class TestFilterByConfidence:
    def _make(self, n=100):
        rng = np.random.default_rng(7)
        pts = rng.random((n, 3)).astype(np.float32)
        cols = rng.integers(0, 256, (n, 3), dtype=np.uint8)
        conf = np.arange(n, dtype=np.float32)   # 0..99
        return pts, cols, conf

    def test_zero_threshold_keeps_all(self):
        pts, cols, conf = self._make(50)
        out_pts, out_cols = filter_points_by_confidence(pts, cols, conf, 0.0)
        assert out_pts.shape[0] == 50

    def test_50pct_threshold_keeps_top_half(self):
        pts, cols, conf = self._make(100)
        out_pts, _ = filter_points_by_confidence(pts, cols, conf, 50.0)
        # percentile(50) of 0..99 = 49.5 → keep conf >= 49.5 → 50 points
        assert out_pts.shape[0] == 50

    def test_shapes_match_after_filter(self):
        pts, cols, conf = self._make(100)
        out_pts, out_cols = filter_points_by_confidence(pts, cols, conf, 30.0)
        assert out_pts.shape[0] == out_cols.shape[0]

    def test_100pct_keeps_only_max(self):
        pts, cols, conf = self._make(100)
        out_pts, _ = filter_points_by_confidence(pts, cols, conf, 100.0)
        assert out_pts.shape[0] >= 1


# ── sliding_window_indices ────────────────────────────────────────────────────

class TestSlidingWindowIndices:
    def test_all_indices_covered(self):
        windows = sliding_window_indices(total=10, window=4, stride=2)
        all_seen = set()
        for w in windows:
            all_seen.update(w)
        assert all_seen == set(range(10))

    def test_last_index_always_present(self):
        for total in range(5, 20):
            windows = sliding_window_indices(total=total, window=4, stride=3)
            last_window = windows[-1]
            assert (total - 1) in last_window

    def test_window_size_respected(self):
        windows = sliding_window_indices(total=20, window=5, stride=3)
        # All windows except possibly the last have size == window
        for w in windows[:-1]:
            assert len(w) == 5

    def test_single_window_when_total_equals_window(self):
        windows = sliding_window_indices(total=4, window=4, stride=2)
        assert len(windows) == 1
        assert windows[0] == [0, 1, 2, 3]

    def test_consecutive_windows_overlap_by_stride(self):
        windows = sliding_window_indices(total=10, window=4, stride=2)
        for i in range(len(windows) - 1):
            overlap = set(windows[i]) & set(windows[i + 1])
            assert len(overlap) == 2   # window - stride = 4 - 2


# ── relative_rotation_angle ───────────────────────────────────────────────────

class TestRelativeRotationAngle:
    def test_same_rotation_is_zero(self):
        R = np.eye(3)
        assert relative_rotation_angle(R, R) == pytest.approx(0.0, abs=1e-10)

    def test_90_degree_rotation(self):
        R1 = np.eye(3)
        R2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        angle = relative_rotation_angle(R1, R2)
        assert angle == pytest.approx(90.0, abs=1e-8)

    def test_180_degree_rotation(self):
        R1 = np.eye(3)
        R2 = np.diag([-1.0, -1.0, 1.0])
        angle = relative_rotation_angle(R1, R2)
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_symmetry(self):
        rng = np.random.default_rng(3)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        angle_ab = relative_rotation_angle(np.eye(3), Q)
        angle_ba = relative_rotation_angle(Q, np.eye(3))
        assert angle_ab == pytest.approx(angle_ba, abs=1e-10)


# ── normalize_scale ───────────────────────────────────────────────────────────

class TestNormalizeScale:
    def _straight_line_poses(self, n: int, step: float) -> np.ndarray:
        """N poses moving along x-axis with given step size."""
        poses = np.tile(np.eye(3, 4), (n, 1, 1)).astype(np.float64)
        for i in range(n):
            poses[i, 0, 3] = i * step
        return poses

    def test_unit_step_no_change(self):
        poses = self._straight_line_poses(5, step=1.0)
        scaled, factor = normalize_scale(poses)
        assert factor == pytest.approx(1.0, abs=1e-10)
        np.testing.assert_allclose(scaled, poses, atol=1e-10)

    def test_scale_factor_applied(self):
        poses = self._straight_line_poses(5, step=2.0)
        scaled, factor = normalize_scale(poses)
        assert factor == pytest.approx(0.5, abs=1e-10)
        # After scaling, step should be 1.0
        for i in range(1, 5):
            assert scaled[i, 0, 3] == pytest.approx(float(i), abs=1e-10)

    def test_stationary_trajectory_returns_one(self):
        poses = self._straight_line_poses(5, step=0.0)
        _, factor = normalize_scale(poses)
        assert factor == pytest.approx(1.0, abs=1e-10)
