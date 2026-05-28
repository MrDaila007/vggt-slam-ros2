"""Unit tests for core/scale_anchor.py."""

import numpy as np

from vggt_slam_ros2.core.scale_anchor import (
    ScaleAnchor,
    _cam_positions,
    _umeyama,
    _apply_sim3_extrinsics,
    _apply_sim3_points,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extrinsics(cam_positions: np.ndarray) -> np.ndarray:
    """Build (S, 3, 4) identity-rotation extrinsics from camera positions."""
    S = cam_positions.shape[0]
    ext = np.zeros((S, 3, 4))
    ext[:, :3, :3] = np.eye(3)
    # p_cam = R @ p_world + t  =>  t = -R @ pos = -pos  (R=I)
    ext[:, :3, 3] = -cam_positions
    return ext


def _make_world_points(S: int, scale: float = 1.0) -> np.ndarray:
    """Random (S, 8, 8, 3) world points."""
    rng = np.random.default_rng(0)
    return (rng.random((S, 8, 8, 3)) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# _cam_positions
# ---------------------------------------------------------------------------

class TestCamPositions:
    def test_identity_rotation(self):
        # With identity rotation: pos = -t
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ext = _make_extrinsics(pos)
        out = _cam_positions(ext)
        np.testing.assert_allclose(out, pos, atol=1e-6)

    def test_90deg_rotation(self):
        # R rotates 90° around Z: x→y, y→-x
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t = np.array([1.0, 0.0, 0.0])
        # pos = -R^T @ t = -R^T @ [1,0,0] = -[0,1,0] = [0,-1,0]
        ext = np.zeros((1, 3, 4))
        ext[0, :3, :3] = R
        ext[0, :3, 3] = t
        pos = _cam_positions(ext)
        expected = (-R.T @ t).reshape(1, 3)
        np.testing.assert_allclose(pos, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# _umeyama
# ---------------------------------------------------------------------------

class TestUmeyama:
    def test_identity(self):
        src = np.eye(3, dtype=float)
        scale, R, t = _umeyama(src, src)
        assert abs(scale - 1.0) < 1e-5
        np.testing.assert_allclose(R, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-5)

    def test_pure_scale(self):
        rng = np.random.default_rng(42)
        src = rng.random((6, 3))
        s = 2.5
        dst = s * src
        scale, R, t = _umeyama(src, dst)
        assert abs(scale - s) < 1e-4
        np.testing.assert_allclose(R, np.eye(3), atol=1e-4)

    def test_scale_rotation_translation(self):
        rng = np.random.default_rng(7)
        src = rng.random((8, 3))
        s = 1.8
        angle = 0.4
        R_gt = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        t_gt = np.array([1.0, -0.5, 0.3])
        dst = s * (R_gt @ src.T).T + t_gt

        scale, R, t = _umeyama(src, dst)
        dst_hat = scale * (R @ src.T).T + t

        assert abs(scale - s) < 1e-4
        np.testing.assert_allclose(dst_hat, dst, atol=1e-4)

    def test_degenerate_static_points(self):
        # All points at the same location — sigma_src ≈ 0, should return scale=1
        src = np.ones((5, 3)) * 2.0
        dst = np.ones((5, 3)) * 5.0
        scale, R, t = _umeyama(src, dst)
        assert scale == 1.0


# ---------------------------------------------------------------------------
# _apply_sim3_extrinsics
# ---------------------------------------------------------------------------

class TestApplySim3Extrinsics:
    def _random_sim3(self, seed=0):
        rng = np.random.default_rng(seed)
        scale = 2.0
        angle = 0.3
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        t = rng.random(3)
        return scale, R, t

    def test_camera_positions_transform_correctly(self):
        """Corrected extrinsic must give pos_global = scale*R@pos_curr + t."""
        scale, R_sim3, t_sim3 = self._random_sim3()
        rng = np.random.default_rng(5)
        pos_curr = rng.random((4, 3))
        ext_curr = _make_extrinsics(pos_curr)

        ext_global = _apply_sim3_extrinsics(ext_curr, scale, R_sim3, t_sim3)
        pos_global = _cam_positions(ext_global)

        expected = scale * (R_sim3 @ pos_curr.T).T + t_sim3
        np.testing.assert_allclose(pos_global, expected, atol=1e-5)

    def test_rotation_is_proper(self):
        """Output rotation matrices should have det ≈ +1."""
        scale, R_sim3, t_sim3 = self._random_sim3(seed=3)
        rng = np.random.default_rng(9)
        pos_curr = rng.random((6, 3))
        ext_curr = _make_extrinsics(pos_curr)

        ext_global = _apply_sim3_extrinsics(ext_curr, scale, R_sim3, t_sim3)
        for i in range(ext_global.shape[0]):
            R_i = ext_global[i, :3, :3]
            assert abs(np.linalg.det(R_i) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# _apply_sim3_points
# ---------------------------------------------------------------------------

class TestApplySim3Points:
    def test_pure_scale(self):
        pts = _make_world_points(3, scale=1.0)
        scale, R_sim3, t_sim3 = 3.0, np.eye(3), np.zeros(3)
        out = _apply_sim3_points(pts, scale, R_sim3, t_sim3)
        np.testing.assert_allclose(out, 3.0 * pts, atol=1e-5)

    def test_shape_preserved(self):
        pts = _make_world_points(5)
        out = _apply_sim3_points(pts, 1.5, np.eye(3), np.zeros(3))
        assert out.shape == pts.shape


# ---------------------------------------------------------------------------
# ScaleAnchor integration
# ---------------------------------------------------------------------------

class TestScaleAnchor:
    def _make_window(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ext = _make_extrinsics(positions)
        S = positions.shape[0]
        pts = _make_world_points(S)
        return ext, pts

    def test_first_window_passthrough(self):
        sa = ScaleAnchor()
        pos = np.random.default_rng(0).random((8, 3))
        ext, pts = self._make_window(pos)
        ext_out, pts_out = sa.process(ext, pts, overlap=4)
        np.testing.assert_array_equal(ext_out, ext)
        np.testing.assert_array_equal(pts_out, pts)

    def test_overlap_below_min_passthrough(self):
        sa = ScaleAnchor(min_overlap=5)
        pos = np.random.default_rng(1).random((8, 3))
        ext, pts = self._make_window(pos)
        # First window
        sa.process(ext, pts, overlap=8)
        # Second window with overlap < min_overlap → passthrough
        pos2 = np.random.default_rng(2).random((8, 3))
        ext2, pts2 = self._make_window(pos2)
        ext_out, pts_out = sa.process(ext2, pts2, overlap=3)
        np.testing.assert_array_equal(ext_out, ext2)

    def test_second_window_overlap_aligned(self):
        """
        Build two synthetic windows that share an overlap region.
        After processing, the overlap frames' global positions should match.
        """
        rng = np.random.default_rng(99)
        overlap = 4
        stride = 4
        total = overlap + stride

        # Window 1: frames 0..7 at positions p0..p7 in frame-1 world
        pos1_global = rng.random((total, 3))
        ext1, pts1 = self._make_window(pos1_global)

        sa = ScaleAnchor(min_overlap=overlap)
        # First window: no prior reference, but pass the future overlap so
        # ScaleAnchor stores the right number of frames for the next window.
        ext1_out, _ = sa.process(ext1, pts1, overlap=overlap)
        _cam_positions(ext1_out)

        # Window 2: the same physical cameras but seen by VGGT in its own frame.
        # The overlap frames (last `overlap` of window 1 = first `overlap` of window 2)
        # are known in global frame as pos1_global[-overlap:].
        # Simulate VGGT outputting them in a rotated+scaled coordinate frame.
        scale_true = 2.3
        angle = 0.5
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        t_true = rng.random(3)

        # Inverse: pos_curr = (pos_global - t_true) / scale_true @ R_true
        pos_global_w2 = np.vstack([
            pos1_global[-overlap:],             # overlap frames (global)
            rng.random((stride, 3)),            # new frames (global)
        ])
        # Map to current window frame (inverse Sim3)
        pos_curr_w2 = (R_true.T @ ((pos_global_w2 - t_true) / scale_true).T).T

        ext2, pts2 = self._make_window(pos_curr_w2)
        ext2_out, _ = sa.process(ext2, pts2, overlap=overlap)
        pos2_out = _cam_positions(ext2_out)

        # The overlap region of window 2, corrected, should match global positions
        np.testing.assert_allclose(
            pos2_out[:overlap],
            pos_global_w2[:overlap],
            atol=1e-3,
        )

    def test_reset_clears_state(self):
        sa = ScaleAnchor()
        pos = np.random.default_rng(0).random((8, 3))
        ext, pts = self._make_window(pos)
        sa.process(ext, pts, overlap=4)
        sa.reset()
        assert sa._prev_overlap_global is None
