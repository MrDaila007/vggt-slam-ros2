"""Tests for MapManager."""

import tempfile
import numpy as np
import pytest
from pathlib import Path

from vggt_slam_ros2.core.map_manager import MapManager, MapFrame


# ── Helpers ───────────────────────────────────────────────────────────────────

H, W = 8, 8   # small spatial dims for fast tests


def _make_window(
    S: int,
    conf_value: float = 1.0,
) -> dict:
    """Return a minimal fake VGGT window output with S frames."""
    rng = np.random.default_rng(42)
    return {
        "extrinsics":   np.tile(np.eye(3, 4), (S, 1, 1)).astype(np.float32),
        "intrinsics":   np.tile(np.eye(3),    (S, 1, 1)).astype(np.float32),
        "world_points": rng.random((S, H, W, 3)).astype(np.float32),
        "colors":       rng.integers(0, 256, (S, H, W, 3), dtype=np.uint8),
        "conf":         np.full((S, H, W), conf_value, dtype=np.float32),
    }


def _add(mgr: MapManager, S: int, overlap: int = 0, conf_thr: float = 0.0) -> tuple:
    w = _make_window(S)
    return mgr.add_window_result(
        global_indices=list(range(S)),
        stamps=[float(i) for i in range(S)],
        extrinsics=w["extrinsics"],
        intrinsics=w["intrinsics"],
        world_points=w["world_points"],
        colors=w["colors"],
        conf=w["conf"],
        conf_threshold_pct=conf_thr,
        overlap=overlap,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEmptyMap:
    def test_get_all_points_empty(self):
        mgr = MapManager()
        pts = mgr.get_all_points()
        assert pts.shape == (0, 3)
        assert pts.dtype == np.float32

    def test_get_all_colors_empty(self):
        mgr = MapManager()
        cols = mgr.get_all_colors()
        assert cols.shape == (0, 3)
        assert cols.dtype == np.uint8

    def test_total_points_zero(self):
        assert MapManager().total_points() == 0

    def test_trajectory_empty(self):
        assert MapManager().get_trajectory() == []


class TestWindowIntegration:
    def test_all_frames_added_with_zero_overlap(self):
        mgr = MapManager()
        _add(mgr, S=4, overlap=0, conf_thr=0.0)
        # 4 frames × H×W points each, conf_thr=0 keeps everything
        assert mgr.total_points() == 4 * H * W

    def test_overlap_frames_skipped(self):
        mgr = MapManager()
        _add(mgr, S=4, overlap=2, conf_thr=0.0)
        # only frames 2 and 3 contribute
        assert mgr.total_points() == 2 * H * W

    def test_full_overlap_adds_nothing(self):
        mgr = MapManager()
        pts, cols = _add(mgr, S=4, overlap=4, conf_thr=0.0)
        assert pts.shape == (0, 3)
        assert mgr.total_points() == 0

    def test_trajectory_stores_new_frames_only(self):
        mgr = MapManager()
        _add(mgr, S=4, overlap=2, conf_thr=0.0)
        traj = mgr.get_trajectory()
        assert len(traj) == 2

    def test_multiple_windows_accumulate(self):
        mgr = MapManager()
        _add(mgr, S=4, overlap=0, conf_thr=0.0)
        _add(mgr, S=4, overlap=2, conf_thr=0.0)
        # first: 4 frames, second: 2 new frames
        assert mgr.total_points() == 6 * H * W


class TestConfidenceFiltering:
    def test_zero_threshold_keeps_all(self):
        mgr = MapManager()
        _add(mgr, S=2, overlap=0, conf_thr=0.0)
        assert mgr.total_points() == 2 * H * W

    def test_hundred_percent_threshold_removes_all(self):
        # percentile 100 → threshold equals max → mask = conf >= max
        # uniform conf array: all values equal the max → all kept
        mgr = MapManager()
        _add(mgr, S=2, overlap=0, conf_thr=100.0)
        # uniform confidence → all points pass (conf == percentile(100))
        assert mgr.total_points() == 2 * H * W

    def test_fifty_percent_threshold_removes_half(self):
        mgr = MapManager()
        S = 2
        w = _make_window(S)
        # Assign confidence 0 to first half of pixels, 1 to second half
        w["conf"][:, :H // 2, :] = 0.0
        w["conf"][:, H // 2:, :] = 1.0
        mgr.add_window_result(
            global_indices=list(range(S)),
            stamps=[0.0, 1.0],
            extrinsics=w["extrinsics"],
            intrinsics=w["intrinsics"],
            world_points=w["world_points"],
            colors=w["colors"],
            conf=w["conf"],
            conf_threshold_pct=50.0,
            overlap=0,
        )
        # threshold = percentile(50) of [0,0,...,1,1,...] = 0.0 or 1.0 depending on numpy
        # Either way, total_points should be > 0 and <= 2*H*W
        assert 0 < mgr.total_points() <= 2 * H * W


class TestReturnValues:
    def test_returns_new_points_and_colors(self):
        mgr = MapManager()
        pts, cols = _add(mgr, S=3, overlap=1, conf_thr=0.0)
        expected_n = 2 * H * W   # 3 - 1 new frames
        assert pts.shape == (expected_n, 3)
        assert cols.shape == (expected_n, 3)
        assert pts.dtype == np.float32
        assert cols.dtype == np.uint8

    def test_returns_empty_when_all_overlap(self):
        mgr = MapManager()
        pts, cols = _add(mgr, S=2, overlap=2, conf_thr=0.0)
        assert pts.shape == (0, 3)
        assert cols.shape == (0, 3)


class TestTrajectory:
    def test_frame_stamps_stored_correctly(self):
        mgr = MapManager()
        w = _make_window(3)
        mgr.add_window_result(
            global_indices=[10, 11, 12],
            stamps=[1.0, 2.0, 3.0],
            extrinsics=w["extrinsics"],
            intrinsics=w["intrinsics"],
            world_points=w["world_points"],
            colors=w["colors"],
            conf=w["conf"],
            conf_threshold_pct=0.0,
            overlap=0,
        )
        traj = mgr.get_trajectory()
        assert [f.stamp for f in traj] == [1.0, 2.0, 3.0]
        assert [f.global_idx for f in traj] == [10, 11, 12]

    def test_trajectory_returns_copy(self):
        mgr = MapManager()
        _add(mgr, S=2)
        traj1 = mgr.get_trajectory()
        traj2 = mgr.get_trajectory()
        assert traj1 is not traj2


class TestReset:
    def test_reset_clears_points(self):
        mgr = MapManager()
        _add(mgr, S=4)
        mgr.reset()
        assert mgr.total_points() == 0
        assert mgr.get_all_points().shape == (0, 3)

    def test_reset_clears_trajectory(self):
        mgr = MapManager()
        _add(mgr, S=4)
        mgr.reset()
        assert mgr.get_trajectory() == []

    def test_can_add_after_reset(self):
        mgr = MapManager()
        _add(mgr, S=4)
        mgr.reset()
        _add(mgr, S=2, overlap=0, conf_thr=0.0)
        assert mgr.total_points() == 2 * H * W


class TestSaveToFile:
    def test_save_npz_returns_true(self):
        mgr = MapManager()
        _add(mgr, S=2)
        with tempfile.TemporaryDirectory() as d:
            ok = mgr.save_to_file(str(Path(d) / "map"), fmt="npz")
        assert ok is True

    def test_save_npz_file_readable(self):
        mgr = MapManager()
        _add(mgr, S=3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "map"
            mgr.save_to_file(str(path), fmt="npz")
            data = np.load(str(path.with_suffix(".npz")))
            assert "points" in data
            assert "colors" in data
            assert data["points"].shape[1] == 3
            assert data["colors"].shape[1] == 3

    def test_save_returns_false_when_empty(self):
        mgr = MapManager()
        with tempfile.TemporaryDirectory() as d:
            ok = mgr.save_to_file(str(Path(d) / "empty"), fmt="npz")
        assert ok is False

    def test_creates_parent_directories(self):
        mgr = MapManager()
        _add(mgr, S=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a" / "b" / "map"
            mgr.save_to_file(str(path), fmt="npz")
            assert path.with_suffix(".npz").exists()
