"""
Unit tests for core/pose_graph.py.

Pure SE(3) helper functions are tested directly (no GTSAM required).
PoseGraph itself is tested with a mock GTSAM that mimics the real API
so the tests run without GTSAM installed on the host.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from vggt_slam_ros2.core.pose_graph import (
    extrinsic_to_world,
    world_to_extrinsic,
    relative_pose,
)


# ===========================================================================
# SE(3) helpers — no GTSAM required
# ===========================================================================

class TestExtrinsicToWorld:
    def test_identity(self):
        ext = np.eye(3, 4, dtype=np.float64)
        T = extrinsic_to_world(ext)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-9)

    def test_roundtrip(self):
        R = _rot_z(0.5)
        t_world = np.array([1.0, 2.0, 3.0])
        # build cam-from-world extrinsic
        ext = np.zeros((3, 4), dtype=np.float64)
        ext[:3, :3] = R.T
        ext[:3, 3] = -R.T @ t_world
        T = extrinsic_to_world(ext)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-9)
        np.testing.assert_allclose(T[:3, 3], t_world, atol=1e-9)


class TestWorldToExtrinsic:
    def test_identity(self):
        T = np.eye(4, dtype=np.float64)
        ext = world_to_extrinsic(T)
        np.testing.assert_allclose(ext, np.eye(3, 4), atol=1e-9)

    def test_roundtrip(self):
        R = _rot_z(1.2)
        t_world = np.array([0.5, -1.0, 2.0])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t_world

        ext = world_to_extrinsic(T)
        T_back = extrinsic_to_world(ext)

        np.testing.assert_allclose(T_back, T, atol=1e-9)


class TestRelativePose:
    def test_identity_relative(self):
        T = _random_se3(seed=0)
        T_rel = relative_pose(T, T)
        np.testing.assert_allclose(T_rel, np.eye(4), atol=1e-9)

    def test_compose(self):
        T_a = _random_se3(seed=1)
        T_b = _random_se3(seed=2)
        T_ab = relative_pose(T_a, T_b)
        # T_a @ T_ab should reconstruct T_b
        np.testing.assert_allclose(T_a @ T_ab, T_b, atol=1e-9)


# ===========================================================================
# PoseGraph — using a mock GTSAM
# ===========================================================================

@pytest.fixture
def mock_gtsam_module():
    """Patch gtsam in pose_graph and return a namespace of mocks."""
    mock_gtsam = MagicMock()

    # symbol: return a unique integer key for each (char, idx) pair
    def _symbol(char, idx):
        return hash((char, idx))
    mock_gtsam.symbol.side_effect = _symbol

    # Noise models
    mock_noise = MagicMock()
    mock_gtsam.noiseModel.Diagonal.Sigmas.return_value = mock_noise

    # Pose3: just wrap the numpy array for comparison
    class _MockPose3:
        def __init__(self, R=None, pt=None, T=None):
            self.T = T if T is not None else np.eye(4)

        def rotation(self):
            m = MagicMock()
            m.matrix.return_value = self.T[:3, :3]
            return m

        def translation(self):
            return self.T[:3, 3]

    def _make_rot3(R):
        m = MagicMock()
        m.matrix.return_value = R
        return m

    mock_gtsam.Rot3.side_effect = lambda R: _make_rot3(R)
    mock_gtsam.Point3.side_effect = lambda x, y, z: np.array([x, y, z])
    mock_gtsam.Pose3.side_effect = lambda rot, pt: MagicMock()

    # Values
    values = MagicMock()
    def _at_pose3(key):
        # Return an identity-ish pose
        p = MagicMock()
        p.rotation().matrix.return_value = np.eye(3)
        p.translation.return_value = np.zeros(3)
        return p
    values.atPose3.side_effect = _at_pose3
    mock_gtsam.Values.return_value = values

    # Graph
    graph = MagicMock()
    mock_gtsam.NonlinearFactorGraph.return_value = graph

    # Optimizer
    optimizer = MagicMock()
    optimizer.optimize.return_value = values
    mock_gtsam.LevenbergMarquardtOptimizer.return_value = optimizer
    mock_gtsam.LevenbergMarquardtParams.return_value = MagicMock()

    return mock_gtsam


@pytest.fixture
def pose_graph(mock_gtsam_module):
    """Return a PoseGraph backed by the mock gtsam."""
    with patch.dict('sys.modules', {'gtsam': mock_gtsam_module}):
        import importlib
        import vggt_slam_ros2.core.pose_graph as pg_mod
        importlib.reload(pg_mod)

        pg = pg_mod.PoseGraph.__new__(pg_mod.PoseGraph)
        # Bypass _check_gtsam
        pg._graph = mock_gtsam_module.NonlinearFactorGraph()
        pg._initial = mock_gtsam_module.Values()
        pg._poses = []
        return pg, pg_mod


class TestPoseGraphInit:
    def test_pose_count_starts_at_zero(self, pose_graph):
        pg, _ = pose_graph
        assert pg.pose_count == 0


class TestAddPose:
    def test_first_pose_increments_count(self, pose_graph):
        pg, pg_mod = pose_graph
        T = np.eye(4)
        pg.add_pose(T)
        assert pg.pose_count == 1

    def test_multiple_poses_increment_count(self, pose_graph):
        pg, _ = pose_graph
        for i in range(5):
            pg.add_pose(np.eye(4))
        assert pg.pose_count == 5

    def test_returns_correct_index(self, pose_graph):
        pg, _ = pose_graph
        idx0 = pg.add_pose(np.eye(4))
        idx1 = pg.add_pose(np.eye(4))
        assert idx0 == 0
        assert idx1 == 1

    def test_first_pose_adds_prior(self, pose_graph):
        pg, pg_mod = pose_graph
        pg.add_pose(np.eye(4))
        assert pg._graph.add.call_count == 1
        args = pg._graph.add.call_args[0]
        # Should be a PriorFactorPose3
        assert pg_mod.gtsam.PriorFactorPose3.called

    def test_second_pose_adds_between(self, pose_graph):
        pg, pg_mod = pose_graph
        pg.add_pose(np.eye(4))
        pg.add_pose(np.eye(4))
        assert pg._graph.add.call_count == 2
        assert pg_mod.gtsam.BetweenFactorPose3.called


class TestAddLoop:
    def test_loop_adds_to_graph(self, pose_graph):
        pg, pg_mod = pose_graph
        pg.add_pose(np.eye(4))
        pg.add_pose(np.eye(4))
        count_before = pg._graph.add.call_count
        pg.add_loop(0, 1, np.eye(4))
        assert pg._graph.add.call_count == count_before + 1


class TestOptimize:
    def test_optimize_returns_dict(self, pose_graph):
        pg, _ = pose_graph
        pg.add_pose(np.eye(4))
        pg.add_pose(np.eye(4))
        result = pg.optimize()
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1}

    def test_optimize_calls_lm(self, pose_graph):
        pg, pg_mod = pose_graph
        pg.add_pose(np.eye(4))
        pg.optimize()
        assert pg_mod.gtsam.LevenbergMarquardtOptimizer.called


# ===========================================================================
# Helpers
# ===========================================================================

def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _random_se3(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = rng.random() * np.pi
    R = _rot_z(angle)
    t = rng.random(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
