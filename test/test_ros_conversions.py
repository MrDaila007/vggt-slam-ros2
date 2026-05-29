"""
Tests for ros_conversions.py.

Skipped automatically when ROS2 / rclpy is not installed
(e.g. in a plain Python virtualenv without ROS2 sourced).
"""

import numpy as np
import pytest

# Skip the entire module if ROS2 message types are unavailable.
# This allows running `pytest test/` in a non-ROS2 environment.
geometry_msgs = pytest.importorskip("geometry_msgs")
sensor_msgs = pytest.importorskip("sensor_msgs")
builtin_interfaces = pytest.importorskip("builtin_interfaces")

from builtin_interfaces.msg import Time  # noqa: E402
from vggt_slam_ros2.utils.ros_conversions import (  # noqa: E402
    _rot_to_quat,
    stamp_to_float,
    numpy_to_pointcloud2,
    extrinsic_to_transform,
    extrinsic_to_pose_stamped,
)


def _make_stamp(sec: int = 1, nanosec: int = 500_000_000) -> Time:
    t = Time()
    t.sec = sec
    t.nanosec = nanosec
    return t


# ── _rot_to_quat ──────────────────────────────────────────────────────────────

class TestRotToQuat:
    def test_identity_gives_unit_quaternion(self):
        w, x, y, z = _rot_to_quat(np.eye(3))
        assert w == pytest.approx(1.0, abs=1e-10)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_quaternion_is_unit_norm(self):
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        w, x, y, z = _rot_to_quat(Q)
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_90deg_rotation_around_z(self):
        # R_z(90°): x→y, y→-x
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        w, x, y, z = _rot_to_quat(R)
        # Expected: (w=cos45°, x=0, y=0, z=sin45°)
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        assert w == pytest.approx(c, abs=1e-8)
        assert x == pytest.approx(0.0, abs=1e-8)
        assert y == pytest.approx(0.0, abs=1e-8)
        assert z == pytest.approx(s, abs=1e-8)

    def test_all_four_branches(self):
        # Force each branch of the if/elif chain with known rotations
        rotations = [
            np.eye(3),                                                      # trace > 0
            np.diag([1.0, -1.0, -1.0]),                                    # R[0,0] dominant
            np.diag([-1.0, 1.0, -1.0]),                                    # R[1,1] dominant
            np.diag([-1.0, -1.0, 1.0]),                                    # R[2,2] dominant
        ]
        for R in rotations:
            w, x, y, z = _rot_to_quat(R)
            norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
            assert norm == pytest.approx(1.0, abs=1e-8), f"Failed for R=\n{R}"


# ── stamp_to_float ────────────────────────────────────────────────────────────

class TestStampToFloat:
    def test_integer_seconds(self):
        t = _make_stamp(sec=5, nanosec=0)
        assert stamp_to_float(t) == pytest.approx(5.0)

    def test_fractional_seconds(self):
        t = _make_stamp(sec=1, nanosec=500_000_000)
        assert stamp_to_float(t) == pytest.approx(1.5, abs=1e-9)

    def test_zero_stamp(self):
        t = _make_stamp(sec=0, nanosec=0)
        assert stamp_to_float(t) == pytest.approx(0.0)


# ── numpy_to_pointcloud2 ──────────────────────────────────────────────────────

class TestNumpyToPointCloud2:
    def _make_pc(self, n: int = 10):
        rng = np.random.default_rng(0)
        pts = rng.random((n, 3)).astype(np.float32)
        cols = rng.integers(0, 256, (n, 3), dtype=np.uint8)
        return pts, cols

    def test_width_equals_n_points(self):
        pts, cols = self._make_pc(25)
        msg = numpy_to_pointcloud2(pts, cols, "map", _make_stamp())
        assert msg.width == 25

    def test_height_is_one(self):
        pts, cols = self._make_pc(10)
        msg = numpy_to_pointcloud2(pts, cols, "map", _make_stamp())
        assert msg.height == 1

    def test_point_step_is_16(self):
        pts, cols = self._make_pc(5)
        msg = numpy_to_pointcloud2(pts, cols, "map", _make_stamp())
        assert msg.point_step == 16

    def test_data_length_matches(self):
        n = 15
        pts, cols = self._make_pc(n)
        msg = numpy_to_pointcloud2(pts, cols, "map", _make_stamp())
        assert len(msg.data) == n * 16

    def test_frame_id_stored(self):
        pts, cols = self._make_pc(3)
        msg = numpy_to_pointcloud2(pts, cols, "my_frame", _make_stamp())
        assert msg.header.frame_id == "my_frame"

    def test_field_names(self):
        pts, cols = self._make_pc(3)
        msg = numpy_to_pointcloud2(pts, cols, "map", _make_stamp())
        names = [f.name for f in msg.fields]
        assert names == ["x", "y", "z", "rgb"]

    def test_mismatched_lengths_raise(self):
        pts = np.zeros((10, 3), dtype=np.float32)
        cols = np.zeros((5, 3), dtype=np.uint8)
        with pytest.raises(AssertionError):
            numpy_to_pointcloud2(pts, cols, "map", _make_stamp())


# ── extrinsic_to_transform ────────────────────────────────────────────────────

class TestExtrinsicToTransform:
    def test_identity_extrinsic_gives_zero_translation(self):
        ext = np.eye(3, 4, dtype=np.float64)
        ts = extrinsic_to_transform(ext, "map", "camera", _make_stamp())
        assert ts.transform.translation.x == pytest.approx(0.0, abs=1e-10)
        assert ts.transform.translation.y == pytest.approx(0.0, abs=1e-10)
        assert ts.transform.translation.z == pytest.approx(0.0, abs=1e-10)

    def test_identity_extrinsic_gives_unit_quaternion(self):
        ext = np.eye(3, 4, dtype=np.float64)
        ts = extrinsic_to_transform(ext, "map", "camera", _make_stamp())
        q = ts.transform.rotation
        norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert norm == pytest.approx(1.0, abs=1e-10)
        assert q.w == pytest.approx(1.0, abs=1e-10)

    def test_frame_ids_stored(self):
        ext = np.eye(3, 4, dtype=np.float64)
        ts = extrinsic_to_transform(ext, "parent", "child", _make_stamp())
        assert ts.header.frame_id == "parent"
        assert ts.child_frame_id == "child"

    def test_translation_inverted(self):
        # cam-from-world with t=[1,2,3] → world-from-cam t=[-1,-2,-3]
        ext = np.eye(3, 4, dtype=np.float64)
        ext[:3, 3] = [1.0, 2.0, 3.0]
        ts = extrinsic_to_transform(ext, "map", "camera", _make_stamp())
        assert ts.transform.translation.x == pytest.approx(-1.0, abs=1e-10)
        assert ts.transform.translation.y == pytest.approx(-2.0, abs=1e-10)
        assert ts.transform.translation.z == pytest.approx(-3.0, abs=1e-10)

    def test_accepts_4x4_input(self):
        ext = np.eye(4, dtype=np.float64)
        ts = extrinsic_to_transform(ext, "map", "camera", _make_stamp())
        assert ts.transform.rotation.w == pytest.approx(1.0, abs=1e-10)


# ── extrinsic_to_pose_stamped ─────────────────────────────────────────────────

class TestExtrinsicToPoseStamped:
    def test_identity_gives_origin_pose(self):
        ext = np.eye(3, 4, dtype=np.float64)
        ps = extrinsic_to_pose_stamped(ext, "map", _make_stamp())
        assert ps.pose.position.x == pytest.approx(0.0, abs=1e-10)
        assert ps.pose.position.y == pytest.approx(0.0, abs=1e-10)
        assert ps.pose.position.z == pytest.approx(0.0, abs=1e-10)
        assert ps.pose.orientation.w == pytest.approx(1.0, abs=1e-10)

    def test_frame_id_stored(self):
        ext = np.eye(3, 4, dtype=np.float64)
        ps = extrinsic_to_pose_stamped(ext, "world", _make_stamp())
        assert ps.header.frame_id == "world"
