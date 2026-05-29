"""Conversions between ROS2 messages and numpy/torch tensors."""

import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


def stamp_to_float(stamp: Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def numpy_to_pointcloud2(
    points: np.ndarray,
    colors: np.ndarray,
    frame_id: str,
    stamp,
) -> PointCloud2:
    """Convert (N,3) points and (N,3) uint8 colors to PointCloud2 XYZRGB."""
    assert points.shape[0] == colors.shape[0], "points and colors must have the same length"
    n = points.shape[0]

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 16

    rgb_packed = np.zeros(n, dtype=np.uint32)
    rgb_packed[:] = (
        (colors[:, 0].astype(np.uint32) << 16) |
        (colors[:, 1].astype(np.uint32) << 8) |
        colors[:, 2].astype(np.uint32)
    )
    rgb_float = rgb_packed.view(np.float32)

    data = np.zeros(n, dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
    ])
    data['x'] = points[:, 0].astype(np.float32)
    data['y'] = points[:, 1].astype(np.float32)
    data['z'] = points[:, 2].astype(np.float32)
    data['rgb'] = rgb_float

    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id)
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


def extrinsic_to_transform(
    extrinsic: np.ndarray,
    parent_frame: str,
    child_frame: str,
    stamp,
) -> TransformStamped:
    """
    Convert a (3,4) or (4,4) OpenCV extrinsic matrix (cam-from-world) to a TF TransformStamped.
    The transform represents world-from-camera (parent=world, child=camera).
    """
    if extrinsic.shape == (3, 4):
        E = np.eye(4)
        E[:3, :] = extrinsic
    else:
        E = extrinsic.copy()

    # Invert: from cam-from-world  →  world-from-cam
    R = E[:3, :3]
    t = E[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    # Rotation matrix → quaternion (x, y, z, w)
    qw, qx, qy, qz = _rot_to_quat(R_inv)

    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent_frame
    ts.child_frame_id = child_frame
    ts.transform.translation.x = float(t_inv[0])
    ts.transform.translation.y = float(t_inv[1])
    ts.transform.translation.z = float(t_inv[2])
    ts.transform.rotation.x = float(qx)
    ts.transform.rotation.y = float(qy)
    ts.transform.rotation.z = float(qz)
    ts.transform.rotation.w = float(qw)
    return ts


def extrinsic_to_pose_stamped(
    extrinsic: np.ndarray,
    frame_id: str,
    stamp,
) -> PoseStamped:
    """Convert (3,4) cam-from-world extrinsic to world-from-cam PoseStamped."""
    if extrinsic.shape == (3, 4):
        E = np.eye(4)
        E[:3, :] = extrinsic
    else:
        E = extrinsic.copy()

    R = E[:3, :3]
    t = E[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    qw, qx, qy, qz = _rot_to_quat(R_inv)

    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(t_inv[0])
    ps.pose.position.y = float(t_inv[1])
    ps.pose.position.z = float(t_inv[2])
    ps.pose.orientation.x = float(qx)
    ps.pose.orientation.y = float(qy)
    ps.pose.orientation.z = float(qz)
    ps.pose.orientation.w = float(qw)
    return ps


def camera_info_to_intrinsics(msg: CameraInfo) -> np.ndarray:
    """Extract (3,3) intrinsic matrix from sensor_msgs/CameraInfo."""
    K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
    return K


def _rot_to_quat(R: np.ndarray):
    """Convert (3,3) rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z
