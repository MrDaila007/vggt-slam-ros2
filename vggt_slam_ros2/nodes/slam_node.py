"""
VGGT SLAM — ROS2 Lifecycle Node.

Subscribes to a monocular camera stream and builds an incremental dense
3D map using VGGT as the visual front-end.

Topics published:
  /vggt_slam/pointcloud          sensor_msgs/PointCloud2  — incremental map (new points)
  /vggt_slam/pointcloud_full     sensor_msgs/PointCloud2  — full accumulated map
  /vggt_slam/path                nav_msgs/Path            — camera trajectory
  /vggt_slam/pose                geometry_msgs/PoseStamped
  /vggt_slam/depth               sensor_msgs/Image        — latest VGGT depth map

TF broadcasts:
  map → camera_frame  (configurable names)

Topics subscribed:
  /camera/image_raw              sensor_msgs/Image
  /camera/camera_info            sensor_msgs/CameraInfo   (optional)

Services:
  /vggt_slam/save_map            (TODO: custom srv)
  /vggt_slam/reset               std_srvs/Empty
"""

from __future__ import annotations

import threading
import queue
import time
import traceback

import numpy as np
import cv2

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_OK = True
except ImportError:
    _CV_BRIDGE_OK = False

from vggt_slam_ros2.core.vggt_wrapper import VGGTWrapper
from vggt_slam_ros2.core.keyframe_selector import KeyframeSelector
from vggt_slam_ros2.core.sliding_window import SlidingWindow, Keyframe
from vggt_slam_ros2.core.map_manager import MapManager
from vggt_slam_ros2.utils.ros_conversions import (
    numpy_to_pointcloud2,
    extrinsic_to_transform,
    extrinsic_to_pose_stamped,
    stamp_to_float,
)


class VGGTSlamNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__('vggt_slam_node')
        self._declare_parameters()
        self._bridge = CvBridge() if _CV_BRIDGE_OK else None

        # State initialised in on_configure
        self._vggt: VGGTWrapper | None = None
        self._kf_selector: KeyframeSelector | None = None
        self._window: SlidingWindow | None = None
        self._map: MapManager | None = None
        self._tf_broadcaster: TransformBroadcaster | None = None
        self._path_msg = Path()

        # Async inference queue (image → inference thread → result)
        self._infer_queue: queue.Queue = queue.Queue(maxsize=2)
        self._infer_thread: threading.Thread | None = None
        self._running = False

    # ==================================================================
    # ROS2 Lifecycle callbacks
    # ==================================================================

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring VGGT SLAM node...")
        try:
            p = self._get_params()

            self._kf_selector = KeyframeSelector(
                min_flow=p['min_flow'],
                min_rotation_deg=p['min_rotation_deg'],
                max_frames_between_keyframes=p['max_frames_between_kf'],
            )
            self._window = SlidingWindow(
                window_size=p['window_size'],
                stride=p['window_stride'],
                callback=self._on_window_ready,
            )
            self._map = MapManager(voxel_size=p['voxel_size'] or None)
            self._overlap = p['window_size'] - p['window_stride']
            self._conf_threshold_pct = p['conf_threshold_pct']
            self._map_frame = p['map_frame']
            self._camera_frame = p['camera_frame']
            self._publish_full_map = p['publish_full_map']
            self._full_map_period = p['full_map_period']
            self._last_full_map_pub = 0.0

            # Load model (can be slow — done here so activation is fast)
            self.get_logger().info(f"Loading VGGT from {p['checkpoint']} ...")
            self._vggt = VGGTWrapper(
                checkpoint=p['checkpoint'],
                use_bf16=p['use_bf16'],
            )
            self.get_logger().info("VGGT loaded.")

            # Publishers / subscribers / tf created in on_activate
        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}\n{traceback.format_exc()}")
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating VGGT SLAM node...")

        qos_sensor = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscribers
        self._img_sub = self.create_subscription(
            Image, 'image_raw', self._image_callback, qos_sensor)
        self._info_sub = self.create_subscription(
            CameraInfo, 'camera_info', self._camera_info_callback, qos_sensor)

        # Publishers
        self._pc_pub = self.create_publisher(PointCloud2, '~/pointcloud', 10)
        self._pc_full_pub = self.create_publisher(PointCloud2, '~/pointcloud_full', qos_latched)
        self._path_pub = self.create_publisher(Path, '~/path', qos_latched)
        self._pose_pub = self.create_publisher(PoseStamped, '~/pose', 10)
        self._depth_pub = self.create_publisher(Image, '~/depth', 10)

        # Services
        self._reset_srv = self.create_service(Empty, '~/reset', self._reset_callback)

        # TF
        self._tf_broadcaster = TransformBroadcaster(self)

        # Path header
        self._path_msg = Path()
        self._path_msg.header.frame_id = self._map_frame

        # Inference thread
        self._running = True
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True)
        self._infer_thread.start()

        self.get_logger().info("VGGT SLAM node active.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating...")
        self._running = False
        self._infer_queue.put(None)  # unblock thread
        if self._infer_thread:
            self._infer_thread.join(timeout=5.0)

        self.destroy_subscription(self._img_sub)
        self.destroy_subscription(self._info_sub)
        self.destroy_publisher(self._pc_pub)
        self.destroy_publisher(self._pc_full_pub)
        self.destroy_publisher(self._path_pub)
        self.destroy_publisher(self._pose_pub)
        self.destroy_publisher(self._depth_pub)
        self.destroy_service(self._reset_srv)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._vggt = None
        self._kf_selector = None
        if self._window:
            self._window.reset()
        if self._map:
            self._map.reset()
        return TransitionCallbackReturn.SUCCESS

    # ==================================================================
    # Subscription callbacks
    # ==================================================================

    def _image_callback(self, msg: Image) -> None:
        if not self._running:
            return

        # Convert to BGR numpy
        if self._bridge:
            try:
                frame_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().warn(f"cv_bridge conversion failed: {e}")
                return
        else:
            # Fallback: assume RGB8
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            frame_bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

        if not self._kf_selector.should_accept(frame_bgr):
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        stamp = stamp_to_float(msg.header.stamp)
        self._window.add(frame_rgb, stamp)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        # Store camera info for potential use; VGGT infers its own intrinsics
        self._last_camera_info = msg

    # ==================================================================
    # Sliding-window callback (called from _image_callback thread)
    # ==================================================================

    def _on_window_ready(self, frames: list[Keyframe]) -> None:
        """Put window batch into queue for async VGGT inference."""
        try:
            self._infer_queue.put_nowait(frames)
        except queue.Full:
            self.get_logger().warn(
                "Inference queue full — dropping window. "
                "Consider increasing window_stride or using a faster GPU."
            )

    # ==================================================================
    # Inference thread
    # ==================================================================

    def _inference_loop(self) -> None:
        """Runs in a background thread; processes windows sequentially."""
        while self._running:
            try:
                frames = self._infer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frames is None:
                break  # shutdown signal

            try:
                self._process_window(frames)
            except Exception as e:
                self.get_logger().error(
                    f"VGGT inference error: {e}\n{traceback.format_exc()}"
                )

    def _process_window(self, frames: list[Keyframe]) -> None:
        images_rgb = [kf.image_rgb for kf in frames]
        stamps = [kf.stamp for kf in frames]
        global_indices = [kf.index for kf in frames]

        t0 = time.monotonic()
        result = self._vggt.infer(images_rgb)
        dt = time.monotonic() - t0
        self.get_logger().debug(
            f"VGGT inferred {len(frames)} frames in {dt:.3f}s "
            f"({len(frames)/dt:.1f} fps)"
        )

        # Colors from input images
        colors = np.stack([
            np.array(img, dtype=np.uint8) for img in images_rgb
        ])  # (S, H, W, 3)

        new_pts, new_cols = self._map.add_window_result(
            global_indices=global_indices,
            stamps=stamps,
            extrinsics=result['extrinsics'],
            intrinsics=result['intrinsics'],
            world_points=result['world_points'],
            colors=colors,
            conf=result['world_points_conf'],
            conf_threshold_pct=self._conf_threshold_pct,
            overlap=self._overlap,
        )

        # Use the last frame's stamp and pose for TF / path publishing
        last_stamp_float = stamps[-1]
        last_extrinsic = result['extrinsics'][-1]
        ros_stamp = self._float_to_stamp(last_stamp_float)

        self._publish_tf(last_extrinsic, ros_stamp)
        self._publish_pose(last_extrinsic, ros_stamp)
        self._publish_path(last_extrinsic, ros_stamp)

        if new_pts.shape[0] > 0:
            pc_msg = numpy_to_pointcloud2(new_pts, new_cols, self._map_frame, ros_stamp)
            self._pc_pub.publish(pc_msg)

        # Optionally publish full map at reduced frequency
        now = time.monotonic()
        if self._publish_full_map and (now - self._last_full_map_pub) >= self._full_map_period:
            self._publish_full_pointcloud(ros_stamp)
            self._last_full_map_pub = now

        # Depth image of the last frame
        self._publish_depth(result['depth'][-1], ros_stamp)

    # ==================================================================
    # Publishers
    # ==================================================================

    def _publish_tf(self, extrinsic: np.ndarray, stamp) -> None:
        tf_msg = extrinsic_to_transform(
            extrinsic, self._map_frame, self._camera_frame, stamp)
        self._tf_broadcaster.sendTransform(tf_msg)

    def _publish_pose(self, extrinsic: np.ndarray, stamp) -> None:
        ps = extrinsic_to_pose_stamped(extrinsic, self._map_frame, stamp)
        self._pose_pub.publish(ps)

    def _publish_path(self, extrinsic: np.ndarray, stamp) -> None:
        ps = extrinsic_to_pose_stamped(extrinsic, self._map_frame, stamp)
        self._path_msg.poses.append(ps)
        self._path_msg.header.stamp = stamp
        self._path_pub.publish(self._path_msg)

    def _publish_full_pointcloud(self, stamp) -> None:
        pts = self._map.get_all_points()
        cols = self._map.get_all_colors()
        if pts.shape[0] > 0:
            pc_msg = numpy_to_pointcloud2(pts, cols, self._map_frame, stamp)
            self._pc_full_pub.publish(pc_msg)

    def _publish_depth(self, depth: np.ndarray, stamp) -> None:
        """Publish depth map as a 32FC1 ROS2 image."""
        if self._bridge is None:
            return
        depth_msg = self._bridge.cv2_to_imgmsg(
            depth.astype(np.float32), encoding='32FC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self._camera_frame
        self._depth_pub.publish(depth_msg)

    # ==================================================================
    # Service callbacks
    # ==================================================================

    def _reset_callback(self, request, response):
        self.get_logger().info("Resetting map and trajectory...")
        self._map.reset()
        self._kf_selector.reset()
        self._window.reset()
        self._path_msg = Path()
        self._path_msg.header.frame_id = self._map_frame
        return response

    # ==================================================================
    # Parameter helpers
    # ==================================================================

    def _declare_parameters(self) -> None:
        self.declare_parameter('checkpoint', 'facebook/VGGT-1B')
        self.declare_parameter('use_bf16', True)
        self.declare_parameter('window_size', 16)
        self.declare_parameter('window_stride', 8)
        self.declare_parameter('min_flow', 10.0)
        self.declare_parameter('min_rotation_deg', 2.0)
        self.declare_parameter('max_frames_between_kf', 30)
        self.declare_parameter('conf_threshold_pct', 20.0)
        self.declare_parameter('voxel_size', 0.0)        # 0 = no downsampling
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('camera_frame', 'camera')
        self.declare_parameter('publish_full_map', True)
        self.declare_parameter('full_map_period', 5.0)   # seconds

    def _get_params(self) -> dict:
        return {
            'checkpoint':          self.get_parameter('checkpoint').value,
            'use_bf16':            self.get_parameter('use_bf16').value,
            'window_size':         self.get_parameter('window_size').value,
            'window_stride':       self.get_parameter('window_stride').value,
            'min_flow':            self.get_parameter('min_flow').value,
            'min_rotation_deg':    self.get_parameter('min_rotation_deg').value,
            'max_frames_between_kf': self.get_parameter('max_frames_between_kf').value,
            'conf_threshold_pct':  self.get_parameter('conf_threshold_pct').value,
            'voxel_size':          self.get_parameter('voxel_size').value,
            'map_frame':           self.get_parameter('map_frame').value,
            'camera_frame':        self.get_parameter('camera_frame').value,
            'publish_full_map':    self.get_parameter('publish_full_map').value,
            'full_map_period':     self.get_parameter('full_map_period').value,
        }

    @staticmethod
    def _float_to_stamp(t: float):
        from builtin_interfaces.msg import Time
        msg = Time()
        msg.sec = int(t)
        msg.nanosec = int((t - int(t)) * 1e9)
        return msg


# ==================================================================
# Entry point
# ==================================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VGGTSlamNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
