"""
Standalone PointCloud node — simpler version without full SLAM.

Useful for:
  - Testing VGGT inference on a camera stream without the overhead of the SLAM backend
  - Getting a dense depth cloud for mapping tasks where pose is provided by another source

Subscribes:
  image_raw          sensor_msgs/Image
  camera_info        sensor_msgs/CameraInfo  (optional)

Publishes:
  ~/pointcloud       sensor_msgs/PointCloud2
  ~/depth            sensor_msgs/Image
"""

from __future__ import annotations

import threading
import queue
import traceback

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_OK = True
except ImportError:
    _CV_BRIDGE_OK = False

from vggt_slam_ros2.core.vggt_wrapper import VGGTWrapper
from vggt_slam_ros2.core.keyframe_selector import KeyframeSelector
from vggt_slam_ros2.core.sliding_window import SlidingWindow, Keyframe
from vggt_slam_ros2.utils.ros_conversions import (
    numpy_to_pointcloud2,
    stamp_to_float,
)


class VGGTPointCloudNode(Node):

    def __init__(self) -> None:
        super().__init__('vggt_pointcloud_node')

        self.declare_parameter('checkpoint', 'facebook/VGGT-1B')
        self.declare_parameter('use_bf16', True)
        self.declare_parameter('window_size', 8)
        self.declare_parameter('window_stride', 4)
        self.declare_parameter('min_flow', 8.0)
        self.declare_parameter('conf_threshold_pct', 20.0)
        self.declare_parameter('map_frame', 'camera')

        self._bridge = CvBridge() if _CV_BRIDGE_OK else None

        self.get_logger().info("Loading VGGT model...")
        checkpoint = self.get_parameter('checkpoint').value
        use_bf16 = self.get_parameter('use_bf16').value
        self._vggt = VGGTWrapper(checkpoint=checkpoint, use_bf16=use_bf16)
        self.get_logger().info("VGGT ready.")

        window_size = self.get_parameter('window_size').value
        window_stride = self.get_parameter('window_stride').value
        min_flow = self.get_parameter('min_flow').value
        self._conf_thr = self.get_parameter('conf_threshold_pct').value
        self._map_frame = self.get_parameter('map_frame').value

        self._kf_selector = KeyframeSelector(min_flow=min_flow)
        self._window = SlidingWindow(
            window_size=window_size,
            stride=window_stride,
            callback=self._on_window_ready,
        )

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._img_sub = self.create_subscription(Image, 'image_raw', self._image_cb, qos)
        self._pc_pub = self.create_publisher(PointCloud2, '~/pointcloud', 10)
        self._depth_pub = self.create_publisher(Image, '~/depth', 10)

        self._infer_queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image) -> None:
        if self._bridge:
            try:
                bgr = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception:
                return
        else:
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

        if not self._kf_selector.should_accept(bgr):
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._window.add(rgb, stamp_to_float(msg.header.stamp))

    def _on_window_ready(self, frames: list[Keyframe]) -> None:
        try:
            self._infer_queue.put_nowait(frames)
        except queue.Full:
            self.get_logger().warn("Inference queue full — dropping window.")

    def _inference_loop(self) -> None:
        while rclpy.ok():
            try:
                frames = self._infer_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process(frames)
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}\n{traceback.format_exc()}")

    def _process(self, frames: list[Keyframe]) -> None:
        images = [kf.image_rgb for kf in frames]
        result = self._vggt.infer(images)

        S = result['world_points'].shape[0]
        colors = np.stack([np.array(img, dtype=np.uint8) for img in images])

        pts_list, col_list = [], []
        for i in range(S):
            pts = result['world_points'][i].reshape(-1, 3)
            col = colors[i].reshape(-1, 3)
            conf = result['world_points_conf'][i].reshape(-1)
            thr = np.percentile(conf, self._conf_thr)
            mask = conf >= thr
            pts_list.append(pts[mask])
            col_list.append(col[mask])

        all_pts = np.concatenate(pts_list, 0).astype(np.float32)
        all_col = np.concatenate(col_list, 0).astype(np.uint8)

        ros_stamp = self._float_to_stamp(frames[-1].stamp)
        pc_msg = numpy_to_pointcloud2(all_pts, all_col, self._map_frame, ros_stamp)
        self._pc_pub.publish(pc_msg)

        if self._bridge:
            depth = result['depth'][-1].astype(np.float32)
            d_msg = self._bridge.cv2_to_imgmsg(depth, '32FC1')
            d_msg.header.stamp = ros_stamp
            d_msg.header.frame_id = self._map_frame
            self._depth_pub.publish(d_msg)

    @staticmethod
    def _float_to_stamp(t: float):
        from builtin_interfaces.msg import Time
        msg = Time()
        msg.sec = int(t)
        msg.nanosec = int((t - int(t)) * 1e9)
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VGGTPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
