#!/usr/bin/env python3
"""
Publish a TUM RGB-D sequence to ROS2 topics for vggt_slam_ros2.

Usage
-----
  # Terminal 1 — SLAM (Docker)
  make run-humble

  # Terminal 2 — play TUM sequence (inside the running SLAM container)
  make play-tum TUM_DATASET=src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_room

  # Or manually via docker compose exec:
  docker compose --profile humble exec vggt-slam-humble bash -c \\
    'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \\
     python3 /ros2_ws/src/vggt_slam_ros2/scripts/play_tum_to_ros.py \\
       --dataset /ros2_ws/src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_desk --rate 10'

Expected dataset layout:
  <dataset>/rgb.txt
  <dataset>/rgb/*.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_OK = True
except ImportError:
    _CV_BRIDGE_OK = False

# TUM freiburg1 default intrinsics (640×480)
TUM_FR1 = {
    'width': 640,
    'height': 480,
    'fx': 517.306408,
    'fy': 516.469215,
    'cx': 318.643040,
    'cy': 255.313989,
}


def load_tum_rgb(dataset_dir: Path, max_frames: int = 0) -> list[tuple[float, Path]]:
    rgb_txt = dataset_dir / 'rgb.txt'
    if not rgb_txt.exists():
        raise FileNotFoundError(f'rgb.txt not found in {dataset_dir}')

    entries: list[tuple[float, Path]] = []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            img_path = dataset_dir / parts[1]
            entries.append((ts, img_path))

    if max_frames > 0:
        entries = entries[:max_frames]
    if not entries:
        raise ValueError(f'No RGB frames found in {dataset_dir}')
    return entries


def float_to_time(stamp: float) -> Time:
    t = Time()
    t.sec = int(stamp)
    t.nanosec = int(round((stamp - t.sec) * 1e9))
    return t


def make_camera_info(stamp: float, frame_id: str) -> CameraInfo:
    info = CameraInfo()
    info.header.stamp = float_to_time(stamp)
    info.header.frame_id = frame_id
    info.width = TUM_FR1['width']
    info.height = TUM_FR1['height']
    fx, fy, cx, cy = TUM_FR1['fx'], TUM_FR1['fy'], TUM_FR1['cx'], TUM_FR1['cy']
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.distortion_model = 'plumb_bob'
    return info


class TumPlayer(Node):
    def __init__(
        self,
        image_topic: str,
        camera_info_topic: str,
        frame_id: str,
    ) -> None:
        super().__init__('tum_player')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub_img = self.create_publisher(Image, image_topic, qos)
        self._pub_info = self.create_publisher(CameraInfo, camera_info_topic, qos)
        self._bridge = CvBridge() if _CV_BRIDGE_OK else None
        self._frame_id = frame_id

    def _bgr_to_image(self, bgr: np.ndarray, stamp: float) -> Image:
        if self._bridge is not None:
            msg = self._bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        else:
            msg = Image()
            msg.height, msg.width = bgr.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = msg.width * 3
            msg.data = bgr.tobytes()
        msg.header = Header()
        msg.header.stamp = float_to_time(stamp)
        msg.header.frame_id = self._frame_id
        return msg

    def play(
        self,
        entries: list[tuple[float, Path]],
        rate: float,
        loop: bool,
        start_delay: float,
        use_dataset_timing: bool,
    ) -> None:
        if start_delay > 0:
            self.get_logger().info(f'Waiting {start_delay:.1f}s for subscribers...')
            time.sleep(start_delay)

        n = len(entries)
        self.get_logger().info(
            f'Playing {n} frames → {self._pub_img.topic_name} '
            f'at {rate:.1f} Hz (loop={loop})'
        )

        first_info = make_camera_info(entries[0][0], self._frame_id)
        self._pub_info.publish(first_info)
        rclpy.spin_once(self, timeout_sec=0)

        iteration = 0
        while rclpy.ok():
            t_wall_start = time.monotonic()
            for i, (ts, img_path) in enumerate(entries):
                if not rclpy.ok():
                    return

                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    self.get_logger().warn(f'Could not read {img_path} — skipping')
                    continue

                self._pub_img.publish(self._bgr_to_image(bgr, ts))
                if i % 50 == 0:
                    self.get_logger().info(f'  frame {i + 1}/{n}  ({img_path.name})')

                rclpy.spin_once(self, timeout_sec=0)

                if i + 1 < n:
                    if use_dataset_timing:
                        dt = entries[i + 1][0] - ts
                        sleep_s = max(0.0, dt / rate)
                    else:
                        sleep_s = 1.0 / rate
                    time.sleep(sleep_s)

            iteration += 1
            if not loop:
                break
            self.get_logger().info(f'Loop {iteration + 1} — restarting sequence')

        elapsed = time.monotonic() - t_wall_start
        self.get_logger().info(f'Done. Published {n} frames in {elapsed:.1f}s')


def main() -> None:
    parser = argparse.ArgumentParser(description='Play a TUM RGB sequence to ROS2 topics')
    parser.add_argument(
        '--dataset', required=True, type=Path,
        help='Path to TUM sequence directory (contains rgb.txt and rgb/)',
    )
    parser.add_argument(
        '--rate', type=float, default=10.0,
        help='Playback rate in Hz (default: 10). With --dataset-timing, scales TUM deltas.',
    )
    parser.add_argument(
        '--max-frames', type=int, default=0,
        help='Cap number of frames (0 = all)',
    )
    parser.add_argument(
        '--image-topic', default='/camera/image_raw',
        help='Image topic (default: /camera/image_raw)',
    )
    parser.add_argument(
        '--camera-info-topic', default='/camera/camera_info',
        help='CameraInfo topic (default: /camera/camera_info)',
    )
    parser.add_argument(
        '--frame-id', default='camera',
        help='TF / header frame_id (default: camera)',
    )
    parser.add_argument(
        '--loop', action='store_true',
        help='Restart the sequence when finished',
    )
    parser.add_argument(
        '--start-delay', type=float, default=2.0,
        help='Seconds to wait before publishing (default: 2)',
    )
    parser.add_argument(
        '--dataset-timing', action='store_true',
        help='Sleep according to TUM timestamps (scaled by --rate) instead of fixed Hz',
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        print(f'Error: dataset directory not found: {dataset}', file=sys.stderr)
        sys.exit(1)

    entries = load_tum_rgb(dataset, max_frames=args.max_frames)

    rclpy.init()
    node = TumPlayer(args.image_topic, args.camera_info_topic, args.frame_id)
    try:
        node.play(
            entries,
            rate=args.rate,
            loop=args.loop,
            start_delay=args.start_delay,
            use_dataset_timing=args.dataset_timing,
        )
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
