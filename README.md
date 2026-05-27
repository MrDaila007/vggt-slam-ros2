# vggt_slam_ros2

A ROS2 Visual SLAM package that uses [VGGT](https://github.com/facebookresearch/vggt) (Visual Geometry Grounded Transformer) as a dense visual front-end for real-time 3D map building.

## Key Design Choices

| Feature | This package | VGGT-SLAM |
|---|---|---|
| Input | ROS2 image topic (live or bag) | Image folder |
| Window strategy | **Sliding window** with configurable overlap | Fixed-size submaps |
| Integration | ROS2 Lifecycle Node, TF2, Nav2-compatible | Standalone script |
| Map update | Incremental (publishes only new points) | Batch at end |
| Scale consistency | Overlap frames anchor each new window to the existing map | SL(4) solver per submap boundary |
| Pose output | TF tree + `nav_msgs/Path` + `geometry_msgs/PoseStamped` | Viser viewer only |
| Depth output | `sensor_msgs/Image` (32FC1) per window | Not published |

## Architecture

```
Camera topic
     │
     ▼
KeyframeSelector          ← optical-flow gating, dual-threshold
     │
     ▼
SlidingWindow             ← buffer of N keyframes, stride S
     │  (window ready)
     ▼
VGGT inference thread     ← async, queue-backed
     │
     ├──► TF2 broadcast   (map → camera)
     ├──► /path           (nav_msgs/Path)
     ├──► /pose           (geometry_msgs/PoseStamped)
     ├──► /pointcloud     (PointCloud2 — incremental delta)
     ├──► /pointcloud_full(PointCloud2 — full map, periodic)
     └──► /depth          (sensor_msgs/Image 32FC1)
```

## Installation

```bash
# 1. Install VGGT
cd ~/ros2_ws/src
git clone https://github.com/facebookresearch/vggt
cd vggt && pip install -e .

# 2. Build this package
cd ~/ros2_ws
colcon build --packages-select vggt_slam_ros2
source install/setup.bash
```

## Quick Start

```bash
# With a live camera (e.g. RealSense):
ros2 launch vggt_slam_ros2 vggt_slam.launch.py \
  image_topic:=/camera/color/image_raw

# With a ROS2 bag:
ros2 bag play my_bag.db3
ros2 launch vggt_slam_ros2 vggt_slam.launch.py

# Point-cloud only (no SLAM backend):
ros2 launch vggt_slam_ros2 vggt_pointcloud.launch.py
```

## Parameters

See `config/params.yaml` for full documentation. Key params:

| Parameter | Default | Description |
|---|---|---|
| `checkpoint` | `facebook/VGGT-1B` | HF model ID or local path |
| `window_size` | `16` | Frames per VGGT call |
| `window_stride` | `8` | New frames between calls (overlap = size − stride) |
| `min_flow` | `10.0` | Min optical-flow (px) to accept a keyframe |
| `conf_threshold_pct` | `20.0` | Filter bottom N% low-confidence points |
| `voxel_size` | `0.0` | Voxel downsampling (0 = off) |

## Lifecycle Management

The SLAM node is a `rclpy.lifecycle.LifecycleNode`:

```bash
# Manually activate after launch (if autostart=false):
ros2 lifecycle set /vggt_slam/vggt_slam_node configure
ros2 lifecycle set /vggt_slam/vggt_slam_node activate

# Reset the map at runtime:
ros2 service call /vggt_slam/vggt_slam_node/reset std_srvs/srv/Empty
```

## License

Apache-2.0. Note: VGGT model weights have their own license (see the VGGT repo).
