# vggt_slam_ros2

[![CI](https://github.com/MrDaila007/vggt-slam-ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/MrDaila007/vggt-slam-ros2/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-135%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-84%25%20core-blue)

A ROS2 Visual SLAM package that uses [VGGT](https://github.com/facebookresearch/vggt)
(Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) as a dense visual
front-end for real-time 3D map building from a monocular camera stream.

## Key Design Choices

| Feature | This package | VGGT-SLAM |
|---|---|---|
| Input | ROS2 image topic (live camera or bag) | Image folder on disk |
| Window strategy | **Sliding window** with configurable overlap | Fixed-size submaps |
| Integration | ROS2 Lifecycle Node, TF2, Nav2-compatible | Standalone Python script |
| Map update | Incremental — publishes only new points per window | Batch at end of run |
| Scale consistency | Overlap frames anchor each new window to the existing map | SL(4) solver per submap boundary |
| Pose output | TF tree + `nav_msgs/Path` + `geometry_msgs/PoseStamped` | Viser viewer only |
| Depth output | `sensor_msgs/Image` 32FC1 per window | Not published |

---

## Architecture

```
Camera topic  (/camera/image_raw)
     │
     ▼
KeyframeSelector          ← optical-flow gating, dual-threshold
     │
     ▼
SlidingWindow             ← buffer of N keyframes, stride S, overlap = N−S
     │  (window ready)
     ▼
VGGT inference thread     ← async, queue-backed (GPU)
     │
     ├──► TF2 broadcast       map → camera
     ├──► ~/pose              geometry_msgs/PoseStamped
     ├──► ~/path              nav_msgs/Path
     ├──► ~/pointcloud        PointCloud2  (incremental — new points only)
     ├──► ~/pointcloud_full   PointCloud2  (full map, published periodically)
     └──► ~/depth             sensor_msgs/Image 32FC1
```

---

## Docker (recommended)

Running via Docker is the recommended way to deploy the SLAM module on a robot.
The container ships all dependencies (CUDA, PyTorch, VGGT, ROS2) and connects
to the robot's ROS2 network transparently via host networking.

**Project folders are mounted into the container** — you can edit
`config/params.yaml` or `docker/cyclonedds.xml` on the host without rebuilding
the image. Results are written back to `./results/` on the host automatically.

### 1 — Host prerequisites

```bash
# Install nvidia-container-toolkit (once per machine)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### 2 — Build the image

```bash
# ROS2 Humble  (Ubuntu 22.04 + CUDA 12.1, driver ≥ 525)
make build-humble

# ROS2 Jazzy   (Ubuntu 24.04 + CUDA 12.4, driver ≥ 550)
make build-jazzy
```

Build time: ~15–25 min (downloads PyTorch, VGGT, ROS2 packages).
The VGGT-1B model (~2.4 GB) is **not** baked into the image — it is
downloaded on first run and cached in the `hf_cache` Docker volume.

### 3 — Run

```bash
# Default — subscribes to /camera/image_raw
make run-humble

# Custom camera topic (e.g. Intel RealSense D435)
IMAGE_TOPIC=/camera/color/image_raw make run-humble

# Different ROS2 domain (must match the robot)
ROS_DOMAIN_ID=5 IMAGE_TOPIC=/camera/color/image_raw make run-jazzy
```

### 4 — Mounted folders

| Host path | Container path | Purpose |
|---|---|---|
| `./` (source) | `/ros2_ws/src/vggt_slam_ros2/` | Package source — rebuilt on every container start |
| `./config/` | `/ros2_ws/src/vggt_slam_ros2/config/` | Edit `params.yaml` live, no rebuild needed |
| `./results/` | `/ros2_ws/results/` | Evaluation outputs written to host |
| `./docker/cyclonedds.xml` | `/etc/cyclonedds/cyclonedds.xml` | DDS network config |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | Persistent VGGT model cache |

### 5 — Connecting to a robot

The container uses `network_mode: host` — it shares the host machine's network
stack, so DDS node discovery works exactly like a native ROS2 process.

```
Robot  ←── same LAN ──→  Host PC  ←── host network ──→  Docker container
```

Set the same `ROS_DOMAIN_ID` on both the robot and the container.
That is all — no port forwarding, no bridge network needed.

**Unicast-only networks (multicast blocked):** edit `docker/cyclonedds.xml`
and add the robot's IP address. The file is mounted as a volume so no image
rebuild is required:

```xml
<Discovery>
  <Peers>
    <Peer Address="192.168.1.100"/>   <!-- robot IP -->
  </Peers>
</Discovery>
```

### 6 — Interactive shell

```bash
make shell-humble   # bash inside the Humble container
make shell-jazzy    # bash inside the Jazzy container

# Inside the container:
ros2 topic list
ros2 topic echo /vggt_slam/pose
```

### All make targets

| Target | Description |
|---|---|
| `make build-humble` | Build the ROS2 Humble image |
| `make build-jazzy` | Build the ROS2 Jazzy image |
| `make build-all` | Build both images |
| `make run-humble` | Start the Humble container via Compose |
| `make run-jazzy` | Start the Jazzy container via Compose |
| `make shell-humble` | Interactive bash — Humble |
| `make shell-jazzy` | Interactive bash — Jazzy |
| `make stop` | Stop running containers |
| `make clean` | Stop + remove images and volumes |

---

## Native Installation

If you prefer to run without Docker:

```bash
# 1. Install VGGT
git clone https://github.com/facebookresearch/vggt
cd vggt && pip install -e .

# 2. Build this package
cd ~/ros2_ws
colcon build --packages-select vggt_slam_ros2
source install/setup.bash
```

---

## Quick Start (native)

```bash
# Live camera (e.g. RealSense D435)
ros2 launch vggt_slam_ros2 vggt_slam.launch.py \
  image_topic:=/camera/color/image_raw

# From a ROS2 bag
ros2 bag play my_bag.db3
ros2 launch vggt_slam_ros2 vggt_slam.launch.py

# Point-cloud only (no SLAM backend)
ros2 launch vggt_slam_ros2 vggt_pointcloud.launch.py
```

---

## Parameters

Edit `config/params.yaml` (changes apply without container rebuild). Key params:

| Parameter | Default | Description |
|---|---|---|
| `checkpoint` | `facebook/VGGT-1B` | HuggingFace model ID or local path |
| `window_size` | `16` | Frames per VGGT call |
| `window_stride` | `8` | New frames between calls (overlap = size − stride) |
| `min_flow` | `10.0` | Min optical-flow (px) to accept a keyframe |
| `conf_threshold_pct` | `20.0` | Filter bottom N% low-confidence points |
| `voxel_size` | `0.0` | Voxel downsampling leaf size in metres (0 = off) |
| `map_frame` | `map` | TF parent frame |
| `camera_frame` | `camera` | TF child frame |

---

## Lifecycle Management

The SLAM node is a `rclpy.lifecycle.LifecycleNode`:

```bash
# Manually configure and activate (if autostart=false)
ros2 lifecycle set /vggt_slam/vggt_slam_node configure
ros2 lifecycle set /vggt_slam/vggt_slam_node activate

# Reset the map at runtime
ros2 service call /vggt_slam/vggt_slam_node/reset std_srvs/srv/Empty
```

---

## Evaluation on TUM RGB-D

```bash
# Download a sequence
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar xzf rgbd_dataset_freiburg1_desk.tgz

# Run (no ROS2 required)
python scripts/test_on_tum.py --dataset rgbd_dataset_freiburg1_desk

# Results in ./results/
#   *_metrics.txt          ATE RMSE, RPE RMSE
#   *_estimated_tum.txt    evo-compatible trajectory
#   *_trajectory.png       top-down plot vs ground truth
```

See [docs/get_tum_dataset.md](docs/get_tum_dataset.md) for all sequences.

---

## Evaluation on EuRoC MAV

```bash
# Download MH_01_easy (machine hall, 3683 frames)
# See ETH Research Collection: https://www.research-collection.ethz.ch

# Run inside the Docker container
python scripts/test_on_euroc.py \
  --dataset /data/MH_01_easy \
  --loop_closure \
  --lc_strategy dedup
```

EuRoC images are greyscale (cam0); the script converts to RGB automatically.
Timestamps are nanoseconds — association is done with a 20 ms tolerance.

---

## Benchmark Results

All runs: `window_size=16`, `stride=8`, loop closure strategy `dedup`.
ATE = Absolute Trajectory Error after Sim(3) alignment (Umeyama 1991).
RPE = Relative Pose Error, delta=1 keyframe.
Hardware: NVIDIA GPU, ~1.9 s/window average.

### Summary

| Sequence | Dataset | Frames | Poses | No-LC ATE | With-LC ATE | Δ ATE | Loops |
|---|---|---|---|---|---|---|---|
| freiburg1_desk | TUM RGB-D | 200 | 103 | **0.125 m** | — | baseline | — |
| freiburg1_360 | TUM RGB-D | 759 | 184 | 0.126 m | **0.123 m** | −2.6% | 1 |
| freiburg1_room | TUM RGB-D | 1362 | 280 | 0.696 m | **0.681 m** | −2.2% | 15 |
| V1_01_easy | EuRoC MAV | 2912 | 184 | 1.502 m | **1.099 m** | −26.8% | 13 |
| MH_01_easy | EuRoC MAV | 3683 | 230 | 2.325 m | **0.784 m** | **−66.3%** | 49 |

### TUM RGB-D — detailed

#### freiburg1_desk (baseline, no loop closure)

| Metric | Value |
|---|---|
| ATE RMSE | **0.125 m** |
| ATE Mean | 0.112 m |
| ATE Median | 0.117 m |
| ATE Std | 0.056 m |
| ATE Max | 0.231 m |
| RPE RMSE | 0.783 m |
| RPE Mean | 0.363 m |
| RPE Max | 3.071 m |
| Sim3 scale | 0.245 |
| Avg window time | 2.05 s/window |

#### freiburg1_360

| Metric | No-LC | With-LC |
|---|---|---|
| ATE RMSE | 0.126 m | **0.123 m** |
| ATE Mean | 0.118 m | 0.111 m |
| ATE Median | 0.108 m | 0.103 m |
| ATE Max | 0.269 m | 0.286 m |
| RPE RMSE | 0.054 m | 0.054 m |
| Sim3 scale | 0.781 | 0.829 |

#### freiburg1_room

| Metric | No-LC | With-LC |
|---|---|---|
| ATE RMSE | 0.696 m | **0.681 m** |
| ATE Mean | 0.644 m | 0.594 m |
| ATE Median | 0.653 m | 0.444 m |
| ATE Max | 1.323 m | 1.510 m |
| RPE RMSE | 0.075 m | 0.078 m |
| Sim3 scale | 0.993 | 1.463 |

### EuRoC MAV — detailed

EuRoC ATE is ~10× higher than TUM fr1: drone footage has fast translational
motion and motion blur. VGGT is optimised for slow indoor camera motion.
Loop closure gives strong correction where the sequence contains a clear 360°
revisit (MH_01_easy machine hall: −66.3%).

#### V1_01_easy (Vicon room, 2912 frames)

| Metric | No-LC | With-LC (13 loops from 40 detected) |
|---|---|---|
| ATE RMSE | 1.502 m | **1.099 m** |
| ATE Mean | 1.363 m | 0.938 m |
| ATE Median | 1.187 m | 0.852 m |
| ATE Std | 0.630 m | 0.573 m |
| ATE Max | 3.279 m | 3.266 m |
| RPE RMSE | 0.427 m | 0.428 m |
| Sim3 scale | 1.622 | 3.324 |

#### MH_01_easy (Machine hall, 3683 frames)

| Metric | No-LC | With-LC (49 loops from 157 detected) |
|---|---|---|
| ATE RMSE | 2.325 m | **0.784 m** |
| ATE Mean | 1.699 m | 0.685 m |
| ATE Median | 0.986 m | 0.613 m |
| ATE Std | 1.588 m | 0.380 m |
| ATE Max | 7.998 m | 2.783 m |
| RPE RMSE | 0.326 m | 0.329 m |
| Sim3 scale | 1.796 | 5.332 |

### Loop closure strategies

Three strategies selectable via `--lc_strategy`:

| Strategy | Description | Best for |
|---|---|---|
| `rotation` | VGGT rotation + odometry translation | Scale-sensitive trajectories |
| `normalize` | VGGT rotation + VGGT translation rescaled to odometry magnitude | Balanced |
| `dedup` *(default)* | Deduplicate candidates to ≤1 per ±5-frame region, full VGGT T_rel | Long sequences with clear loops |

---

## Tests

Unit tests run without a GPU or a ROS2 runtime. All 135 tests pass in ~1 s.

```bash
# Inside the Docker container
python3 -m pytest test/ --ignore=test/test_ros_conversions.py -v \
  --cov=vggt_slam_ros2 --cov-report=term-missing
```

### Results (Python 3.10, pytest 9.0.3, Docker — ROS2 Humble)

```
======================== 135 passed, 1 warning in 1.09s ========================
```

| Test file | Tests | Status |
|---|---|---|
| `test_auto_params.py` | 10 | ✅ pass |
| `test_euroc_loader.py` | 12 | ✅ pass |
| `test_geometry.py` | 22 | ✅ pass |
| `test_image_retrieval.py` | 10 | ✅ pass |
| `test_keyframe_selector.py` | 10 | ✅ pass |
| `test_map_manager.py` | 14 | ✅ pass |
| `test_pose_graph.py` | 12 | ✅ pass |
| `test_scale_anchor.py` | 12 | ✅ pass |
| `test_sliding_window.py` | 13 | ✅ pass |
| **Total** | **135** | **✅ all pass** |

### Coverage

| Module | Coverage | Notes |
|---|---|---|
| `core/keyframe_selector.py` | 100% | |
| `core/scale_anchor.py` | 100% | |
| `core/sliding_window.py` | 100% | |
| `utils/geometry.py` | 92% | |
| `core/pose_graph.py` | 85% | |
| `core/map_manager.py` | 77% | |
| `core/image_retrieval.py` | 71% | |
| `utils/auto_params.py` | 53% | |
| `core/vggt_wrapper.py` | 0% | requires GPU + HuggingFace download |
| `nodes/slam_node.py` | 0% | requires ROS2 runtime |
| `nodes/pointcloud_node.py` | 0% | requires ROS2 runtime |
| `utils/ros_conversions.py` | 0% | requires ROS2 runtime |

Core modules (excludes GPU/ROS2 runtime): **~84% coverage**.

---

## License

The **source code** of this package is licensed under **Apache-2.0**
(see [LICENSE](LICENSE)).

### ⚠️ VGGT model license — read before use

VGGT source code and model weights are under a **separate Meta Research License**,
not Apache-2.0.

| Checkpoint | Commercial use |
|---|---|
| `facebook/VGGT-1B` *(default)* | **Non-commercial / research only** |
| `facebook/VGGT-1B-Commercial` | Allowed after Meta approval via HuggingFace |

**Military and weapons applications are explicitly prohibited.**

For commercial use: apply for `VGGT-1B-Commercial` on HuggingFace and set
`checkpoint: "facebook/VGGT-1B-Commercial"` in `config/params.yaml`.

Full details: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
