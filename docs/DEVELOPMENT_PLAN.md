# Development Plan — vggt_slam_ros2

## Project Goal

Build an original, publishable ROS2 package for real-time Visual SLAM using
[VGGT](https://github.com/facebookresearch/vggt) as a dense visual front-end.
The package must be clearly distinct from existing work (VGGT-SLAM by MIT-SPARK),
production-quality in its ROS2 integration, and reproducible by third parties.

---

## Stage 1 — Basic Functionality (MVP)

**Goal:** a working pipeline that reads a camera stream, runs VGGT inference,
and publishes a 3D point cloud with camera poses via standard ROS2 messages.

### 1.1 Package scaffold ✅

Standard ROS2 Python package layout: `package.xml`, `setup.py`, `setup.cfg`,
`resource/` marker. Declares all ROS2 dependencies (`rclpy`, `sensor_msgs`,
`geometry_msgs`, `nav_msgs`, `tf2_ros`, `cv_bridge`).

### 1.2 VGGTWrapper ✅

Thin Python class around the VGGT model:
- Loads checkpoint from HuggingFace (`facebook/VGGT-1B` by default)
- Auto-selects `bfloat16` on Ampere GPUs, `float16` otherwise
- Single `infer(images_rgb)` method — returns `extrinsics`, `intrinsics`,
  `world_points`, `world_points_conf`, `depth`, `depth_conf`
- Wrapped in `torch.inference_mode()` for safety

### 1.3 KeyframeSelector ✅

Decides which incoming frames enter the sliding window.
Two independent criteria run in parallel:

| Criterion | Purpose |
|---|---|
| Optical flow magnitude (Farneback) | Rejects near-static frames |
| Max frame gap | Prevents map starvation on slow motion |

This dual-threshold design is more robust than the single-flow approach in VGGT-SLAM.

### 1.4 SlidingWindow ✅

Buffer of the most recent N keyframes. Fires a callback when enough new frames
have accumulated (`stride` parameter). Consecutive windows share `overlap =
window_size - stride` frames, giving VGGT cross-window context that prevents
scale discontinuities at window boundaries.

```
Window 1:  [f0  f1  f2  ... f15]
Window 2:          [f8  f9  ... f23]
                    ^^^^ overlap (8 frames)
```

### 1.5 MapManager ✅

Incremental accumulator for the global point cloud:
- Skips the overlap frames already integrated from the previous window
- Filters low-confidence points by percentile threshold
- Optional Open3D voxel downsampling to keep memory bounded
- Stores all camera poses (`MapFrame` dataclass) for trajectory output

### 1.6 SLAM Node — ROS2 Lifecycle Node ✅

`slam_node.py` implements `rclpy.lifecycle.LifecycleNode` with four states:

| State | What happens |
|---|---|
| `configure` | Load VGGT model, create KeyframeSelector / SlidingWindow / MapManager |
| `activate` | Create subscribers, publishers, TF broadcaster, start inference thread |
| `deactivate` | Stop inference thread, destroy pub/sub |
| `cleanup` | Release model and map memory |

The background inference thread reads from a bounded queue so the ROS2
callback thread is never blocked by GPU compute.

**Published topics:**

| Topic | Type | Content |
|---|---|---|
| `~/pointcloud` | `PointCloud2` | New points from the latest window |
| `~/pointcloud_full` | `PointCloud2` | Full accumulated map (periodic) |
| `~/path` | `nav_msgs/Path` | Full camera trajectory |
| `~/pose` | `PoseStamped` | Latest camera pose |
| `~/depth` | `sensor_msgs/Image` (32FC1) | Latest VGGT depth map |

**TF broadcast:** `map → camera` on every processed window.

### 1.7 PointCloud Node ✅

Lightweight alternative to the SLAM node — no trajectory or map management,
just VGGT inference → PointCloud2. Useful for quick depth visualisation
or integration into an external SLAM back-end.

### 1.8 ROS2 utilities ✅

`utils/ros_conversions.py`:
- `numpy_to_pointcloud2` — packed XYZRGB PointCloud2
- `extrinsic_to_transform` — (3×4) cam-from-world → `TransformStamped`
- `extrinsic_to_pose_stamped` — (3×4) cam-from-world → `PoseStamped`
- `camera_info_to_intrinsics` — `CameraInfo` → (3×3) K matrix

### 1.9 Configuration ✅

All parameters exposed via ROS2 parameter server (`config/params.yaml`).
Every parameter is documented inline. No hardcoded values in node code.

### 1.10 Scale anchoring ✅

**Problem:** VGGT infers scene geometry up to an unknown scale. Consecutive
windows are independently scaled, so the global map accumulates scale drift.

**Solution (`core/scale_anchor.py`):**
1. After each new window, take the overlap frames (already in the map).
2. Compute the ratio between their VGGT-estimated translations and their
   positions recorded in the previous window.
3. Apply this ratio as a correction factor to all new points and poses.

This keeps scale consistent across window boundaries without requiring a
full pose-graph optimisation at every step.

### 1.11 TUM RGB-D evaluation ✅

Run `scripts/test_on_tum.py` on all 9 fr1 sequences. Record ATE RMSE as the
Stage 1 baseline. Compare against published VGGT-SLAM numbers.

The script:
- Reads `rgb.txt` / `groundtruth.txt` in TUM format
- Runs the full pipeline (KeyframeSelector → SlidingWindow → VGGTWrapper)
- Associates estimated poses to ground truth by timestamp
- Aligns with Sim(3) (Umeyama 1991) to handle unknown scale
- Outputs `metrics.txt`, `estimated_tum.txt` (evo-compatible), `trajectory.png`

**Baseline (freiburg1_desk, 200 frames, window_size=16, stride=8):**

| Metric | Value |
|---|---|
| ATE RMSE | 0.125 m |
| ATE Mean | 0.112 m |
| RPE RMSE | 0.783 m |
| Sim3 scale | 0.245 |
| Avg window time | 2.05 s/window |

---

## Stage 2 — Loop Closure

**Goal:** detect revisited places and correct the accumulated trajectory error
with a global optimisation.

### 2.1 Image retrieval (`core/image_retrieval.py`) ✅

Use **DINOv2** (ViT-B/14, Apache-2.0) to embed keyframes into a descriptor space.
At each new keyframe, compute cosine similarity against all previous embeddings.
If the best match exceeds a configurable threshold and the matched frame is
sufficiently far back in time (to avoid matching adjacent frames), a loop
candidate is returned.

DINOv2 was chosen over NetVLAD because:
- Apache-2.0 license — no commercial restrictions
- Available on HuggingFace without manual download
- Competitive recall on indoor scenes

### 2.2 Pose graph (`core/pose_graph.py`) ✅

GTSAM factor graph with:
- **Between factors** for consecutive window poses (from VGGT relative pose)
- **Loop closure factors** for matched frame pairs (relative pose from VGGT
  re-inference on the matched frame pair)
- **Prior factor** on the first pose to fix gauge freedom

Graph is optimised with Levenberg-Marquardt after each loop closure.
All map points are then rigidly transformed to match the corrected poses.

### 2.3 Integration into slam_node ✅

After every window is processed:
1. Query image retrieval for a loop candidate.
2. If found, run VGGT on (current keyframe, matched keyframe) to get a
   relative pose constraint.
3. Add loop factor to pose graph and re-optimise.
4. Republish corrected full point cloud and path.

### 2.4 Evaluation ✅

Test on `freiburg1_room` (explicit loop) and `freiburg1_360` (360° rotation).
Report ATE RMSE before and after loop closure to quantify the improvement.

**Results:**

| Sequence | Frames | No-LC ATE | With-LC ATE | Improvement | Strategy |
|----------|--------|-----------|-------------|-------------|----------|
| freiburg1_desk | 200 | 0.125 m | — | baseline | — |
| freiburg1_room | 1362 | 0.696 m | 0.681 m | +2.2% | dedup (15 loops) |
| freiburg1_360 | 759 | 0.126 m | 0.123 m | +2.6% | dedup (1 loop) |

Three selectable loop closure strategies available via `--lc_strategy`:
- `rotation` — VGGT rotation + odometry translation (scale-safe, default)
- `normalize` — VGGT rotation + VGGT translation rescaled to odometry magnitude
- `dedup` — deduplicate candidates to ≤1 per ±5-frame region + full VGGT T_rel

---

## Stage 3 — Polish and Usability

**Goal:** the project is ready for public use and GitHub publication.

### 3.1 RViz2 configuration ✅

`config/vggt_slam.rviz` pre-configured with:
- **PointCloud2** display for `~/pointcloud_full` (RGB colouring)
- **Path** display for `~/path`
- **TF** display showing `map → camera` frame
- **Image** display for `~/depth` (colormap)
- Fixed frame set to `map`

### 3.2 Docker ✅

`Dockerfile` based on `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04`:
- ROS2 Humble base
- VGGT installed from source
- This package installed with `colcon build`
- Entrypoint: `ros2 launch vggt_slam_ros2 vggt_slam.launch.py`

`docker-compose.yaml` with two services:
- `slam` — GPU-enabled SLAM node
- `rviz` — RViz2 with X11 forwarding

### 3.3 GitHub Actions CI ✅

`.github/workflows/ci.yml`:
- Trigger: push and pull request to `main`
- Jobs: `flake8` (style), `mypy` (types), `colcon build --packages-select vggt_slam_ros2`
- Docker image cache to keep builds fast

### 3.4 Batch evaluation script ✅

`scripts/eval_all_tum.sh` — loops over all 9 fr1 sequences, calls
`test_on_tum.py` for each, collects results into `results/summary.csv`.
Prints a final table of ATE RMSE per sequence.

### 3.5 Demo video ⬜

Record a ROS2 bag in an office or apartment, run the SLAM node, capture
the RViz2 visualisation. Embed the video in the README to attract users.

---

## Stage 4 — Advanced Features

**Goal:** extend the package beyond the MVP to cover real-world robotic use cases.

### 4.1 Stereo support ⬜

Subscribe to a second camera topic (`image_raw_right`). Use the known baseline
between the two cameras to recover metric scale directly — eliminating the
Sim(3) ambiguity that requires scale anchoring in the monocular case.

### 4.2 Nav2 integration ⬜

Periodically project the accumulated point cloud into a 2D occupancy grid
(`nav_msgs/OccupancyGrid`). Publish on `~/map` so Nav2's costmap can consume
it directly, enabling autonomous navigation without a separate mapping layer.

### 4.3 Automatic parameter tuning ⬜

At startup, query `torch.cuda.mem_get_info()` and select `window_size` /
`stride` to keep GPU memory usage below a configurable budget. Print the
chosen parameters so the user can reproduce the setting manually.

### 4.4 EuRoC evaluation ⬜

Download the EuRoC MAV dataset (drone footage, stereo + IMU).
Run evaluation on the `MH_01` and `V1_01` sequences.
EuRoC is harder than TUM fr1 due to faster motion and greater blur.

Script: `scripts/test_on_euroc.py` — reuses the full TUM pipeline with a
dedicated EuRoC loader (nanosecond timestamps, greyscale→RGB conversion,
`state_groundtruth_estimate0/data.csv` GT parser).

### 4.5 SaveMap service ✅

Implement `srv/SaveMap.srv` (request: file path and format PCD/PLY, response:
success flag). When called, serialise the full accumulated point cloud to disk.

**Implementation:**
- `srv/SaveMap.srv`: `{string path, string format}` → `{bool success, string message}`
- Hybrid `ament_cmake` build with `rosidl_generate_interfaces` for interface generation
- `slam_node.py` wires `~/save_map` service; gracefully skips if srv not yet built
- `MapManager.save_to_file` does the actual serialisation (PCD/PLY/npz)
Useful for offline processing and 3D printing workflows.

---

## File Structure (target state)

```
vggt-slam-ros2/
├── .github/workflows/ci.yml
├── config/
│   ├── params.yaml
│   └── vggt_slam.rviz           ← Stage 3
├── docs/
│   ├── DEVELOPMENT_PLAN.md
│   ├── TODO.md
│   └── get_tum_dataset.md
├── launch/
│   ├── vggt_slam.launch.py
│   └── vggt_pointcloud.launch.py
├── scripts/
│   ├── test_on_tum.py
│   └── eval_all_tum.sh          ← Stage 3
├── vggt_slam_ros2/
│   ├── core/
│   │   ├── vggt_wrapper.py
│   │   ├── keyframe_selector.py
│   │   ├── sliding_window.py
│   │   ├── map_manager.py
│   │   ├── scale_anchor.py      ← Stage 1
│   │   ├── image_retrieval.py   ← Stage 2
│   │   └── pose_graph.py        ← Stage 2
│   ├── nodes/
│   │   ├── slam_node.py
│   │   └── pointcloud_node.py
│   └── utils/
│       ├── ros_conversions.py
│       └── geometry.py
├── Dockerfile                   ← Stage 3
├── docker-compose.yaml          ← Stage 3
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── README.md
├── requirements.txt
└── package.xml
```

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| Sliding window instead of fixed submaps | Overlap frames give VGGT cross-window 3D context; prevents scale jumps at boundaries |
| ROS2 Lifecycle Node | Proper state machine; allows external orchestration (Nav2 lifecycle manager) |
| Async inference queue | GPU inference (~0.5–2 s/window) must not block the ROS2 callback thread |
| DINOv2 for loop detection | Apache-2.0 license; no manual download; good indoor recall |
| Sim(3) alignment in evaluation | VGGT scale is arbitrary; Umeyama gives a fair metric comparison |
| Apache-2.0 for package code | Compatible with ROS2 ecosystem; clear separation from VGGT's Meta license |
