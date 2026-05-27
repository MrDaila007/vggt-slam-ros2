# TODO — vggt_slam_ros2 Development Plan

## Stage 1 — Basic Functionality (MVP)

- [x] ROS2 package scaffold (`package.xml`, `setup.py`, `setup.cfg`)
- [x] `VGGTWrapper` — model loading, preprocessing, inference
- [x] `KeyframeSelector` — dual-threshold frame selection (optical flow + max interval)
- [x] `SlidingWindow` — sliding window with configurable overlap
- [x] `MapManager` — incremental point cloud accumulator
- [x] `slam_node.py` — ROS2 Lifecycle Node (full SLAM pipeline)
- [x] `pointcloud_node.py` — lightweight point-cloud-only node
- [x] `ros_conversions.py` — numpy/torch ↔ PointCloud2, TF2, PoseStamped
- [x] Launch files (`vggt_slam.launch.py`, `vggt_pointcloud.launch.py`)
- [x] `config/params.yaml` — all parameters with documentation
- [x] `LICENSE` — Apache-2.0
- [x] `THIRD_PARTY_LICENSES.md` — VGGT license notices
- [x] `.gitignore` and `requirements.txt`
- [x] `scripts/test_on_tum.py` — ROS2-free pipeline test, ATE/RPE metrics, Sim(3) alignment
- [x] `docs/get_tum_dataset.md` — TUM RGB-D download instructions
- [ ] **Scale anchoring** — anchor each new window to overlap frames of the previous one to eliminate scale drift (`core/scale_anchor.py`)
- [ ] **Test on TUM fr1/desk** — run `test_on_tum.py`, record ATE RMSE as baseline
- [ ] **Test on all 9 fr1 sequences** — compare against VGGT-SLAM results

---

## Stage 2 — Loop Closure

- [ ] `core/image_retrieval.py` — DINOv2 embeddings + cosine similarity for loop detection (no NetVLAD license issues)
- [ ] `core/pose_graph.py` — GTSAM factor graph for global trajectory optimisation
- [ ] Integrate loop closure into `slam_node.py` — trigger on detection
- [ ] Test loop closure on `freiburg1_room` (long loop, ~1.5 GB)
- [ ] Compare ATE before / after loop closure

---

## Stage 3 — Polish and Usability

- [ ] `config/vggt_slam.rviz` — RViz2 config with PointCloud2, Path, TF, Depth displays
- [x] `docker/humble/Dockerfile` — Ubuntu 22.04 + CUDA 12.1 + ROS2 Humble
- [x] `docker/jazzy/Dockerfile` — Ubuntu 24.04 + CUDA 12.4 + ROS2 Jazzy
- [x] `docker-compose.yml` — profiles for Humble and Jazzy, NVIDIA GPU passthrough, host network
- [x] `docker/entrypoint.sh` — sources ROS2 and workspace on container start
- [x] `docker/cyclonedds.xml` — DDS config for robot network connectivity
- [x] `Makefile` — build / run / shell / clean convenience targets
- [x] `.dockerignore` — exclude build artifacts and model files from build context
- [x] `docker/README.md` — host setup, build, run, and robot connection instructions
- [ ] GitHub Actions CI — flake8 + mypy + `colcon build` check on every push
- [ ] `scripts/eval_all_tum.sh` — run all 9 fr1 sequences and write results to `results/`
- [ ] Demo video — recorded on an office/apartment dataset for the README

---

## Stage 4 — Advanced Features

- [ ] **Stereo support** — second camera as an absolute metric scale reference (eliminates Sim(3) ambiguity)
- [ ] **Nav2 integration** — publish `OccupancyGrid` from accumulated point cloud for autonomous navigation
- [ ] **Auto parameter tuning** — select `window_size` / `stride` automatically based on available GPU memory
- [ ] **EuRoC dataset** — evaluation on drone footage (challenging lighting and motion)
- [ ] **srv/SaveMap.srv** — save map to PCD/PLY file via ROS2 service
