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
- [x] **Scale anchoring** — `core/scale_anchor.py` — inter-window Sim(3) correction
- [x] **TUM baseline** — ATE RMSE 0.125 m on freiburg1_desk (200 frames)

---

## Stage 2 — Loop Closure

- [x] `core/image_retrieval.py` — DINOv2 embeddings + cosine similarity for loop detection
- [x] `core/pose_graph.py` — GTSAM factor graph for global trajectory optimisation
- [x] Integrate loop closure into `slam_node.py` — trigger on detection
- [ ] **Add `--loop_closure` flag to `test_on_tum.py`** — integrate ImageRetrieval + PoseGraph into eval script
- [ ] **Test loop closure on `freiburg1_room`** — ATE before / after loop closure
- [ ] **Test loop closure on `freiburg1_360`** — 360° rotation sequence

---

## Stage 3 — Polish and Usability

- [x] `config/vggt_slam.rviz` — RViz2 config with PointCloud2, Path, TF, Depth displays
- [x] `docker/humble/Dockerfile` — Ubuntu 22.04 + CUDA 12.8 + ROS2 Humble
- [x] `docker/jazzy/Dockerfile` — Ubuntu 24.04 + CUDA 12.8 + ROS2 Jazzy
- [x] `docker-compose.yml` — profiles for Humble and Jazzy, NVIDIA GPU passthrough, host network
- [x] `docker/entrypoint.sh` — rebuilds package from mounted source on every container start
- [x] `docker/cyclonedds.xml` — DDS config for robot network connectivity
- [x] `Makefile` — build / run / shell / clean convenience targets
- [x] `.dockerignore` — exclude build artifacts and model files from build context
- [x] `docker/README.md` — host setup, build, run, and robot connection instructions
- [x] `docker-compose.yml` volumes — project source, config, results, hf_cache mounted from host
- [x] README Docker section
- [x] `test/test_keyframe_selector.py` — 10 tests
- [x] `test/test_sliding_window.py` — 16 tests
- [x] `test/test_map_manager.py` — 16 tests
- [x] `test/test_geometry.py` — 19 tests
- [x] `test/test_ros_conversions.py` — 18 tests (1 skipped without ROS2)
- [x] `test/conftest.py` — shared fixtures
- [ ] GitHub Actions CI — flake8 + mypy + `colcon build` check on every push
- [ ] `scripts/eval_all_tum.sh` — run all 9 fr1 sequences and write results to `results/`
- [ ] Demo video — recorded on an office/apartment dataset for the README

---

## Stage 4 — Advanced Features

- [ ] **Stereo support** — second camera as an absolute metric scale reference
- [ ] **Nav2 integration** — publish `OccupancyGrid` from accumulated point cloud
- [x] **Auto parameter tuning** — `core/auto_params.py` — select window_size/stride based on GPU memory
- [ ] **EuRoC dataset** — evaluation on drone footage (challenging lighting and motion)
- [x] **srv/SaveMap.srv** — save map to PCD/PLY/npz via ROS2 service
