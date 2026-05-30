# Docker — vggt-slam-ros2

Two ROS distros (**Humble** / **Jazzy**), each with two image targets:

| Target | Tag | Use case | Approx. size |
|--------|-----|----------|--------------|
| `runtime` (default) | `vggt-slam-ros2:humble` | Robot / SLAM inference | ~25–35 GB |
| `dev` | `vggt-slam-ros2:humble-dev` | RViz, colcon rebuild, TUM eval | ~30–40 GB |

Runtime uses `nvidia/cuda:*-cudnn-runtime` (no CUDA toolkit). Builder stage uses `cudnn-devel` only during `docker build`.

---

## Host prerequisites

```bash
# Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify (driver >= 570 for CUDA 12.8)
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

---

## Build

```bash
# SLAM runtime (smaller — default for robots)
make build-humble
make build-jazzy

# Dev (RViz, colcon in entrypoint, matplotlib for eval-tum)
make build-humble-dev
make build-jazzy-dev

# Free stale build cache (~100 GB possible after experiments)
make docker-prune
```

Build time: ~15–30 min on first build (PyTorch, VGGT, ROS2). VGGT-1B is **not** in the image — cached in volume `vggt-slam-ros2_hf_cache`.

`.dockerignore` excludes `data/`, `results/`, tests — do not remove.

---

## Run

```bash
make run-humble              # runtime image
make run-humble-dev          # dev image (RViz / colcon)

IMAGE_TOPIC=/camera/color/image_raw make run-humble
ROS_DOMAIN_ID=5 make run-humble
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SLAM_IMAGE` | `vggt-slam-ros2:humble` | Override compose image tag |
| `DOCKER_TARGET` | `runtime` | Compose build target (`runtime` / `dev`) |
| `IMAGE_TOPIC` | `/camera/image_raw` | Camera image topic |
| `CAMERA_INFO_TOPIC` | `/camera/camera_info` | Camera calibration topic |
| `ROS_DOMAIN_ID` | `0` | Must match the robot |
| `DISPLAY` | `:0` | X11 for RViz (dev image) |

---

## Demo / RViz

```bash
make build-humble-dev
make demo
```

`make demo` sets `SLAM_IMAGE=vggt-slam-ros2:humble-dev` automatically.

---

## Connect to a robot

`network_mode: host` — same DDS discovery as a native node. Set matching `ROS_DOMAIN_ID`.

For unicast-only networks, edit `docker/cyclonedds.xml` (mounted volume, no rebuild).

---

## Interactive shell

```bash
make shell-humble
make shell-humble-dev
```

## Stop and clean

```bash
make stop
make clean          # containers + local images + compose volumes
make docker-prune   # build cache only (keeps images)
```

---

## Changing CMake / package.xml

**Runtime image:** rebuild the image (`make build-humble`) — entrypoint has no colcon.

**Dev image:** entrypoint runs `colcon` when build-critical files change (volumes preserve `install/`).

Pure `.py` edits work with mounted source on both images (symlink-install from baked `install/` or volume).
