# ============================================================
# vggt-slam-ros2 — ROS2 Jazzy (Ubuntu 24.04 + CUDA 12.4)
# ============================================================
#
# Build:
#   docker build -f docker/jazzy/Dockerfile -t vggt-slam-ros2:jazzy .
#
# Requirements on the host:
#   - nvidia-container-toolkit installed and configured
#   - NVIDIA driver >= 550 (CUDA 12.4 compatible)

FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── NVIDIA runtime flags ─────────────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,display

# ── System base ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
    python3-pip python3-dev \
    git wget \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── ROS2 Jazzy ───────────────────────────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-base \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-diagnostic-msgs \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=jazzy

# ── PyTorch (CUDA 12.4) ───────────────────────────────────────────────────────
# Ubuntu 24.04 ships Python 3.12 — use --break-system-packages for global install
RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy>=1.26.0 \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.9.0 \
    huggingface_hub>=0.23.0 \
    open3d>=0.18.0

# ── VGGT ─────────────────────────────────────────────────────────────────────
ARG VGGT_REF=main
RUN pip3 install --no-cache-dir --break-system-packages \
    "git+https://github.com/facebookresearch/vggt.git@${VGGT_REF}"

# ── Copy and build our package ───────────────────────────────────────────────
WORKDIR /ros2_ws
COPY . src/vggt_slam_ros2/

RUN bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build \
        --packages-select vggt_slam_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log"

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "vggt_slam_ros2", "vggt_slam.launch.py"]
