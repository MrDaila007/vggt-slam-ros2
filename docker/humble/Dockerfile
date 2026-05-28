# ============================================================
# vggt-slam-ros2 — ROS2 Humble (Ubuntu 22.04 + CUDA 12.8)
# ============================================================
#
# Build:
#   docker build -f docker/humble/Dockerfile -t vggt-slam-ros2:humble .
#
# Requirements on the host:
#   - nvidia-container-toolkit installed and configured
#   - NVIDIA driver >= 570 (CUDA 12.8, supports Blackwell sm_120)

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

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

# ── ROS2 Humble ──────────────────────────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-diagnostic-msgs \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble

# ── PyTorch (CUDA 12.8 — Blackwell sm_120 support requires PyTorch 2.7+) ───────
RUN pip3 install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    "numpy>=1.24.0,<2.0" \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.8.0 \
    huggingface_hub>=0.20.0 \
    open3d>=0.17.0 \
    matplotlib \
    einops \
    timm \
    transformers>=4.38.0 \
    gtsam>=4.2 \
    "pytest>=7" pytest-cov

# ── VGGT ─────────────────────────────────────────────────────────────────────
# pip install creates a broken UNKNOWN-0.0.0 wheel (pyproject.toml has no name).
# Clone the repo and add it to PYTHONPATH instead.
ARG VGGT_REF=main
RUN git clone https://github.com/facebookresearch/vggt.git /opt/vggt && \
    git -C /opt/vggt checkout ${VGGT_REF}
ENV PYTHONPATH="/opt/vggt:${PYTHONPATH}"

# ── Copy and build our package ───────────────────────────────────────────────
WORKDIR /ros2_ws
COPY . src/vggt_slam_ros2/

RUN bash -c "source /opt/ros/humble/setup.bash && \
    colcon build \
        --packages-select vggt_slam_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log"
# install/ is kept so the workspace is already set up for the entrypoint.
# The entrypoint will rebuild from mounted source on every container start,
# so the baked-in install is just a warm base that avoids re-downloading deps.

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "vggt_slam_ros2", "vggt_slam.launch.py"]
