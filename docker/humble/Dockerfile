# syntax=docker/dockerfile:1
# ============================================================
# vggt-slam-ros2 — ROS2 Humble (Ubuntu 22.04 + CUDA 12.8)
# ============================================================
#
# Targets:
#   runtime (default) — cudnn-runtime, SLAM inference (~25–35 GB)
#   dev               — runtime + RViz2, colcon, pytest, matplotlib
#
# Build:
#   docker build -f docker/humble/Dockerfile -t vggt-slam-ros2:humble .
#   docker build -f docker/humble/Dockerfile --target dev -t vggt-slam-ros2:humble-dev .
#
# Host: nvidia-container-toolkit, driver >= 570 (CUDA 12.8)

# ── Builder (CUDA devel — compile/install only) ───────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG VGGT_REF=main
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
    python3-pip python3-dev \
    git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libxcb-cursor0 libxkbcommon-x11-0 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

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
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble

RUN pip3 install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir \
    "numpy>=1.24.0,<2.0" \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.8.0 \
    huggingface_hub>=0.20.0 \
    open3d>=0.17.0 \
    einops \
    timm \
    transformers>=4.38.0 \
    gtsam>=4.2

RUN git clone --depth 1 --branch "${VGGT_REF}" \
    https://github.com/facebookresearch/vggt.git /opt/vggt \
    && rm -rf /opt/vggt/.git

WORKDIR /ros2_ws
COPY package.xml CMakeLists.txt setup.py setup.cfg LICENSE \
    /ros2_ws/src/vggt_slam_ros2/
COPY vggt_slam_ros2/ /ros2_ws/src/vggt_slam_ros2/vggt_slam_ros2/
COPY launch/ /ros2_ws/src/vggt_slam_ros2/launch/
COPY config/ /ros2_ws/src/vggt_slam_ros2/config/
COPY srv/ /ros2_ws/src/vggt_slam_ros2/srv/
COPY resource/ /ros2_ws/src/vggt_slam_ros2/resource/

RUN bash -c "source /opt/ros/humble/setup.bash && \
    colcon build \
        --packages-select vggt_slam_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log"

# ── Runtime (CUDA cudnn-runtime — robot / production) ─────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV ROS_DISTRO=humble
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,display
ENV PYTHONPATH=/opt/vggt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libxcb-cursor0 libxkbcommon-x11-0 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/ros/humble /opt/ros/humble
COPY --from=builder /opt/vggt /opt/vggt
COPY --from=builder /usr/local/lib/python3.10/dist-packages \
    /usr/local/lib/python3.10/dist-packages
COPY --from=builder /ros2_ws/install /ros2_ws/install

WORKDIR /ros2_ws
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "vggt_slam_ros2", "vggt_slam.launch.py"]

# ── Dev (RViz, colcon rebuild in entrypoint, offline eval) ────────────────────
FROM runtime AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rviz2 \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    matplotlib \
    "pytest>=7" pytest-cov
