# syntax=docker/dockerfile:1
# ============================================================
# vggt-slam-ros2 — ROS2 Jazzy (Ubuntu 24.04 + CUDA 12.8)
# ============================================================
#
# Targets: runtime (default), dev
# Build:
#   docker build -f docker/jazzy/Dockerfile -t vggt-slam-ros2:jazzy .
#   docker build -f docker/jazzy/Dockerfile --target dev -t vggt-slam-ros2:jazzy-dev .

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

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
    ros-jazzy-ros-base \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-diagnostic-msgs \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=jazzy

RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir --break-system-packages \
    "numpy>=1.26.0,<2.0" \
    Pillow>=10.0.0 \
    opencv-python-headless>=4.9.0 \
    huggingface_hub>=0.23.0 \
    open3d>=0.18.0 \
    einops \
    timm

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

RUN bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build \
        --packages-select vggt_slam_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log"

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV ROS_DISTRO=jazzy
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,display
ENV PYTHONPATH=/opt/vggt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libxcb-cursor0 libxkbcommon-x11-0 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/ros/jazzy /opt/ros/jazzy
COPY --from=builder /opt/vggt /opt/vggt
COPY --from=builder /usr/local/lib/python3.12/dist-packages \
    /usr/local/lib/python3.12/dist-packages
COPY --from=builder /ros2_ws/install /ros2_ws/install

WORKDIR /ros2_ws
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "vggt_slam_ros2", "vggt_slam.launch.py"]

FROM runtime AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-rviz2 \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    matplotlib \
    "pytest>=7" pytest-cov
