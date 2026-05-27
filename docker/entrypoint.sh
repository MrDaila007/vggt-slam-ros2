#!/bin/bash
# Container entrypoint.
# Rebuilds the package from the mounted source on every start,
# then sources the workspace and runs CMD.
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"

WS=/ros2_ws
SRC="${WS}/src/vggt_slam_ros2"
INSTALL="${WS}/install/vggt_slam_ros2"

# Rebuild if the source is mounted (directory exists and has setup.py)
if [ -f "${SRC}/setup.py" ]; then
    echo "[entrypoint] Building vggt_slam_ros2 from mounted source..."
    cd "${WS}"
    colcon build \
        --packages-select vggt_slam_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --symlink-install \
        2>&1 | grep -v "^Starting\|^Finished\|^Summary"
    echo "[entrypoint] Build complete."
fi

# Source the built workspace
if [ -f "${WS}/install/setup.bash" ]; then
    source "${WS}/install/setup.bash"
fi

exec "$@"
