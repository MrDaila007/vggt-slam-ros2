#!/bin/bash
# Container entrypoint — sources ROS2 and the workspace before running CMD.
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "/ros2_ws/install/setup.bash"

# Forward all args (CMD or docker run arguments)
exec "$@"
