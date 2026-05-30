#!/bin/bash
# Container entrypoint.
# Rebuilds the package only when build-critical files change.
# Pure Python changes (.py) are picked up instantly via --symlink-install.
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"

WS=/ros2_ws
SRC="${WS}/src/vggt_slam_ros2"
HASH_FILE="${WS}/build/.vggt_slam_src_hash"

_install_ok() {
    # install/ is not always persisted (only build/ may be in a Docker volume).
    # Require the full layout, not just package.xml, before skipping colcon.
    local py_pkg
    py_pkg=$(find "${WS}/install/vggt_slam_ros2/local/lib" \
        -maxdepth 3 -type d -path "*/dist-packages/vggt_slam_ros2" 2>/dev/null \
        | head -1)
    [ -n "${py_pkg}" ] \
        && [ -d "${py_pkg}/nodes" ] \
        && [ -x "${WS}/install/vggt_slam_ros2/lib/vggt_slam_ros2/slam_node" ] \
        && [ -f "${WS}/install/vggt_slam_ros2/share/vggt_slam_ros2/package.xml" ]
}

_needs_rebuild() {
    # Hash files that actually affect the colcon/CMake/rosidl build output.
    # Changes to .py files don't need a rebuild (symlink-install).
    local current_hash
    current_hash=$(find "${SRC}" \
        \( -name "CMakeLists.txt" \
           -o -name "package.xml" \
           -o -name "setup.py" \
           -o -name "setup.cfg" \
           -o -path "${SRC}/scripts/*" \
           -o -name "*.srv" \
           -o -name "*.msg" \
           -o -name "*.action" \
        \) -type f | sort | xargs md5sum 2>/dev/null | md5sum)

    local prev_hash
    prev_hash=$(cat "${HASH_FILE}" 2>/dev/null || echo "")

    if [ "${current_hash}" = "${prev_hash}" ] && _install_ok; then
        echo "${current_hash}"   # non-empty = unchanged
        return 1                 # no rebuild needed
    fi
    echo "${current_hash}"
    return 0
}

if [ -f "${SRC}/CMakeLists.txt" ] || [ -f "${SRC}/setup.py" ]; then
    src_hash=$(_needs_rebuild) && REBUILD=1 || REBUILD=0

    if [ "${REBUILD}" -eq 1 ]; then
        if ! _install_ok; then
            echo "[entrypoint] Install tree missing or outdated — rebuilding vggt_slam_ros2..."
        else
            echo "[entrypoint] Build-critical files changed — rebuilding vggt_slam_ros2..."
        fi
        cd "${WS}"
        rm -rf "${WS}/build/vggt_slam_ros2" "${WS}/install/vggt_slam_ros2"
        colcon build \
            --packages-select vggt_slam_ros2 \
            --cmake-args -DCMAKE_BUILD_TYPE=Release \
            --symlink-install \
            2>&1 | grep -v "^Starting\|^Finished\|^Summary"
        mkdir -p "$(dirname "${HASH_FILE}")"
        echo "${src_hash}" > "${HASH_FILE}"
        echo "[entrypoint] Build complete."
    else
        echo "[entrypoint] Source unchanged — skipping build."
    fi
else
    echo "[entrypoint] No package manifest found — skipping build."
fi

if [ -f "${WS}/install/setup.bash" ]; then
    source "${WS}/install/setup.bash"
fi

exec "$@"
