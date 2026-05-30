#!/usr/bin/env bash
# Full demo launcher: SLAM + RViz (in container) + TUM dataset playback.
#
# Usage:
#   ./scripts/demo.sh              # SLAM + RViz + TUM playback (default)
#   ./scripts/demo.sh --no-play    # SLAM + RViz only
#   ./scripts/demo.sh --detach     # background SLAM (+ playback if not --no-play)
#
# Prerequisites:
#   make build-humble   # image includes ros-humble-rviz2
#   export DISPLAY=:1   # your X11 display (check with echo $DISPLAY)
#
# Environment (optional):
#   TUM_DATASET=src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_room
#   TUM_RATE=10
#   PLAY_MAX_FRAMES=200  (0 = full sequence)
#   ROS_DOMAIN_ID=0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DETACH=0
PLAY_TUM=1
for arg in "$@"; do
    case "$arg" in
        --detach)    DETACH=1 ;;
        --no-play)   PLAY_TUM=0 ;;
        --play-tum)  PLAY_TUM=1 ;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

TUM_DATASET="${TUM_DATASET:-src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_desk}"
TUM_RATE="${TUM_RATE:-10}"
PLAY_MAX_FRAMES="${PLAY_MAX_FRAMES:-200}"
COMPOSE="docker compose --profile humble"
SLAM_SERVICE="vggt-slam-humble"
RVIZ_CONFIG="/ros2_ws/install/vggt_slam_ros2/share/vggt_slam_ros2/config/vggt_slam.rviz"
RVIZ_LOG="/tmp/vggt_rviz_container.log"

allow_x11() {
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "[demo] ERROR: DISPLAY is not set — RViz cannot open." >&2
        echo "[demo]   export DISPLAY=:1   (or your active display)" >&2
        exit 1
    fi
    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null 2>&1 || true
    fi
}

container_running() {
    $COMPOSE ps --status running -q "$SLAM_SERVICE" 2>/dev/null | grep -q .
}

rviz_running_in_container() {
    $COMPOSE exec -T "$SLAM_SERVICE" pgrep -x rviz2 >/dev/null 2>&1
}

ensure_rviz_in_image() {
    if $COMPOSE exec -T "$SLAM_SERVICE" bash -c \
        'source /opt/ros/humble/setup.bash && command -v rviz2' >/dev/null 2>&1; then
        return 0
    fi
    echo "[demo] ERROR: ros-humble-rviz2 not found in container." >&2
    echo "[demo] Rebuild the image, then restart:" >&2
    echo "[demo]   make build-humble" >&2
    echo "[demo]   make stop && make demo" >&2
    exit 1
}

wait_for_slam() {
    echo "[demo] Waiting for SLAM node to become active (VGGT load may take ~30s)..."
    local i state_out
    for i in $(seq 1 180); do
        if docker logs "$($COMPOSE ps -q "$SLAM_SERVICE" 2>/dev/null)" 2>&1 \
            | grep -q 'VGGT SLAM node active'; then
            echo "[demo] SLAM active."
            return 0
        fi
        state_out=$($COMPOSE exec -T "$SLAM_SERVICE" bash -c \
            'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
             ros2 service call /vggt_slam/vggt_slam_node/get_state \
               lifecycle_msgs/srv/GetState "{}" --timeout 3 2>/dev/null' \
            || true)
        if echo "$state_out" | grep -q "label='active'"; then
            echo "[demo] SLAM active."
            return 0
        fi
        sleep 2
    done
    echo "[demo] ERROR: SLAM did not reach active state in time." >&2
    return 1
}

start_rviz() {
    ensure_rviz_in_image

    if rviz_running_in_container; then
        echo "[demo] RViz already running in container."
        return 0
    fi

    echo "[demo] Starting RViz in container (DISPLAY=${DISPLAY})..."
    $COMPOSE exec -d \
        -e "DISPLAY=${DISPLAY}" \
        -e "QT_X11_NO_MITSHM=1" \
        "$SLAM_SERVICE" bash -c \
        "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
         ros2 run rviz2 rviz2 -d ${RVIZ_CONFIG}" \
        >"${RVIZ_LOG}" 2>&1

    local i
    for i in $(seq 1 10); do
        sleep 1
        if rviz_running_in_container; then
            echo "[demo] RViz started."
            return 0
        fi
    done

    echo "[demo] ERROR: RViz failed to start in container." >&2
    echo "[demo] Log: ${RVIZ_LOG}" >&2
    tail -10 "${RVIZ_LOG}" >&2 || true
    echo "[demo] Try: xhost +local:docker && export DISPLAY=${DISPLAY}" >&2
    return 1
}

play_tum() {
    echo "[demo] Playing TUM dataset: ${TUM_DATASET} (${PLAY_MAX_FRAMES} frames max, ${TUM_RATE} Hz)"
    $COMPOSE exec -T "$SLAM_SERVICE" bash -c \
        "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
         python3 /ros2_ws/src/vggt_slam_ros2/scripts/play_tum_to_ros.py \
           --dataset /ros2_ws/${TUM_DATASET} \
           --rate ${TUM_RATE} \
           --max-frames ${PLAY_MAX_FRAMES} \
           --start-delay 2"
}

run_demo() {
    if [[ "$PLAY_TUM" == "1" ]]; then
        play_tum &
        PLAY_PID=$!
        echo "[demo] TUM playback running (pid ${PLAY_PID}). Streaming SLAM logs..."
        $COMPOSE logs -f "$SLAM_SERVICE" &
        LOG_PID=$!
        wait "$PLAY_PID" || true
        kill "$LOG_PID" 2>/dev/null || true
        echo "[demo] Playback finished. SLAM + RViz still running — stop with: make stop"
    else
        $COMPOSE logs -f "$SLAM_SERVICE"
    fi
}

allow_x11

if container_running; then
    echo "[demo] SLAM container already running."
    wait_for_slam
    start_rviz
    if [[ "$PLAY_TUM" == "1" ]]; then
        run_demo
    else
        echo "[demo] SLAM + RViz ready. Start playback: make play-tum"
        $COMPOSE logs -f "$SLAM_SERVICE"
    fi
    exit 0
fi

if [[ "$DETACH" == "1" ]] || [[ "$PLAY_TUM" == "1" ]]; then
    echo "[demo] Starting SLAM container (detached)..."
    DISPLAY="${DISPLAY}" $COMPOSE up -d
    wait_for_slam
    start_rviz
    run_demo
else
    echo "[demo] Starting SLAM (foreground). Ctrl+C to stop."
    DISPLAY="${DISPLAY}" $COMPOSE up
fi
