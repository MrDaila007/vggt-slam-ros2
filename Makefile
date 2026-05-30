# Convenience targets for building and running the Docker containers.
# All targets forward NVIDIA GPU and host network automatically.

HUMBLE_IMAGE := vggt-slam-ros2:humble
JAZZY_IMAGE  := vggt-slam-ros2:jazzy
COMPOSE_HUMBLE := docker compose --profile humble
SLAM_SERVICE   := vggt-slam-humble

# ── Build ────────────────────────────────────────────────────────────────────

.PHONY: build-humble
build-humble:
	docker build -f docker/humble/Dockerfile -t $(HUMBLE_IMAGE) .

.PHONY: build-jazzy
build-jazzy:
	docker build -f docker/jazzy/Dockerfile -t $(JAZZY_IMAGE) .

.PHONY: build-all
build-all: build-humble build-jazzy

# ── Run (docker compose) ─────────────────────────────────────────────────────

.PHONY: run-humble
run-humble:
	$(COMPOSE_HUMBLE) up

.PHONY: run-jazzy
run-jazzy:
	docker compose --profile jazzy up

# ── Demo (SLAM + RViz + TUM playback) ───────────────────────────────────────
#
#   make demo              SLAM + RViz (container) + TUM playback
#   make demo-room         freiburg1_room, full sequence
#   make demo-slam         SLAM + RViz only
#   make rviz              RViz in running container
#
# First-time setup (RViz + deps in image):
#   make build-humble && make stop

.PHONY: x11-allow
x11-allow:
	@command -v xhost >/dev/null 2>&1 && xhost +local:docker 2>/dev/null || true

.PHONY: demo demo-humble demo-tum
demo demo-humble demo-tum: x11-allow
	@./scripts/demo.sh --play-tum

.PHONY: demo-room demo-tum-room
demo-room demo-tum-room: x11-allow
	@TUM_DATASET=src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_room \
	 PLAY_MAX_FRAMES=0 ./scripts/demo.sh --play-tum

.PHONY: demo-slam
demo-slam: x11-allow
	@./scripts/demo.sh --no-play

.PHONY: demo-detach
demo-detach: x11-allow
	@./scripts/demo.sh --detach --play-tum

.PHONY: rviz
rviz: x11-allow
	@$(COMPOSE_HUMBLE) up -d
	@$(COMPOSE_HUMBLE) exec -d \
	  -e DISPLAY="$(DISPLAY)" \
	  -e QT_X11_NO_MITSHM=1 \
	  $(SLAM_SERVICE) bash -c \
	  'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
	   ros2 run rviz2 rviz2 -d /ros2_ws/install/vggt_slam_ros2/share/vggt_slam_ros2/config/vggt_slam.rviz'
	@sleep 2; $(COMPOSE_HUMBLE) exec -T $(SLAM_SERVICE) pgrep -x rviz2 >/dev/null \
	  && echo "RViz running in container." \
	  || (echo "RViz failed — rebuild: make build-humble"; exit 1)

# ── Interactive shell ────────────────────────────────────────────────────────

.PHONY: shell-humble
shell-humble:
	docker run --rm -it \
	  --runtime nvidia \
	  --network host \
	  -e NVIDIA_VISIBLE_DEVICES=all \
	  -e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
	  -e DISPLAY=$(DISPLAY) \
	  -v /tmp/.X11-unix:/tmp/.X11-unix \
	  -v hf_cache:/root/.cache/huggingface \
	  $(HUMBLE_IMAGE) bash

.PHONY: shell-jazzy
shell-jazzy:
	docker run --rm -it \
	  --runtime nvidia \
	  --network host \
	  -e NVIDIA_VISIBLE_DEVICES=all \
	  -e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
	  -e DISPLAY=$(DISPLAY) \
	  -v /tmp/.X11-unix:/tmp/.X11-unix \
	  -v hf_cache:/root/.cache/huggingface \
	  $(JAZZY_IMAGE) bash

# ── Utilities ────────────────────────────────────────────────────────────────

.PHONY: stop
stop:
	docker compose down

.PHONY: clean
clean:
	docker compose down --rmi local --volumes

.PHONY: logs-humble
logs-humble:
	$(COMPOSE_HUMBLE) logs -f

.PHONY: logs-jazzy
logs-jazzy:
	docker compose --profile jazzy logs -f

# ── TUM RGB-D evaluation (offline, no ROS2) ─────────────────────────────────
# Usage:
#   make eval-tum
#   make eval-tum TUM_DATASET=src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_desk MAX_FRAMES=200

TUM_DATASET ?= src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_desk
MAX_FRAMES  ?= 0

.PHONY: eval-tum
eval-tum:
	docker run --rm \
	  --runtime nvidia \
	  --network host \
	  -e NVIDIA_VISIBLE_DEVICES=all \
	  -v hf_cache:/root/.cache/huggingface \
	  -v $(PWD)/data:/ros2_ws/data:ro \
	  -v $(PWD)/results:/ros2_ws/results:rw \
	  -v $(PWD):/ros2_ws/src/vggt_slam_ros2:ro \
	  -e PYTHONPATH=/opt/vggt:/ros2_ws/src/vggt_slam_ros2 \
	  $(HUMBLE_IMAGE) \
	  python3 /ros2_ws/src/vggt_slam_ros2/scripts/test_on_tum.py \
	    --dataset /ros2_ws/$(TUM_DATASET) \
	    --out_dir /ros2_ws/results \
	    --max_frames $(MAX_FRAMES)

# ── TUM → ROS2 playback (requires running SLAM container) ───────────────────
# Usage:
#   make play-tum
#   make play-tum TUM_DATASET=src/vggt_slam_ros2/data/rgbd_dataset_freiburg1_room PLAY_MAX_FRAMES=0

TUM_RATE        ?= 10
PLAY_MAX_FRAMES ?= 80

.PHONY: play-tum
play-tum: x11-allow
	$(COMPOSE_HUMBLE) up -d
	$(COMPOSE_HUMBLE) exec $(SLAM_SERVICE) bash -c '\
	  source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
	  python3 /ros2_ws/src/vggt_slam_ros2/scripts/play_tum_to_ros.py \
	    --dataset /ros2_ws/$(TUM_DATASET) \
	    --rate $(TUM_RATE) \
	    --max-frames $(PLAY_MAX_FRAMES) \
	    --start-delay 2'
