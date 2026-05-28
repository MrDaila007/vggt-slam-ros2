# Convenience targets for building and running the Docker containers.
# All targets forward NVIDIA GPU and host network automatically.

HUMBLE_IMAGE := vggt-slam-ros2:humble
JAZZY_IMAGE  := vggt-slam-ros2:jazzy

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
	docker compose --profile humble up

.PHONY: run-jazzy
run-jazzy:
	docker compose --profile jazzy up

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
	docker compose --profile humble logs -f

.PHONY: logs-jazzy
logs-jazzy:
	docker compose --profile jazzy logs -f

# ── TUM RGB-D evaluation (Stage 1.11) ─────────────────────────────────────
# Usage:
#   make eval-tum DATASET=/path/to/rgbd_dataset_freiburg1_desk
#   make eval-tum DATASET=data/rgbd_dataset_freiburg1_desk MAX_FRAMES=200
TUM_DATASET ?= data/rgbd_dataset_freiburg1_desk
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
