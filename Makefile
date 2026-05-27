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
