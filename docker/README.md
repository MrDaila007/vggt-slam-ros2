# Docker — vggt-slam-ros2

Two images are provided: one for **ROS2 Humble** (Ubuntu 22.04 + CUDA 12.1)
and one for **ROS2 Jazzy** (Ubuntu 24.04 + CUDA 12.4).
Both images pass the NVIDIA GPU through to the container and connect to the
robot's ROS2 network via host networking.

---

## Host prerequisites

```bash
# Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

---

## Build

```bash
# ROS2 Humble
make build-humble
# or: docker build -f docker/humble/Dockerfile -t vggt-slam-ros2:humble .

# ROS2 Jazzy
make build-jazzy
# or: docker build -f docker/jazzy/Dockerfile -t vggt-slam-ros2:jazzy .

# Both at once
make build-all
```

Build time: ~15–25 min (downloads PyTorch, VGGT, ROS2 packages).
The VGGT-1B model (~2.4 GB) is **not** baked into the image — it is
downloaded on first run and cached in the `hf_cache` Docker volume.

---

## Run

```bash
# ROS2 Humble (default camera topics)
make run-humble

# ROS2 Jazzy with custom camera topic
IMAGE_TOPIC=/camera/color/image_raw make run-jazzy

# With a specific ROS2 domain ID
ROS_DOMAIN_ID=5 make run-humble
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `IMAGE_TOPIC` | `/camera/image_raw` | Camera image topic on the robot |
| `CAMERA_INFO_TOPIC` | `/camera/camera_info` | Camera calibration topic |
| `ROS_DOMAIN_ID` | `0` | Must match the robot's domain ID |
| `DISPLAY` | `:0` | X11 display for RViz2 (if needed) |

---

## Connect to a robot

The container uses `network_mode: host`, so it shares the host machine's
network stack. DDS discovery works exactly like a native ROS2 node.

```
Robot  ←── same LAN ──→  Host PC  ←── host network ──→  Docker container
```

**Steps:**

1. Make sure the robot and host PC are on the same network.
2. Set the same `ROS_DOMAIN_ID` on robot and container.
3. Start the container — it will subscribe to the camera topic and publish
   point clouds, poses, and TF automatically.

### Unicast-only networks (no multicast)

If the robot network blocks multicast, edit `docker/cyclonedds.xml`:

```xml
<General>
  <AllowMulticast>false</AllowMulticast>
</General>
<Discovery>
  <Peers>
    <Peer Address="192.168.1.100"/>  <!-- robot IP -->
  </Peers>
</Discovery>
```

No image rebuild needed — the file is mounted as a volume.

---

## Interactive shell

```bash
make shell-humble   # bash inside the Humble container
make shell-jazzy    # bash inside the Jazzy container

# Then inside the container:
ros2 topic list
ros2 topic echo /vggt_slam/pose
```

## Stop and clean

```bash
make stop     # stop containers
make clean    # stop + remove images and hf_cache volume
```
