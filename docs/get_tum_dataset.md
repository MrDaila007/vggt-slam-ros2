# How to Download the TUM RGB-D Dataset

Dataset website: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

---

## Quick Start — Single Sequence

For a first test, `fr1/desk` is recommended — small (~600 MB), indoor office scene.

```bash
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar xzf rgbd_dataset_freiburg1_desk.tgz
```

Verify the layout:

```
rgbd_dataset_freiburg1_desk/
├── rgb/                  ← colour frames (640×480, ~30 fps)
├── depth/                ← depth maps (16-bit PNG, mm)
├── rgb.txt               ← timestamps and file paths
├── depth.txt
├── groundtruth.txt       ← reference trajectory (timestamp tx ty tz qx qy qz qw)
└── accelerometer.txt
```

---

## All fr1 Sequences (same set used by VGGT-SLAM)

| Sequence | Size | Description |
|---|---|---|
| `freiburg1_desk`  | ~600 MB | Desk, office — **recommended for first test** |
| `freiburg1_desk2` | ~590 MB | Desk, different viewpoint |
| `freiburg1_room`  | ~1.5 GB | Full room with a large loop |
| `freiburg1_floor` | ~740 MB | Camera pointing downward |
| `freiburg1_plant` | ~500 MB | Plant on a desk |
| `freiburg1_teddy` | ~660 MB | Soft toy |
| `freiburg1_xyz`   | ~480 MB | Pure translational motion |
| `freiburg1_rpy`   | ~490 MB | Pure rotational motion |
| `freiburg1_360`   | ~1.1 GB | 360° rotation |

Download all at once:

```bash
BASE="https://vision.in.tum.de/rgbd/dataset/freiburg1"
for SEQ in desk desk2 room floor plant teddy xyz rpy 360; do
    wget -c "${BASE}/rgbd_dataset_freiburg1_${SEQ}.tgz"
    tar xzf "rgbd_dataset_freiburg1_${SEQ}.tgz"
done
```

---

## Running the Evaluation Script

```bash
# Single sequence
python scripts/test_on_tum.py --dataset rgbd_dataset_freiburg1_desk

# Results are written to ./results/
#   rgbd_dataset_freiburg1_desk_metrics.txt
#   rgbd_dataset_freiburg1_desk_estimated_tum.txt
#   rgbd_dataset_freiburg1_desk_trajectory.png

# Additional check with evo (install: pip install evo)
evo_ape tum rgbd_dataset_freiburg1_desk/groundtruth.txt \
    results/rgbd_dataset_freiburg1_desk_estimated_tum.txt -a --plot
```

All script options:

```
--dataset        path to the sequence directory (required)
--window_size    16       frames per VGGT call
--window_stride  8        new frames between calls
--min_flow       10.0     minimum optical flow to accept a keyframe (px)
--conf_thr       20.0     filter bottom N% of points by confidence
--max_frames     0        cap on number of frames (0 = all)
--out_dir        results/ output directory
--no_plot                 skip matplotlib visualisation
```

---

## Requirements

```bash
pip install torch torchvision opencv-python numpy matplotlib
pip install -e /path/to/vggt          # VGGT from source

# Optional — additional metric evaluation
pip install evo
```
