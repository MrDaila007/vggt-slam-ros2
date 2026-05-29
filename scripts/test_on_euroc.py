#!/usr/bin/env python3
"""
Evaluate the vggt_slam_ros2 pipeline on an EuRoC MAV sequence without ROS2.

Usage
-----
  python scripts/test_on_euroc.py --dataset /path/to/MH_01_easy

Expected dataset layout (standard EuRoC download):
  <dataset>/
    mav0/
      cam0/
        data/           ← greyscale PNG frames
        data.csv        ← timestamp_ns [ns], filename
      state_groundtruth_estimate0/
        data.csv        ← timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z, ...

Optional arguments
------------------
  --window_size   16
  --window_stride 8
  --min_flow      10.0
  --conf_thr      20.0
  --checkpoint    facebook/VGGT-1B
  --out_dir       results/
  --max_frames    0         (0 = all)
  --no_plot
  --loop_closure
  --lc_threshold  0.85
  --lc_min_gap    5.0
  --lc_strategy   rotation  (rotation | normalize | dedup)

Note: EuRoC cam0 is greyscale — images are converted to RGB before VGGT
inference. Scale is ambiguous (monocular), so Sim(3) alignment is used
(same as TUM evaluation).

Outputs (in --out_dir)
----------------------
  <seq>_estimated_tum.txt     trajectory in TUM format (evo-compatible)
  <seq>_metrics.txt           ATE RMSE, RPE RMSE
  <seq>_trajectory.png        (unless --no_plot)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

# Pipeline + eval helpers from test_on_tum are imported lazily inside main()
# so that this module's loader functions (no GPU deps) can be imported in tests
# without requiring torch/VGGT to be installed on the host.


# ===========================================================================
# EuRoC dataset loader
# ===========================================================================

def load_euroc_associations(
    dataset_dir: Path,
    max_frames: int = 0,
) -> list[tuple[float, Path]]:
    """
    Parse mav0/cam0/data.csv and return list of (timestamp_s, abs_image_path).
    EuRoC timestamps are in nanoseconds; we convert to seconds.
    """
    csv = dataset_dir / "mav0" / "cam0" / "data.csv"
    if not csv.exists():
        raise FileNotFoundError(f"cam0/data.csv not found in {dataset_dir}")

    img_dir = dataset_dir / "mav0" / "cam0" / "data"
    entries = []
    with open(csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts_ns = int(parts[0])
            ts_s = ts_ns * 1e-9
            img_path = img_dir / parts[1].strip()
            entries.append((ts_s, img_path))

    if max_frames > 0:
        entries = entries[:max_frames]
    return entries


def load_euroc_groundtruth(dataset_dir: Path) -> dict[float, np.ndarray]:
    """
    Parse state_groundtruth_estimate0/data.csv.
    Format: timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z, ...
    Returns dict{timestamp_s: (4,4) SE3 world-from-body}.
    """
    gt_csv = (dataset_dir / "mav0"
              / "state_groundtruth_estimate0" / "data.csv")
    if not gt_csv.exists():
        raise FileNotFoundError(
            f"state_groundtruth_estimate0/data.csv not found in {dataset_dir}"
        )

    gt: dict[float, np.ndarray] = {}
    with open(gt_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts_s = int(parts[0]) * 1e-9
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = (
                float(parts[4]), float(parts[5]),
                float(parts[6]), float(parts[7]),
            )
            T = _quat_trans_to_se3(qx, qy, qz, qw, px, py, pz)
            gt[ts_s] = T
    return gt


# ===========================================================================
# Helpers (duplicated from test_on_tum to keep this script standalone-ish)
# ===========================================================================

def _quat_trans_to_se3(qx, qy, qz, qw, tx, ty, tz) -> np.ndarray:
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--dataset',        required=True,
                   help='Path to EuRoC sequence directory (contains mav0/)')
    p.add_argument('--checkpoint',     default='facebook/VGGT-1B')
    p.add_argument('--window_size',    type=int,   default=16)
    p.add_argument('--window_stride',  type=int,   default=8)
    p.add_argument('--min_flow',       type=float, default=10.0)
    p.add_argument('--max_kf_gap',     type=int,   default=30)
    p.add_argument('--conf_thr',       type=float, default=20.0)
    p.add_argument('--out_dir',        default='results')
    p.add_argument('--max_frames',     type=int,   default=0)
    p.add_argument('--gt_max_diff',    type=float, default=0.02)
    p.add_argument('--no_plot',          action='store_true')
    p.add_argument('--no_scale_anchor',  action='store_true')
    p.add_argument('--loop_closure',     action='store_true')
    p.add_argument('--lc_threshold',     type=float, default=0.85)
    p.add_argument('--lc_min_gap',       type=float, default=5.0)
    p.add_argument('--lc_strategy',      default='rotation',
                   choices=['rotation', 'normalize', 'dedup'])
    return p.parse_args()


def main() -> None:
    # Lazy import: keeps module importable without torch on the host.
    from test_on_tum import (  # noqa: E402
        TUMPipelineRunner,
        align_sim3,
        compute_ate,
        compute_rpe,
        save_tum_trajectory,
        save_metrics,
        plot_trajectory,
        associate_timestamps,
    )

    args = parse_args()

    dataset_dir = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_name = dataset_dir.name
    print(f"Dataset : {dataset_dir}")
    print(f"Sequence: {seq_name}")
    print(f"Output  : {out_dir}\n")

    # ---- Load dataset -------------------------------------------------------
    print("Loading dataset ...")
    entries = load_euroc_associations(dataset_dir, max_frames=args.max_frames)
    print(f"  {len(entries)} cam0 frames\n")

    gt = load_euroc_groundtruth(dataset_dir)
    print(f"  {len(gt)} ground-truth poses\n")

    # EuRoC images are greyscale PNG; convert to BGR then RGB for the pipeline
    def _load_rgb(path: Path) -> np.ndarray | None:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Patch entries so the runner can read them: we pre-convert in the runner
    # by monkey-patching cv2.imread to return the RGB image directly.
    # Simpler: pass a custom loader via subclassing. We use the simplest path:
    # convert each frame to a 3-channel BGR image before calling cv2.imread.
    # Since TUMPipelineRunner calls cv2.imread internally, we convert upfront
    # by creating symlinks to synthetic BGR copies — but that's heavy. Instead,
    # we subclass and override the read step.

    class EuRoCRunner(TUMPipelineRunner):
        """TUMPipelineRunner with greyscale→RGB conversion for EuRoC frames."""

        def run(self, entries: list[tuple[float, Path]]) -> None:
            n = len(entries)
            for i, (ts, img_path) in enumerate(entries):
                if i % 50 == 0:
                    print(f"  Frame {i}/{n} ...", flush=True)

                rgb = _load_rgb(img_path)
                if rgb is None:
                    print(f"  Warning: could not read {img_path}")
                    continue

                # Bypass KeyframeSelector optical-flow (needs BGR); pass
                # a fake BGR for the flow check, use rgb for the window.
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if not self._kf_selector.should_accept(bgr):
                    continue

                self._window.add(rgb, ts)

            self._window.flush()
            print(f"\nProcessed {self._window_count} windows.")
            if self._window_count > 0:
                avg = self._total_infer_time / self._window_count
                print(f"Average inference time per window: {avg:.3f}s")

    # ---- Run pipeline -------------------------------------------------------
    print("Running VGGT pipeline ...")
    runner = EuRoCRunner(args)
    runner.run(entries)

    if len(runner._estimated) < 4:
        print("ERROR: fewer than 4 poses — check dataset path.")
        sys.exit(1)

    # ---- Evaluate -----------------------------------------------------------
    def evaluate(label_suffix: str) -> tuple[dict, dict, float]:
        est_poses_world = runner.get_estimated_poses_world()
        est_ts, est_trans = runner.get_estimated_translations()

        matched = associate_timestamps(est_ts, gt, max_diff=args.gt_max_diff)
        if len(matched) < 4:
            print(f"WARNING: only {len(matched)} GT matches — "
                  "check timestamps or --gt_max_diff")
            sys.exit(1)
        print(f"\nMatched {len(matched)} / {len(est_ts)} poses to ground truth.")

        match_ts = [ts for ts, _ in matched]
        ref_trans = np.array([T[:3, 3] for _, T in matched])
        ref_poses = [T for _, T in matched]

        ts_to_idx = {ts: i for i, ts in enumerate(est_ts)}
        est_trans_m = np.array([
            est_trans[ts_to_idx[ts]]
            for ts in match_ts if ts in ts_to_idx
        ])
        est_poses_m = [
            est_poses_world[ts_to_idx[ts]][1]
            for ts in match_ts if ts in ts_to_idx
        ]

        N = min(len(est_trans_m), len(ref_trans))
        est_trans_m = est_trans_m[:N]
        ref_trans = ref_trans[:N]
        ref_poses = ref_poses[:N]
        est_poses_m = est_poses_m[:N]

        print("\nAligning trajectory (Sim3) ...")
        est_aligned, scale, _ = align_sim3(est_trans_m, ref_trans)
        print(f"  Scale factor: {scale:.4f}")

        ate = compute_ate(est_aligned, ref_trans)
        rpe = compute_rpe(est_poses_m, ref_poses, delta=1)

        tum_path = out_dir / f"{seq_name}_estimated{label_suffix}.txt"
        aligned_poses = []
        for i, (ts, T) in enumerate(est_poses_world[:N]):
            T_al = np.eye(4)
            T_al[:3, :3] = T[:3, :3]
            T_al[:3, 3] = est_aligned[i]
            aligned_poses.append((ts, T_al))
        save_tum_trajectory(aligned_poses, tum_path)
        print(f"Trajectory saved → {tum_path}")

        if not args.no_plot:
            plot_path = out_dir / f"{seq_name}_trajectory{label_suffix}.png"
            plot_trajectory(
                est_aligned, ref_trans, plot_path,
                title=f"VGGT SLAM — {seq_name}{label_suffix}  "
                      f"|  ATE RMSE: {ate['rmse']:.3f} m",
            )

        return ate, rpe, scale

    raw_path = out_dir / f"{seq_name}_estimated_raw.txt"
    save_tum_trajectory(runner.get_estimated_poses_world(), raw_path)
    print(f"Raw trajectory saved → {raw_path}")

    suffix_base = "_nolc" if args.loop_closure else "_tum"
    ate_base, rpe_base, scale_base = evaluate(suffix_base)

    metrics_path = out_dir / f"{seq_name}_metrics{suffix_base}.txt"
    label_base = "Without loop closure" if args.loop_closure else "Results"
    save_metrics(ate_base, rpe_base, scale_base, metrics_path, label=label_base)
    print(f"Metrics saved → {metrics_path}")

    if args.loop_closure:
        lc_applied = runner.apply_loop_closure()
        if lc_applied:
            ate_lc, rpe_lc, scale_lc = evaluate("_lc")
            metrics_lc_path = out_dir / f"{seq_name}_metrics_lc.txt"
            save_metrics(
                ate_lc, rpe_lc, scale_lc, metrics_lc_path,
                label="With loop closure",
                before_ate=ate_base,
            )
            print(f"Metrics (with LC) saved → {metrics_lc_path}")
        else:
            print("\nNo loop closures applied — baseline metrics are final.")


if __name__ == '__main__':
    main()
