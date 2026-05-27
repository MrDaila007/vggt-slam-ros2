#!/usr/bin/env python3
"""
Test the vggt_slam_ros2 pipeline on a TUM RGB-D sequence without ROS2.

Usage
-----
  python scripts/test_on_tum.py --dataset /path/to/rgbd_dataset_freiburg1_desk

Expected dataset layout (standard TUM download):
  <dataset>/
    rgb/          ← colour frames
    rgb.txt       ← timestamp filename (one per line, # comments ok)
    groundtruth.txt ← timestamp tx ty tz qx qy qz qw

Optional arguments
------------------
  --window_size   16        frames per VGGT call
  --window_stride 8         new frames between calls
  --min_flow      10.0      min optical-flow to accept a keyframe (px)
  --conf_thr      20.0      filter bottom N% confidence points
  --checkpoint    facebook/VGGT-1B
  --out_dir       results/  where to write trajectory & plots
  --max_frames    0         cap on frames to process (0 = all)
  --no_plot                 skip matplotlib visualisation

Outputs (in --out_dir)
----------------------
  estimated_tum.txt     trajectory in TUM format (evo-compatible)
  metrics.txt           ATE RMSE, RPE RMSE, mean / max errors
  trajectory.png        top-down XZ and XY plots vs ground truth
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Allow running without installing the package (add project root to path)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_ROOT))

from vggt_slam_ros2.core.vggt_wrapper import VGGTWrapper
from vggt_slam_ros2.core.keyframe_selector import KeyframeSelector
from vggt_slam_ros2.core.sliding_window import SlidingWindow, Keyframe


# ===========================================================================
# TUM dataset loader
# ===========================================================================

def load_tum_associations(dataset_dir: Path, max_frames: int = 0) -> list[tuple[float, Path]]:
    """
    Parse rgb.txt and return list of (timestamp, abs_image_path).
    Lines starting with '#' are skipped.
    """
    rgb_txt = dataset_dir / "rgb.txt"
    if not rgb_txt.exists():
        raise FileNotFoundError(f"rgb.txt not found in {dataset_dir}")

    entries = []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            img_path = dataset_dir / parts[1]
            entries.append((ts, img_path))

    if max_frames > 0:
        entries = entries[:max_frames]
    return entries


def load_tum_groundtruth(dataset_dir: Path) -> dict[float, np.ndarray]:
    """
    Parse groundtruth.txt → dict{timestamp: (4,4) SE3 world-from-camera}.
    Format: timestamp tx ty tz qx qy qz qw
    """
    gt_txt = dataset_dir / "groundtruth.txt"
    if not gt_txt.exists():
        raise FileNotFoundError(f"groundtruth.txt not found in {dataset_dir}")

    gt = {}
    with open(gt_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            T = _quat_trans_to_se3(qx, qy, qz, qw, tx, ty, tz)
            gt[ts] = T
    return gt


def associate_timestamps(
    query_ts: list[float],
    gt: dict[float, np.ndarray],
    max_diff: float = 0.02,
) -> list[tuple[float, np.ndarray]]:
    """
    For each query timestamp find the closest ground-truth pose.
    Returns list of (query_ts, gt_pose) for matched pairs.
    Pairs with time difference > max_diff are dropped.
    """
    gt_ts = np.array(sorted(gt.keys()))
    result = []
    for qt in query_ts:
        idx = np.searchsorted(gt_ts, qt)
        candidates = []
        for i in [idx - 1, idx]:
            if 0 <= i < len(gt_ts):
                diff = abs(gt_ts[i] - qt)
                candidates.append((diff, gt_ts[i]))
        if not candidates:
            continue
        best_diff, best_ts = min(candidates)
        if best_diff <= max_diff:
            result.append((qt, gt[best_ts]))
    return result


# ===========================================================================
# Pipeline runner
# ===========================================================================

class TUMPipelineRunner:
    """
    Drives KeyframeSelector → SlidingWindow → VGGTWrapper on TUM images.
    Collects estimated poses for later evaluation.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        print(f"Loading VGGT from '{args.checkpoint}' ...")
        self._vggt = VGGTWrapper(checkpoint=args.checkpoint)
        print("VGGT ready.\n")

        self._kf_selector = KeyframeSelector(
            min_flow=args.min_flow,
            max_frames_between_keyframes=args.max_kf_gap,
        )
        self._window = SlidingWindow(
            window_size=args.window_size,
            stride=args.window_stride,
            callback=self._on_window_ready,
        )

        self._conf_thr = args.conf_thr
        self._overlap = args.window_size - args.window_stride

        # Results accumulated across windows:
        # list of (timestamp, (3,4) extrinsic cam-from-world)
        self._estimated: list[tuple[float, np.ndarray]] = []
        self._all_points: list[np.ndarray] = []
        self._all_colors: list[np.ndarray] = []

        self._window_count = 0
        self._total_infer_time = 0.0

    # ------------------------------------------------------------------

    def run(self, entries: list[tuple[float, Path]]) -> None:
        """Process all dataset entries sequentially."""
        n = len(entries)
        for i, (ts, img_path) in enumerate(entries):
            if i % 50 == 0:
                print(f"  Frame {i}/{n} ...", flush=True)

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"  Warning: could not read {img_path}")
                continue

            if not self._kf_selector.should_accept(bgr):
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._window.add(rgb, ts)

        # Flush any remaining frames in the buffer
        self._window.flush()
        print(f"\nProcessed {self._window_count} windows.")
        if self._window_count > 0:
            avg = self._total_infer_time / self._window_count
            print(f"Average inference time per window: {avg:.3f}s")

    # ------------------------------------------------------------------

    def _on_window_ready(self, frames: list[Keyframe]) -> None:
        images = [kf.image_rgb for kf in frames]
        stamps = [kf.stamp for kf in frames]

        t0 = time.monotonic()
        result = self._vggt.infer(images)
        dt = time.monotonic() - t0
        self._total_infer_time += dt
        self._window_count += 1

        S = result['extrinsics'].shape[0]
        colors_arr = np.stack([np.array(img, dtype=np.uint8) for img in images])

        new_start = self._overlap if self._window_count > 1 else 0

        for i in range(new_start, S):
            ext = result['extrinsics'][i]      # (3,4) cam-from-world
            self._estimated.append((stamps[i], ext))

            # Collect coloured points for optional visualisation
            pts = result['world_points'][i].reshape(-1, 3)
            col = colors_arr[i].reshape(-1, 3)
            conf = result['world_points_conf'][i].reshape(-1)
            thr = np.percentile(conf, self._conf_thr)
            mask = conf >= thr
            self._all_points.append(pts[mask].astype(np.float32))
            self._all_colors.append(col[mask])

        print(
            f"  Window {self._window_count}: {S} frames, "
            f"infer={dt:.2f}s, poses so far={len(self._estimated)}"
        )

    # ------------------------------------------------------------------

    def get_estimated_translations(self) -> tuple[list[float], np.ndarray]:
        """Return (timestamps, (N,3) world-from-camera translations)."""
        timestamps = [ts for ts, _ in self._estimated]
        # extrinsic is cam-from-world: t_world = -R^T @ t_cam
        translations = []
        for _, ext in self._estimated:
            R = ext[:3, :3]
            t = ext[:3, 3]
            t_world = -R.T @ t
            translations.append(t_world)
        return timestamps, np.array(translations)

    def get_estimated_poses_world(self) -> list[tuple[float, np.ndarray]]:
        """Return list of (ts, (4,4) world-from-camera SE3)."""
        result = []
        for ts, ext in self._estimated:
            R = ext[:3, :3]
            t = ext[:3, 3]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R.T
            T[:3, 3] = -R.T @ t
            result.append((ts, T))
        return result


# ===========================================================================
# Trajectory alignment  (Sim3 — handles unknown scale)
# ===========================================================================

def align_sim3(
    est: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute Sim(3) alignment: scale s, rotation R, translation t
    such that  est_aligned[i] = s * R @ est[i] + t  ≈ ref[i].

    Uses the Umeyama (1991) closed-form solution.

    Args:
        est: (N,3) estimated translations
        ref: (N,3) reference (ground-truth) translations

    Returns:
        est_aligned: (N,3)  aligned estimates
        scale:       float
        R:           (3,3) rotation
    """
    assert est.shape == ref.shape and est.ndim == 2
    N = est.shape[0]

    mu_e = est.mean(axis=0)
    mu_r = ref.mean(axis=0)

    est_c = est - mu_e
    ref_c = ref - mu_r

    sigma_e = (est_c ** 2).sum() / N
    H = (ref_c.T @ est_c) / N

    U, d, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    scale = float((d * S.diagonal()).sum() / sigma_e)
    t = mu_r - scale * R @ mu_e

    est_aligned = scale * (R @ est_c.T).T + mu_r
    return est_aligned, scale, R


# ===========================================================================
# Metrics
# ===========================================================================

def compute_ate(est_aligned: np.ndarray, ref: np.ndarray) -> dict:
    errors = np.linalg.norm(est_aligned - ref, axis=1)
    return {
        "rmse":   float(np.sqrt(np.mean(errors ** 2))),
        "mean":   float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max":    float(np.max(errors)),
        "min":    float(np.min(errors)),
        "std":    float(np.std(errors)),
    }


def compute_rpe(est_poses: list[np.ndarray], ref_poses: list[np.ndarray], delta: int = 1) -> dict:
    """
    Relative Pose Error (translation only) over steps of `delta`.
    est_poses / ref_poses: list of (4,4) world-from-camera SE3 matrices.
    """
    errors = []
    N = min(len(est_poses), len(ref_poses))
    for i in range(N - delta):
        # Relative motion: Q_i → Q_{i+delta}
        T_e_rel = np.linalg.inv(est_poses[i]) @ est_poses[i + delta]
        T_r_rel = np.linalg.inv(ref_poses[i]) @ ref_poses[i + delta]
        err = T_r_rel[:3, 3] - T_e_rel[:3, 3]
        errors.append(np.linalg.norm(err))
    errors = np.array(errors)
    return {
        "rmse":   float(np.sqrt(np.mean(errors ** 2))),
        "mean":   float(np.mean(errors)),
        "max":    float(np.max(errors)),
    }


# ===========================================================================
# I/O helpers
# ===========================================================================

def save_tum_trajectory(
    poses: list[tuple[float, np.ndarray]],
    out_path: Path,
) -> None:
    """Write trajectory in TUM format: timestamp tx ty tz qx qy qz qw."""
    with open(out_path, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in poses:
            t = T[:3, 3]
            qx, qy, qz, qw = _rot_to_quat(T[:3, :3])
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def save_metrics(metrics_ate: dict, metrics_rpe: dict, scale: float, out_path: Path) -> None:
    lines = [
        "=== ATE (Absolute Trajectory Error, after Sim3 alignment) ===",
        f"  RMSE   : {metrics_ate['rmse']:.4f} m",
        f"  Mean   : {metrics_ate['mean']:.4f} m",
        f"  Median : {metrics_ate['median']:.4f} m",
        f"  Std    : {metrics_ate['std']:.4f} m",
        f"  Max    : {metrics_ate['max']:.4f} m",
        "",
        "=== RPE (Relative Pose Error, delta=1) ===",
        f"  RMSE   : {metrics_rpe['rmse']:.4f} m",
        f"  Mean   : {metrics_rpe['mean']:.4f} m",
        f"  Max    : {metrics_rpe['max']:.4f} m",
        "",
        f"Sim3 scale factor applied: {scale:.4f}",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, 'w') as f:
        f.write(text + "\n")


def plot_trajectory(
    est_aligned: np.ndarray,
    ref: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    for ax, (xi, zi, xlabel, zlabel) in zip(
        axes,
        [(0, 2, 'X (m)', 'Z (m)'), (0, 1, 'X (m)', 'Y (m)')]
    ):
        ax.plot(ref[:, xi],         ref[:, zi],         'g-',  lw=1.5, label='Ground truth')
        ax.plot(est_aligned[:, xi], est_aligned[:, zi], 'b--', lw=1.5, label='Estimated (aligned)')
        ax.scatter(ref[0, xi],         ref[0, zi],         c='green',  s=60, zorder=5)
        ax.scatter(est_aligned[0, xi], est_aligned[0, zi], c='blue',   s=60, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(zlabel)
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Trajectory plot saved → {out_path}")


# ===========================================================================
# Quaternion / SE3 helpers
# ===========================================================================

def _quat_trans_to_se3(qx, qy, qz, qw, tx, ty, tz) -> np.ndarray:
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--dataset',        required=True,
                   help='Path to TUM RGB-D sequence directory')
    p.add_argument('--checkpoint',     default='facebook/VGGT-1B',
                   help='HuggingFace checkpoint or local path')
    p.add_argument('--window_size',    type=int,   default=16)
    p.add_argument('--window_stride',  type=int,   default=8)
    p.add_argument('--min_flow',       type=float, default=10.0,
                   help='Min optical-flow magnitude (px) for keyframe selection')
    p.add_argument('--max_kf_gap',     type=int,   default=30,
                   help='Max raw frames between forced keyframes')
    p.add_argument('--conf_thr',       type=float, default=20.0,
                   help='Filter bottom N%% confidence points (0=keep all)')
    p.add_argument('--out_dir',        default='results',
                   help='Output directory for trajectory, metrics, and plot')
    p.add_argument('--max_frames',     type=int,   default=0,
                   help='Cap number of dataset frames (0=all)')
    p.add_argument('--gt_max_diff',    type=float, default=0.02,
                   help='Max timestamp difference (s) for GT association')
    p.add_argument('--no_plot',        action='store_true',
                   help='Skip matplotlib visualisation')
    return p.parse_args()


def main() -> None:
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
    entries = load_tum_associations(dataset_dir, max_frames=args.max_frames)
    print(f"  {len(entries)} RGB frames\n")

    gt = load_tum_groundtruth(dataset_dir)
    print(f"  {len(gt)} ground-truth poses\n")

    # ---- Run pipeline -------------------------------------------------------
    print("Running VGGT pipeline ...")
    runner = TUMPipelineRunner(args)
    runner.run(entries)

    if len(runner._estimated) < 4:
        print("ERROR: fewer than 4 poses estimated — check dataset path and VGGT installation.")
        sys.exit(1)

    # ---- Save raw trajectory (evo-compatible) --------------------------------
    est_poses_world = runner.get_estimated_poses_world()
    raw_tum_path = out_dir / f"{seq_name}_estimated_raw.txt"
    save_tum_trajectory(est_poses_world, raw_tum_path)
    print(f"Raw trajectory saved → {raw_tum_path}")

    # ---- Associate with ground truth ----------------------------------------
    est_ts, est_trans = runner.get_estimated_translations()
    matched = associate_timestamps(est_ts, gt, max_diff=args.gt_max_diff)
    if len(matched) < 4:
        print(f"WARNING: only {len(matched)} GT matches — check timestamps or --gt_max_diff")
        sys.exit(1)

    print(f"\nMatched {len(matched)} / {len(est_ts)} estimated poses to ground truth.")

    # Build aligned arrays
    match_ts = [ts for ts, _ in matched]
    ref_trans = np.array([T[:3, 3] for _, T in matched])
    ref_poses_list = [T for _, T in matched]

    # Corresponding estimated translations (same order as matched)
    ts_to_idx = {ts: i for i, ts in enumerate(est_ts)}
    est_trans_matched = np.array([
        est_trans[ts_to_idx[ts]] for ts in match_ts
        if ts in ts_to_idx
    ])
    est_poses_matched = [
        est_poses_world[ts_to_idx[ts]][1]
        for ts in match_ts if ts in ts_to_idx
    ]

    N = min(len(est_trans_matched), len(ref_trans))
    est_trans_matched = est_trans_matched[:N]
    ref_trans         = ref_trans[:N]
    ref_poses_list    = ref_poses_list[:N]
    est_poses_matched = est_poses_matched[:N]

    # ---- Sim(3) alignment ---------------------------------------------------
    print("\nAligning trajectory (Sim3) ...")
    est_aligned, scale, _ = align_sim3(est_trans_matched, ref_trans)
    print(f"  Scale factor: {scale:.4f}")

    # ---- Metrics ------------------------------------------------------------
    metrics_ate = compute_ate(est_aligned, ref_trans)
    metrics_rpe = compute_rpe(est_poses_matched, ref_poses_list, delta=1)

    metrics_path = out_dir / f"{seq_name}_metrics.txt"
    save_metrics(metrics_ate, metrics_rpe, scale, metrics_path)
    print(f"Metrics saved → {metrics_path}")

    # ---- Save aligned trajectory --------------------------------------------
    aligned_tum_path = out_dir / f"{seq_name}_estimated_tum.txt"
    aligned_poses = []
    for i, (ts, T) in enumerate(est_poses_world[:N]):
        R_orig = T[:3, :3]
        t_aligned = est_aligned[i]
        T_aligned = np.eye(4)
        T_aligned[:3, :3] = R_orig
        T_aligned[:3, 3] = t_aligned
        aligned_poses.append((ts, T_aligned))
    save_tum_trajectory(aligned_poses, aligned_tum_path)
    print(f"Aligned trajectory saved → {aligned_tum_path}")
    print(f"  (Use with evo_ape: evo_ape tum groundtruth.txt {aligned_tum_path} -a)")

    # ---- Plot ---------------------------------------------------------------
    if not args.no_plot:
        plot_path = out_dir / f"{seq_name}_trajectory.png"
        plot_trajectory(
            est_aligned, ref_trans, plot_path,
            title=f"VGGT SLAM — {seq_name}  |  ATE RMSE: {metrics_ate['rmse']:.3f} m",
        )


if __name__ == '__main__':
    main()
