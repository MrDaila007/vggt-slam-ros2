"""Geometry helpers for SLAM pipeline."""

import numpy as np
import cv2


def compute_optical_flow_disparity(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Return mean optical flow magnitude between two grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def relative_rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Return the rotation angle (degrees) between two (3,3) rotation matrices."""
    R_rel = R1.T @ R2
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def filter_points_by_confidence(
    points: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    threshold_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove low-confidence points.  Returns filtered (points, colors)."""
    threshold = np.percentile(conf, threshold_percentile)
    mask = conf >= threshold
    return points[mask], colors[mask]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a (4,4) SE3 transform to an (N,3) point array."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3]


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Invert a (4,4) SE3 matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def normalize_scale(poses: np.ndarray, ref_idx: int = 0) -> tuple[np.ndarray, float]:
    """
    Normalize a trajectory so that the mean distance between consecutive poses is 1.
    Returns (normalized_poses, scale_factor).
    poses: (N, 3, 4) extrinsic matrices.
    """
    translations = poses[:, :3, 3]
    diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    mean_dist = float(np.mean(diffs[diffs > 1e-6]))
    if mean_dist < 1e-8:
        return poses, 1.0

    scale = 1.0 / mean_dist
    scaled = poses.copy()
    scaled[:, :3, 3] *= scale
    return scaled, scale


def sliding_window_indices(total: int, window: int, stride: int) -> list[list[int]]:
    """
    Produce overlapping windows over [0, total).
    Each window has `window` frames; windows advance by `stride`.
    Last window is padded to include the final frame.
    """
    windows = []
    start = 0
    while start < total:
        end = min(start + window, total)
        windows.append(list(range(start, end)))
        if end == total:
            break
        start += stride
    return windows
