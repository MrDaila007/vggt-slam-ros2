"""
Scale anchoring across sliding-window VGGT calls.

VGGT infers geometry up to an unknown scale per window.  Consecutive
windows share `overlap` frames; those frames appear in both windows and
their known global positions can be used to compute a Sim(3) correction
(scale + rotation + translation) that maps the current window's coordinate
frame onto the accumulated global map.

Algorithm (Umeyama 1991 closed-form Sim(3)):
  1. First window: treat as global reference, store last `overlap` camera
     positions.
  2. Every subsequent window: align the first `overlap` camera positions
     (current window frame) to the stored global positions.
  3. Apply the Sim(3) transform to all new frames and points.

Rotation convention: extrinsics are (3, 4) cam-from-world [R | t] where
  p_cam = R @ p_world + t
  camera_position_in_world = -R^T @ t
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class ScaleAnchor:
    """
    Maintains inter-window Sim(3) consistency for the sliding-window SLAM pipeline.
    """

    def __init__(self, min_overlap: int = 4) -> None:
        self._min_overlap = min_overlap
        # Camera positions (world-from-cam) for the last `overlap` frames,
        # stored in global coordinates after each window.
        self._prev_overlap_global: Optional[np.ndarray] = None  # (overlap, 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        extrinsics: np.ndarray,    # (S, 3, 4) cam-from-world, current window frame
        world_points: np.ndarray,  # (S, H, W, 3)
        overlap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align the current window to the global map via Sim(3) on overlap frames.

        Returns
        -------
        extrinsics_global : (S, 3, 4)  corrected cam-from-world
        world_points_global : (S, H, W, 3)  corrected world points
        """
        cam_pos = _cam_positions(extrinsics)  # (S, 3)

        if self._prev_overlap_global is None or overlap < self._min_overlap:
            self._store_overlap(cam_pos, overlap)
            return extrinsics, world_points

        # Align overlap region: first `overlap` frames of the current window
        # correspond to the last `overlap` frames of the previous window.
        curr_overlap = cam_pos[:overlap]  # (overlap, 3)
        ref_overlap = self._prev_overlap_global  # (overlap, 3)

        scale, R_sim3, t_sim3 = _umeyama(curr_overlap, ref_overlap)

        extrinsics_global = _apply_sim3_extrinsics(extrinsics, scale, R_sim3, t_sim3)
        world_points_global = _apply_sim3_points(world_points, scale, R_sim3, t_sim3)

        corrected_pos = _cam_positions(extrinsics_global)
        self._store_overlap(corrected_pos, overlap)

        return extrinsics_global, world_points_global

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prev_overlap_global = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_overlap(self, cam_pos: np.ndarray, overlap: int) -> None:
        n = max(overlap, 1)
        self._prev_overlap_global = cam_pos[-n:].copy()


# ===========================================================================
# Pure functions
# ===========================================================================

def _cam_positions(extrinsics: np.ndarray) -> np.ndarray:
    """
    (S, 3, 4) cam-from-world  →  (S, 3) camera positions in world frame.
    position_i = -R_i^T @ t_i
    """
    R = extrinsics[:, :3, :3]   # (S, 3, 3)
    t = extrinsics[:, :3, 3]    # (S, 3)
    # batched: pos[i] = -R[i].T @ t[i]
    return -np.einsum('sij,sj->si', R.transpose(0, 2, 1), t)


def _umeyama(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Closed-form Sim(3) alignment (Umeyama 1991).

    Finds scale, R, t such that  dst ≈ scale * R @ src + t.

    Parameters
    ----------
    src, dst : (N, 3)

    Returns
    -------
    scale : float
    R     : (3, 3) rotation
    t     : (3,) translation
    """
    assert src.shape == dst.shape and src.ndim == 2, "src and dst must be (N, D)"
    N = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = float((src_c ** 2).sum() / N)
    H = (dst_c.T @ src_c) / N

    U, d, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    if sigma_src < 1e-10:
        scale = 1.0
    else:
        scale = float((d * S.diagonal()).sum() / sigma_src)

    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def _apply_sim3_extrinsics(
    extrinsics: np.ndarray,
    scale: float,
    R_sim3: np.ndarray,
    t_sim3: np.ndarray,
) -> np.ndarray:
    """
    Apply Sim(3) correction to cam-from-world extrinsics.

    If the Sim(3) maps current → global:  p_global = scale * R_sim3 @ p_curr + t_sim3
    then the corrected cam-from-world extrinsic for frame i is:
        R_new = R_ext @ R_sim3^T          (proper rotation)
        t_new = scale * t_ext - R_ext @ R_sim3^T @ t_sim3

    This satisfies:  -R_new^T @ t_new = scale * R_sim3 @ pos_curr + t_sim3  ✓
    """
    R_ext = extrinsics[:, :3, :3]  # (S, 3, 3)
    t_ext = extrinsics[:, :3, 3]   # (S, 3)

    R_sim3T = R_sim3.T              # (3, 3)
    R_new = R_ext @ R_sim3T         # (S, 3, 3)
    # t_new[i] = scale * t_ext[i] - R_ext[i] @ R_sim3^T @ t_sim3
    correction = R_ext @ (R_sim3T @ t_sim3)   # (S, 3)
    t_new = scale * t_ext - correction          # (S, 3)

    result = np.empty_like(extrinsics)
    result[:, :3, :3] = R_new
    result[:, :3, 3] = t_new
    return result


def _apply_sim3_points(
    world_points: np.ndarray,
    scale: float,
    R_sim3: np.ndarray,
    t_sim3: np.ndarray,
) -> np.ndarray:
    """
    Apply Sim(3) to world points: p_global = scale * R_sim3 @ p_curr + t_sim3

    world_points : (S, H, W, 3)
    """
    S, H, W, _ = world_points.shape
    pts = world_points.reshape(-1, 3)                   # (S*H*W, 3)
    pts_global = scale * (R_sim3 @ pts.T).T + t_sim3   # (S*H*W, 3)
    return pts_global.reshape(S, H, W, 3)
