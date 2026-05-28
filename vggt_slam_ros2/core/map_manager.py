"""
Incremental map manager.

Stores the global point cloud and camera trajectory.
Supports incremental insertion and optional voxel downsampling
to keep memory bounded.

Design notes:
  - Uses a numpy-based accumulator rather than Open3D to avoid heavy dependencies
    at import time; Open3D is only used for optional voxel downsampling.
  - Scale across windows is kept consistent by anchoring each new window to the
    overlap frames whose world positions are already in the map.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class MapFrame:
    """A single localised camera frame stored in the global map."""
    global_idx: int
    stamp: float
    extrinsic: np.ndarray   # (3,4) cam-from-world
    intrinsic: np.ndarray   # (3,3)


class MapManager:
    def __init__(self, voxel_size: Optional[float] = None) -> None:
        self.voxel_size = voxel_size

        self._points: list[np.ndarray] = []     # each (N_i, 3)
        self._colors: list[np.ndarray] = []     # each (N_i, 3) uint8
        self._frames: list[MapFrame] = []
        self._total_points: int = 0

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_window_result(
        self,
        global_indices: list[int],
        stamps: list[float],
        extrinsics: np.ndarray,     # (S, 3, 4)
        intrinsics: np.ndarray,     # (S, 3, 3)
        world_points: np.ndarray,   # (S, H, W, 3)
        colors: np.ndarray,         # (S, H, W, 3) uint8
        conf: np.ndarray,           # (S, H, W)
        conf_threshold_pct: float,
        overlap: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate a VGGT window output into the global map.
        `overlap` frames at the start of the window are already in the map;
        only the remaining (S - overlap) new frames contribute new points.

        Returns (new_points, new_colors) for incremental publishing.
        """
        S = extrinsics.shape[0]
        new_start = overlap  # skip already-integrated overlap frames

        new_pts_list = []
        new_col_list = []

        for i in range(new_start, S):
            pts_flat = world_points[i].reshape(-1, 3)      # (H*W, 3)
            col_flat = colors[i].reshape(-1, 3)            # (H*W, 3)
            conf_flat = conf[i].reshape(-1)                # (H*W,)

            threshold = np.percentile(conf_flat, conf_threshold_pct)
            mask = conf_flat >= threshold
            pts_clean = pts_flat[mask]
            col_clean = col_flat[mask]

            if pts_clean.shape[0] > 0:
                new_pts_list.append(pts_clean)
                new_col_list.append(col_clean)

            # Store the frame pose
            global_idx = global_indices[i] if i < len(global_indices) else -1
            stamp = stamps[i] if i < len(stamps) else 0.0
            self._frames.append(MapFrame(
                global_idx=global_idx,
                stamp=stamp,
                extrinsic=extrinsics[i].copy(),
                intrinsic=intrinsics[i].copy(),
            ))

        if not new_pts_list:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        new_pts = np.concatenate(new_pts_list, axis=0).astype(np.float32)
        new_col = np.concatenate(new_col_list, axis=0).astype(np.uint8)

        if self.voxel_size is not None:
            new_pts, new_col = self._voxel_downsample(new_pts, new_col)

        self._points.append(new_pts)
        self._colors.append(new_col)
        self._total_points += new_pts.shape[0]
        return new_pts, new_col

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_all_points(self) -> np.ndarray:
        if not self._points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(self._points, axis=0)

    def get_all_colors(self) -> np.ndarray:
        if not self._colors:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.concatenate(self._colors, axis=0)

    def get_trajectory(self) -> list[MapFrame]:
        return list(self._frames)

    def total_points(self) -> int:
        return self._total_points

    def reset(self) -> None:
        self._points.clear()
        self._colors.clear()
        self._frames.clear()
        self._total_points = 0

    def save_to_file(self, path: str, fmt: str = "pcd") -> bool:
        """
        Save the accumulated point cloud to disk.

        Parameters
        ----------
        path : absolute file path (extension is replaced with `fmt`)
        fmt  : "pcd", "ply", or "npz"

        Returns True on success, False on failure.
        Open3D is used for pcd/ply; falls back to npz if not available.
        """
        from pathlib import Path as _Path

        pts = self.get_all_points()
        cols = self.get_all_colors()

        if pts.shape[0] == 0:
            return False

        out_path = _Path(path).with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt in ("pcd", "ply"):
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
                o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
                return True
            except ImportError:
                fmt = "npz"   # fall through to npz
                out_path = _Path(path).with_suffix(".npz")

        np.savez_compressed(str(out_path), points=pts, colors=cols)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _voxel_downsample_impl(
        pts: np.ndarray, cols: np.ndarray, voxel_size: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Voxel grid filter — uses Open3D if available, else passthrough."""
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
            pcd_ds = pcd.voxel_down_sample(voxel_size)
            pts_ds = np.asarray(pcd_ds.points, dtype=np.float32)
            cols_ds = (np.asarray(pcd_ds.colors) * 255).astype(np.uint8)
            return pts_ds, cols_ds
        except ImportError:
            return pts, cols

    def _voxel_downsample(
        self, pts: np.ndarray, cols: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return MapManager._voxel_downsample_impl(pts, cols, self.voxel_size or 0.05)
