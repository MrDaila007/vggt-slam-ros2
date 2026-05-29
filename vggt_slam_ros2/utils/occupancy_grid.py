"""
Project a 3-D world-frame point cloud into a 2-D occupancy grid
for Nav2 consumption (Stage 4.2).

Pure-numpy, no ROS2 dependency — fully unit-testable without a runtime.

Algorithm
---------
1. Filter points to a configurable height band [z_min, z_max].
2. Discretise (x, y) into cells of `resolution` metres.
3. Mark a cell OCCUPIED (100) if it has >= min_points hits, FREE (0) otherwise.

Coordinate convention (ROS2 REP-105):
  x-forward, y-left, z-up.  Grid origin = lower-left corner (min x, min y).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CELL_FREE = np.int8(0)
CELL_OCCUPIED = np.int8(100)
CELL_UNKNOWN = np.int8(-1)


@dataclass
class OccupancyGridData:
    """ROS-independent occupancy grid representation."""
    data: np.ndarray    # (height, width) int8 — row-major, origin at (0,0)
    resolution: float   # metres per cell
    origin_x: float     # world x of the lower-left corner of cell (col=0, row=0)
    origin_y: float     # world y of the lower-left corner of cell (col=0, row=0)
    width: int          # number of columns  (x direction)
    height: int         # number of rows     (y direction)


def build_occupancy_grid(
    points: np.ndarray,
    resolution: float = 0.05,
    z_min: float = 0.1,
    z_max: float = 2.0,
    min_points: int = 2,
    padding_cells: int = 5,
) -> OccupancyGridData | None:
    """
    Build a 2-D occupancy grid from a world-frame point cloud.

    Parameters
    ----------
    points        (N, 3) float array — world-frame XYZ
    resolution    cell size in metres
    z_min         minimum z to include (filters out floor reflections)
    z_max         maximum z to include (filters out ceiling)
    min_points    minimum hits for a cell to be marked OCCUPIED
    padding_cells empty border around the occupied bounding box

    Returns None if no points survive the height filter.
    """
    if points is None or len(points) == 0:
        return None

    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    pts = points[mask]
    if len(pts) == 0:
        return None

    xy = pts[:, :2]
    pad = padding_cells * resolution

    x_min = float(xy[:, 0].min()) - pad
    y_min = float(xy[:, 1].min()) - pad
    x_max = float(xy[:, 0].max()) + pad
    y_max = float(xy[:, 1].max()) + pad

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution)))

    col = np.clip(
        np.floor((xy[:, 0] - x_min) / resolution).astype(np.int32),
        0, width - 1,
    )
    row = np.clip(
        np.floor((xy[:, 1] - y_min) / resolution).astype(np.int32),
        0, height - 1,
    )

    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (row, col), 1)

    grid = np.where(counts >= min_points, CELL_OCCUPIED, CELL_FREE).astype(np.int8)

    return OccupancyGridData(
        data=grid,
        resolution=resolution,
        origin_x=x_min,
        origin_y=y_min,
        width=width,
        height=height,
    )
