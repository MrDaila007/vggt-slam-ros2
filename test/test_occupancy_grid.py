"""Unit tests for utils/occupancy_grid.py — no ROS2 or GPU required."""

from __future__ import annotations

import numpy as np
import pytest

from vggt_slam_ros2.utils.occupancy_grid import (
    build_occupancy_grid,
    OccupancyGridData,
    CELL_FREE,
    CELL_OCCUPIED,
)


def _grid_pts(n: int = 100, z: float = 1.0) -> np.ndarray:
    """Return a dense cluster of points at z=z in a 1×1 m patch."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 1.0, (n, 2))
    return np.column_stack([xy, np.full(n, z)])


class TestBuildOccupancyGrid:

    def test_returns_occupancy_grid_data(self):
        pts = _grid_pts()
        result = build_occupancy_grid(pts)
        assert isinstance(result, OccupancyGridData)

    def test_empty_array_returns_none(self):
        assert build_occupancy_grid(np.empty((0, 3))) is None

    def test_none_returns_none(self):
        assert build_occupancy_grid(None) is None  # type: ignore[arg-type]

    def test_all_below_z_min_returns_none(self):
        pts = _grid_pts(z=0.0)
        assert build_occupancy_grid(pts, z_min=0.1) is None

    def test_all_above_z_max_returns_none(self):
        pts = _grid_pts(z=5.0)
        assert build_occupancy_grid(pts, z_max=2.0) is None

    def test_grid_dimensions_positive(self):
        pts = _grid_pts()
        g = build_occupancy_grid(pts, resolution=0.1)
        assert g.width > 0 and g.height > 0

    def test_data_shape_matches_dimensions(self):
        pts = _grid_pts()
        g = build_occupancy_grid(pts, resolution=0.1)
        assert g.data.shape == (g.height, g.width)

    def test_occupied_cells_present_in_dense_cluster(self):
        pts = _grid_pts(n=500)
        g = build_occupancy_grid(pts, resolution=0.05, min_points=1)
        assert (g.data == CELL_OCCUPIED).any()

    def test_sparse_points_all_free_with_high_min_points(self):
        pts = _grid_pts(n=5)
        g = build_occupancy_grid(pts, resolution=0.5, min_points=100)
        assert (g.data == CELL_FREE).all()

    def test_origin_less_than_min_xy(self):
        pts = np.array([[1.0, 2.0, 1.0], [1.5, 2.5, 1.0]])
        g = build_occupancy_grid(pts, resolution=0.1, padding_cells=0)
        assert g.origin_x <= 1.0
        assert g.origin_y <= 2.0

    def test_padding_increases_grid_size(self):
        pts = _grid_pts()
        g_no_pad = build_occupancy_grid(pts, resolution=0.1, padding_cells=0)
        g_pad = build_occupancy_grid(pts, resolution=0.1, padding_cells=5)
        assert g_pad.width > g_no_pad.width
        assert g_pad.height > g_no_pad.height

    def test_resolution_stored_correctly(self):
        pts = _grid_pts()
        g = build_occupancy_grid(pts, resolution=0.07)
        assert g.resolution == pytest.approx(0.07)

    def test_cell_values_are_0_or_100(self):
        pts = _grid_pts(n=200)
        g = build_occupancy_grid(pts, resolution=0.1, min_points=2)
        unique = set(np.unique(g.data).tolist())
        assert unique.issubset({int(CELL_FREE), int(CELL_OCCUPIED)})

    def test_finer_resolution_gives_larger_grid(self):
        pts = _grid_pts()
        g_coarse = build_occupancy_grid(pts, resolution=0.5, padding_cells=0)
        g_fine = build_occupancy_grid(pts, resolution=0.05, padding_cells=0)
        assert g_fine.width >= g_coarse.width
        assert g_fine.height >= g_coarse.height

    def test_height_filter_excludes_out_of_band(self):
        in_band = _grid_pts(n=100, z=1.0)
        out_band = _grid_pts(n=100, z=5.0)
        pts_all = np.vstack([in_band, out_band])
        g_all = build_occupancy_grid(pts_all, z_min=0.1, z_max=2.0,
                                     resolution=0.1, min_points=1)
        g_in = build_occupancy_grid(in_band, z_min=0.1, z_max=2.0,
                                    resolution=0.1, min_points=1)
        # Grids should be identical — out-of-band points are ignored
        assert g_all is not None and g_in is not None
        np.testing.assert_array_equal(g_all.data, g_in.data)
