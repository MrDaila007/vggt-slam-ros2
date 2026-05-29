"""
Unit tests for the EuRoC dataset loader functions in scripts/test_on_euroc.py.

Tests cover:
  - Timestamp parsing (nanoseconds → seconds)
  - Image path construction
  - Ground-truth quaternion → SE(3) conversion
  - Graceful handling of comment lines and empty files
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts/ to path so we can import test_on_euroc directly
_SCRIPT_DIR = Path(__file__).resolve().parent.parent / 'scripts'
sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Helpers to create fake EuRoC dataset on disk
# ---------------------------------------------------------------------------

def _make_euroc_dataset(tmp_path: Path, n_frames: int = 3, n_gt: int = 5) -> Path:
    """Create a minimal fake EuRoC dataset directory structure."""
    cam0 = tmp_path / 'mav0' / 'cam0'
    (cam0 / 'data').mkdir(parents=True)
    gt_dir = tmp_path / 'mav0' / 'state_groundtruth_estimate0'
    gt_dir.mkdir(parents=True)

    # cam0/data.csv — nanosecond timestamps
    lines = ['# timestamp [ns],filename\n']
    base_ns = 1_403_636_579_763_555_584
    for i in range(n_frames):
        ts_ns = base_ns + i * 50_000_000        # 50 ms between frames
        fname = f'{ts_ns}.png'
        lines.append(f'{ts_ns},{fname}\n')
        # Create a dummy PNG placeholder
        (cam0 / 'data' / fname).write_bytes(b'')

    (cam0 / 'data.csv').write_text(''.join(lines))

    # state_groundtruth_estimate0/data.csv
    gt_lines = [
        '# timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],'
        ' q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [],'
        ' v_RS_R_x [m/s], v_RS_R_y [m/s], v_RS_R_z [m/s],'
        ' b_w_RS_S_x [rad/s], b_w_RS_S_y [rad/s], b_w_RS_S_z [rad/s],'
        ' b_a_RS_S_x [m/s^2], b_a_RS_S_y [m/s^2], b_a_RS_S_z [m/s^2]\n'
    ]
    for i in range(n_gt):
        ts_ns = base_ns + i * 10_000_000        # 10 ms between GT poses
        tx, ty, tz = float(i) * 0.1, 0.0, 0.0
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0   # identity rotation
        gt_lines.append(
            f'{ts_ns},{tx},{ty},{tz},{qw},{qx},{qy},{qz},'
            '0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n'
        )
    (gt_dir / 'data.csv').write_text(''.join(gt_lines))

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadEurocAssociations:

    def test_returns_correct_count(self, tmp_path):
        from test_on_euroc import load_euroc_associations
        _make_euroc_dataset(tmp_path, n_frames=4)
        entries = load_euroc_associations(tmp_path)
        assert len(entries) == 4

    def test_timestamps_in_seconds_range(self, tmp_path):
        from test_on_euroc import load_euroc_associations
        _make_euroc_dataset(tmp_path, n_frames=3)
        entries = load_euroc_associations(tmp_path)
        for ts, _ in entries:
            # Reasonable seconds since epoch (2010–2040)
            assert 1.2e9 < ts < 2.5e9

    def test_image_paths_are_absolute(self, tmp_path):
        from test_on_euroc import load_euroc_associations
        _make_euroc_dataset(tmp_path, n_frames=2)
        entries = load_euroc_associations(tmp_path)
        for _, p in entries:
            assert p.is_absolute()

    def test_max_frames_cap(self, tmp_path):
        from test_on_euroc import load_euroc_associations
        _make_euroc_dataset(tmp_path, n_frames=5)
        entries = load_euroc_associations(tmp_path, max_frames=2)
        assert len(entries) == 2

    def test_comment_lines_skipped(self, tmp_path):
        from test_on_euroc import load_euroc_associations
        _make_euroc_dataset(tmp_path, n_frames=3)
        # Prepend extra comment lines
        csv = tmp_path / 'mav0' / 'cam0' / 'data.csv'
        original = csv.read_text()
        csv.write_text('# extra comment\n# another\n' + original)
        entries = load_euroc_associations(tmp_path)
        assert len(entries) == 3

    def test_missing_csv_raises(self, tmp_path):
        from test_on_euroc import load_euroc_associations
        with pytest.raises(FileNotFoundError):
            load_euroc_associations(tmp_path)


class TestLoadEurocGroundtruth:

    def test_returns_correct_count(self, tmp_path):
        from test_on_euroc import load_euroc_groundtruth
        _make_euroc_dataset(tmp_path, n_gt=6)
        gt = load_euroc_groundtruth(tmp_path)
        assert len(gt) == 6

    def test_timestamps_in_seconds(self, tmp_path):
        from test_on_euroc import load_euroc_groundtruth
        _make_euroc_dataset(tmp_path, n_gt=3)
        gt = load_euroc_groundtruth(tmp_path)
        for ts in gt:
            assert 1.2e9 < ts < 2.5e9

    def test_pose_is_4x4_se3(self, tmp_path):
        from test_on_euroc import load_euroc_groundtruth
        _make_euroc_dataset(tmp_path, n_gt=2)
        gt = load_euroc_groundtruth(tmp_path)
        for T in gt.values():
            assert T.shape == (4, 4)
            # Bottom row must be [0 0 0 1]
            np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-10)

    def test_identity_rotation_for_unit_quaternion(self, tmp_path):
        from test_on_euroc import load_euroc_groundtruth
        _make_euroc_dataset(tmp_path, n_gt=2)
        gt = load_euroc_groundtruth(tmp_path)
        for T in gt.values():
            R = T[:3, :3]
            np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_translation_matches_csv(self, tmp_path):
        from test_on_euroc import load_euroc_groundtruth
        _make_euroc_dataset(tmp_path, n_gt=3)
        gt = load_euroc_groundtruth(tmp_path)
        poses = sorted(gt.items())
        # First pose: tx=0.0, second: tx=0.1, third: tx=0.2
        np.testing.assert_allclose(poses[0][1][:3, 3], [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(poses[1][1][:3, 3], [0.1, 0.0, 0.0], atol=1e-9)

    def test_missing_gt_csv_raises(self, tmp_path):
        from test_on_euroc import load_euroc_groundtruth
        with pytest.raises(FileNotFoundError):
            load_euroc_groundtruth(tmp_path)
