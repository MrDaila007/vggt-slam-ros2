"""Shared pytest fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable without a full colcon build
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Common array factories ────────────────────────────────────────────────────

@pytest.fixture
def random_bgr_frame():
    """Return a random 64×64 BGR uint8 image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def black_bgr_frame():
    """Return a flat black 64×64 BGR image (zero optical flow)."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def dummy_extrinsic():
    """Identity cam-from-world extrinsic (3×4)."""
    return np.eye(4, dtype=np.float64)[:3, :]


@pytest.fixture
def dummy_intrinsic():
    """Typical 640×480 pinhole intrinsic matrix."""
    return np.array([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)
