"""Unit tests for core/image_retrieval.py (no GPU required)."""

import numpy as np
import pytest

from vggt_slam_ros2.core.image_retrieval import ImageRetrieval, LoopCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockImageRetrieval(ImageRetrieval):
    """
    Subclass that bypasses the real DINOv2 model so we can test the
    retrieval logic with controlled embeddings.
    """

    def __init__(self, embeddings_sequence: list[np.ndarray], **kwargs) -> None:
        super().__init__(**kwargs)
        self._embed_seq = embeddings_sequence
        self._embed_call = 0
        # mark model as "loaded"
        self._model = object()
        self._processor = object()

    def _embed(self, image_rgb: np.ndarray) -> np.ndarray:
        emb = self._embed_seq[self._embed_call % len(self._embed_seq)]
        self._embed_call += 1
        # normalise like the real model does
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


def _unit_vec(d: int, idx: int) -> np.ndarray:
    """Return a D-dim unit vector with a 1 at position idx."""
    v = np.zeros(d, dtype=np.float32)
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Basic add_and_query behaviour
# ---------------------------------------------------------------------------

class TestAddAndQuery:
    def test_first_frame_returns_none(self):
        embs = [_unit_vec(128, 0)] * 5
        r = _MockImageRetrieval(embs, similarity_threshold=0.8, min_time_gap=1.0)
        result = r.add_and_query(np.zeros((8, 8, 3), dtype=np.uint8), stamp=0.0)
        assert result is None

    def test_no_loop_when_dissimilar(self):
        # Each frame has an orthogonal embedding → similarity = 0
        embs = [_unit_vec(4, i) for i in range(4)]
        r = _MockImageRetrieval(embs, similarity_threshold=0.8, min_time_gap=0.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r.add_and_query(img, stamp=0.0)
        r.add_and_query(img, stamp=1.0)
        result = r.add_and_query(img, stamp=2.0)
        assert result is None

    def test_loop_detected_when_similar(self):
        # Frame 0 and frame 2 share the same embedding → similarity = 1.0
        emb0 = _unit_vec(128, 0)
        emb1 = _unit_vec(128, 1)  # different
        embs = [emb0, emb1, emb0]  # frame 2 matches frame 0
        r = _MockImageRetrieval(embs, similarity_threshold=0.9, min_time_gap=0.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r.add_and_query(img, stamp=0.0)
        r.add_and_query(img, stamp=1.0)
        result = r.add_and_query(img, stamp=2.0)

        assert result is not None
        assert isinstance(result, LoopCandidate)
        assert result.match_idx == 0
        assert result.query_idx == 2
        assert abs(result.similarity - 1.0) < 1e-5

    def test_returns_best_match(self):
        # Frame 3: best match is frame 1 (closer direction than frame 0)
        emb_base = np.ones(4, dtype=np.float32) / 2.0  # frame 0
        emb_close = np.array([0.9, 0.1, 0.1, 0.1], dtype=np.float32)  # frame 1
        emb_query = np.array([0.9, 0.1, 0.1, 0.1], dtype=np.float32)  # frame 3 = same as frame 1
        emb_other = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # frame 2

        embs = [emb_base, emb_close, emb_other, emb_query]
        r = _MockImageRetrieval(embs, similarity_threshold=0.5, min_time_gap=0.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        for stamp in range(3):
            r.add_and_query(img, stamp=float(stamp))
        result = r.add_and_query(img, stamp=3.0)

        assert result is not None
        assert result.match_idx == 1  # closest to emb_close

    def test_no_loop_within_min_time_gap(self):
        emb = _unit_vec(128, 0)
        embs = [emb] * 10
        r = _MockImageRetrieval(embs, similarity_threshold=0.5, min_time_gap=5.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        # All frames within 4 seconds — time gap filter should suppress matches
        for i in range(5):
            result = r.add_and_query(img, stamp=float(i))
        assert result is None  # stamp 4, oldest is 0 → gap = 4 < 5.0

    def test_loop_when_time_gap_met(self):
        emb = _unit_vec(128, 0)
        embs = [emb] * 10
        r = _MockImageRetrieval(embs, similarity_threshold=0.5, min_time_gap=3.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r.add_and_query(img, stamp=0.0)   # frame 0
        r.add_and_query(img, stamp=1.0)   # frame 1
        r.add_and_query(img, stamp=2.0)   # frame 2
        result = r.add_and_query(img, stamp=4.0)  # frame 3 — gap to frame 0 is 4.0 ≥ 3.0
        assert result is not None
        assert result.match_idx == 0

    def test_size_increments(self):
        embs = [_unit_vec(128, i % 128) for i in range(6)]
        r = _MockImageRetrieval(embs, similarity_threshold=0.8, min_time_gap=0.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert r.size == 0
        for i in range(5):
            r.add_and_query(img, stamp=float(i))
        assert r.size == 5

    def test_reset_clears_database(self):
        emb = _unit_vec(128, 0)
        r = _MockImageRetrieval([emb] * 5, similarity_threshold=0.5, min_time_gap=0.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r.add_and_query(img, stamp=0.0)
        r.add_and_query(img, stamp=1.0)
        r.reset()
        assert r.size == 0
        result = r.add_and_query(img, stamp=2.0)
        assert result is None  # first frame after reset, no previous frames


# ---------------------------------------------------------------------------
# LoopCandidate fields
# ---------------------------------------------------------------------------

class TestLoopCandidateFields:
    def test_stamps_recorded_correctly(self):
        emb = _unit_vec(64, 0)
        r = _MockImageRetrieval([emb] * 5, similarity_threshold=0.5, min_time_gap=0.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r.add_and_query(img, stamp=10.0)
        result = r.add_and_query(img, stamp=20.0)

        assert result is not None
        assert abs(result.query_stamp - 20.0) < 1e-9
        assert abs(result.match_stamp - 10.0) < 1e-9

    def test_similarity_within_bounds(self):
        emb = _unit_vec(64, 0)
        r = _MockImageRetrieval([emb] * 3, similarity_threshold=0.5, min_time_gap=0.0)
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r.add_and_query(img, stamp=0.0)
        result = r.add_and_query(img, stamp=1.0)
        assert result is not None
        assert 0.0 <= result.similarity <= 1.0 + 1e-6
