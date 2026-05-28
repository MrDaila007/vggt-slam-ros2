"""
Image-based loop closure detection using DINOv2 embeddings.

DINOv2 (ViT-B/14, Apache-2.0) embeds keyframes into a descriptor space.
At each new keyframe, cosine similarity is computed against all previous
embeddings. A loop candidate is returned when:
  - similarity > threshold
  - the matched frame is at least `min_time_gap` seconds in the past
    (avoids matching adjacent frames)

DINOv2 was chosen over NetVLAD because:
  - Apache-2.0 license — no commercial restrictions
  - Available on HuggingFace without manual download
  - Competitive recall on indoor scenes (TUM, EuRoC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_DINOV2_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    _DINOV2_AVAILABLE = True
except ImportError:
    pass


_DEFAULT_MODEL = "facebook/dinov2-base"


@dataclass
class LoopCandidate:
    """A detected loop closure candidate."""
    query_idx: int        # index of the current keyframe in the database
    match_idx: int        # index of the matched past keyframe
    query_stamp: float    # timestamp of the current keyframe
    match_stamp: float    # timestamp of the matched keyframe
    similarity: float     # cosine similarity ∈ (0, 1]


class ImageRetrieval:
    """
    Online image retrieval database backed by DINOv2 CLS-token features.

    Usage
    -----
    retrieval = ImageRetrieval()
    retrieval.load_model()

    for kf in keyframes:
        candidate = retrieval.add_and_query(kf.image_rgb, kf.stamp)
        if candidate:
            # process loop closure ...

    The model is lazy-loaded: call load_model() explicitly before inference,
    or pass load_on_init=True to the constructor.
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_MODEL,
        similarity_threshold: float = 0.85,
        min_time_gap: float = 5.0,
        device: str | None = None,
        load_on_init: bool = False,
    ) -> None:
        self._checkpoint = checkpoint
        self._threshold = similarity_threshold
        self._min_time_gap = min_time_gap
        self._device = device
        self._model = None
        self._processor = None

        # Database
        self._embeddings: list[np.ndarray] = []   # list of (D,) float32
        self._stamps: list[float] = []

        if load_on_init:
            self.load_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download / load the DINOv2 model from HuggingFace."""
        if not _DINOV2_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; cannot load DINOv2.")

        from transformers import AutoImageProcessor, AutoModel  # type: ignore
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self._processor = AutoImageProcessor.from_pretrained(self._checkpoint)
        self._model = AutoModel.from_pretrained(self._checkpoint)
        self._model.eval()
        self._model = self._model.to(device)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def add_and_query(
        self,
        image_rgb: np.ndarray,
        stamp: float,
    ) -> Optional[LoopCandidate]:
        """
        Embed the image, store it, and search for a loop candidate.

        Parameters
        ----------
        image_rgb : HxWx3 uint8
        stamp     : frame timestamp (seconds)

        Returns
        -------
        LoopCandidate if a loop is detected, else None.
        """
        embedding = self._embed(image_rgb)
        query_idx = len(self._embeddings)

        candidate = None
        if query_idx > 0:
            candidate = self._search(embedding, stamp, query_idx)

        self._embeddings.append(embedding)
        self._stamps.append(stamp)
        return candidate

    def reset(self) -> None:
        self._embeddings.clear()
        self._stamps.clear()

    @property
    def size(self) -> int:
        return len(self._embeddings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return normalised DINOv2 CLS-token embedding (D,) float32."""
        from PIL import Image as PILImage  # type: ignore

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pil = PILImage.fromarray(image_rgb)
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            # CLS token: (1, D)
            cls = outputs.last_hidden_state[:, 0, :]
            cls = F.normalize(cls, dim=-1)

        return cls[0].float().cpu().numpy()

    def _search(
        self,
        query: np.ndarray,
        query_stamp: float,
        query_idx: int,
    ) -> Optional[LoopCandidate]:
        """Search stored embeddings for a loop candidate."""
        if not self._embeddings:
            return None

        db = np.stack(self._embeddings, axis=0)       # (N, D)
        sims = db @ query                              # (N,) cosine similarity

        # Mask frames that are too recent
        time_diffs = query_stamp - np.array(self._stamps)
        valid = time_diffs >= self._min_time_gap
        if not np.any(valid):
            return None

        sims[~valid] = -1.0
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self._threshold:
            return None

        return LoopCandidate(
            query_idx=query_idx,
            match_idx=best_idx,
            query_stamp=query_stamp,
            match_stamp=self._stamps[best_idx],
            similarity=best_sim,
        )
