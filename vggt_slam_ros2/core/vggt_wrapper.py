"""
Thin wrapper around the VGGT model.
Handles model loading, image preprocessing, and inference.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage


_VGGT_AVAILABLE = False
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    _VGGT_AVAILABLE = True
except ImportError:
    pass

# Default HuggingFace checkpoint URL
_DEFAULT_CHECKPOINT = "facebook/VGGT-1B"


class VGGTWrapper:
    """Loads VGGT and exposes a single `infer` method."""

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        device: str | None = None,
        use_bf16: bool = True,
    ) -> None:
        if not _VGGT_AVAILABLE:
            raise RuntimeError(
                "vggt package is not installed. "
                "Install it from https://github.com/facebookresearch/vggt"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if (
                use_bf16 and torch.cuda.get_device_capability()[0] >= 8
            ) else torch.float16
        else:
            self.dtype = torch.float32

        self.model = VGGT.from_pretrained(checkpoint)
        self.model.eval()
        # Keep model in fp32 on the target device. autocast (below) dynamically
        # casts eligible ops to self.dtype while keeping stable ops (LayerNorm,
        # Softmax) in fp32. Casting the model to bf16 before calling it conflicts
        # with VGGT's internal autocast(enabled=False) blocks that expect fp32 weights.
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def infer(self, images_rgb: list[np.ndarray]) -> dict:
        """
        Run VGGT on a list of HxWx3 uint8 RGB images.

        Returns a dict with:
          - extrinsics:   (S, 3, 4) float32 — cam-from-world
          - intrinsics:   (S, 3, 3) float32
          - world_points: (S, H, W, 3) float32
          - world_points_conf: (S, H, W) float32
          - depth:        (S, H, W) float32
          - depth_conf:   (S, H, W) float32
        """
        tensor = self._preprocess(images_rgb)          # (S, 3, H, W) float32
        tensor = tensor.to(self.device)

        # Model stays in fp32; autocast promotes eligible ops to self.dtype (bf16/fp16)
        # for speed while keeping stable ops (LayerNorm, etc.) in fp32 — consistent
        # with VGGT's internal autocast(enabled=False) blocks which expect fp32 weights.
        device_type = self.device.split(':')[0]  # e.g. 'cuda' from 'cuda:0'
        enabled = (self.dtype != torch.float32)
        with torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=enabled):
            raw = self.model(tensor)

        return self._postprocess(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, images_rgb: list[np.ndarray]) -> torch.Tensor:
        """Convert list of HxWx3 uint8 arrays to a (S,3,H,W) float tensor in [0,1]."""
        pil_images = [PILImage.fromarray(img) for img in images_rgb]
        # Use VGGT's own load_and_preprocess_images if available,
        # otherwise fall back to a simple resize + normalise.
        try:
            tensor = load_and_preprocess_images(pil_images)  # (S, 3, H, W) float
        except (TypeError, AttributeError):
            # load_and_preprocess_images only accepts file paths, not PIL images;
            # fall back to manual preprocessing.
            tensors = []
            target_size = (518, 518)
            for img in pil_images:
                img_resized = img.resize(target_size, PILImage.BILINEAR)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            tensor = torch.stack(tensors)
        return tensor

    def _postprocess(self, raw: dict) -> dict:
        """Extract, convert to float32 numpy, and return structured dict."""
        # Camera poses
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            raw["pose_enc"], raw["images"].shape[-2:]
        )
        # (B=1, S, 3, 4) → (S, 3, 4)
        extrinsics = extrinsics[0].float().cpu().numpy()
        intrinsics = intrinsics[0].float().cpu().numpy()

        world_points = raw["world_points"][0].float().cpu().numpy()       # (S, H, W, 3)
        world_points_conf = raw["world_points_conf"][0].float().cpu().numpy()  # (S, H, W)
        depth = raw["depth"][0, :, :, :, 0].float().cpu().numpy()         # (S, H, W)
        depth_conf = raw["depth_conf"][0].float().cpu().numpy()           # (S, H, W)

        return {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "world_points": world_points,
            "world_points_conf": world_points_conf,
            "depth": depth,
            "depth_conf": depth_conf,
        }
