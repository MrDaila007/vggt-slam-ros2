"""
Automatic GPU memory-aware parameter selection (Stage 4.3).

Queries `torch.cuda.mem_get_info()` at startup and picks `window_size`
and `stride` so that peak GPU memory stays below a configurable budget.

Memory model (empirical approximation for VGGT-1B):
  - VGGT model weights: ~2.5 GB (bfloat16)
  - Activation memory per frame at 518×518: ~0.18 GB
  - Total: weights + window_size * 0.18 GB

The function also prints the chosen parameters so they can be reproduced
in config/params.yaml without the auto-tuning overhead.
"""

from __future__ import annotations

from dataclasses import dataclass


# Empirical constants for VGGT-1B at resolution 518×518 (bfloat16)
_MODEL_WEIGHT_GB = 2.5
_ACTIVATION_PER_FRAME_GB = 0.18
_SAFETY_MARGIN = 0.85       # use at most 85% of available memory


@dataclass
class WindowParams:
    window_size: int
    stride: int
    overlap: int
    estimated_peak_gb: float


def select_window_params(
    memory_budget_gb: float | None = None,
    safety_margin: float = _SAFETY_MARGIN,
    min_window: int = 4,
    max_window: int = 32,
    target_overlap_ratio: float = 0.5,
) -> WindowParams:
    """
    Select window_size and stride given available GPU memory.

    Parameters
    ----------
    memory_budget_gb
        Maximum GPU memory to use (GB). If None, auto-detected from
        `torch.cuda.mem_get_info()`. Falls back to 8 GB if CUDA is
        not available.
    safety_margin
        Fraction of available memory to reserve as headroom.
    min_window
        Minimum acceptable window_size.
    max_window
        Maximum acceptable window_size.
    target_overlap_ratio
        overlap / window_size ratio (stride = window_size * (1 - ratio)).

    Returns
    -------
    WindowParams with the chosen window_size, stride, and estimated peak
    GPU usage.
    """
    if memory_budget_gb is None:
        memory_budget_gb = _detect_free_memory_gb()

    usable_gb = memory_budget_gb * safety_margin
    frames_budget = max(0, usable_gb - _MODEL_WEIGHT_GB) / _ACTIVATION_PER_FRAME_GB
    window_size = int(min(max(min_window, frames_budget), max_window))

    stride = max(1, round(window_size * (1 - target_overlap_ratio)))
    overlap = window_size - stride
    peak_gb = _MODEL_WEIGHT_GB + window_size * _ACTIVATION_PER_FRAME_GB

    params = WindowParams(
        window_size=window_size,
        stride=stride,
        overlap=overlap,
        estimated_peak_gb=peak_gb,
    )
    return params


def _detect_free_memory_gb() -> float:
    """Return free GPU memory in GB, or a fallback value."""
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes / (1024 ** 3)
    except Exception:
        pass
    return 8.0  # fallback: assume 8 GB


def print_params(params: WindowParams) -> None:
    """Print the chosen parameters in a copy-pasteable format."""
    print("=" * 56)
    print("Auto-selected VGGT window parameters")
    print(f"  window_size : {params.window_size}")
    print(f"  stride      : {params.stride}")
    print(f"  overlap     : {params.overlap}")
    print(f"  Est. peak GPU usage : {params.estimated_peak_gb:.1f} GB")
    print()
    print("To fix these values (skip auto-tuning), add to params.yaml:")
    print(f"    window_size:   {params.window_size}")
    print(f"    window_stride: {params.stride}")
    print("=" * 56)
