"""Pluggable monocular metric-depth backend for the perceptual pipeline.

The metric-depth MODEL is NOT chosen yet — a Colab benchmark (see
``experiments/notebooks/metric_depth_benchmark.ipynb``) will pick one of Depth
Pro / UniDepthV2 / Metric3D-v2 on measured localisation error. Until then the
pipeline runs end-to-end against a STUB so the plumbing (SAM3 -> depth ->
``camera_math.point_at_depth`` -> workspace gate) is exercisable without the real
weights.

Contract: ``infer_depth(image_rgb)`` returns an ``(H, W)`` float32 map of METRIC
depth in metres measured along the camera OPTICAL / FORWARD axis — the SAME
convention ``camera_math.point_at_depth`` consumes (depth along -Z of the MuJoCo
camera). A real backend must convert its native output (often Euclidean range or
up-to-affine disparity) to this forward-axis metric depth before returning.

Selection is by env var ``PERCEPT_DEPTH_MODEL`` (default ``constant_stub``). The
stub returns a flat plane — it MEASURES NOTHING — so ``is_stub_backend`` flags it
and an honest localisation eval must refuse it.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

DEFAULT_BACKEND = "constant_stub"
_ENV_VAR = "PERCEPT_DEPTH_MODEL"
# A plausible side-cam forward-axis standoff (m) so the stub yields in-envelope
# points for smoke tests. NOT a measurement — the real model replaces this.
_STUB_DEPTH_M = 1.15


@runtime_checkable
class MetricDepthBackend(Protocol):
    """Structural type for a monocular metric-depth model."""

    name: str

    def infer_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """(H, W) float32 forward-axis metric depth in metres for ``image_rgb``."""
        ...


class ConstantDepthBackend:
    """STUB backend: a uniform metric-depth map.

    Exercises the SAM3 -> depth -> deproject plumbing without a real model. It
    measures NOTHING (every pixel gets the same depth), so ``is_stub`` is True and
    it must not be used for an honest localisation benchmark.
    """

    is_stub = True

    def __init__(self, value_m: float = _STUB_DEPTH_M):
        self.value_m = float(value_m)
        self.name = f"constant_stub({self.value_m:.3f}m)"

    def infer_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(image_rgb)
        h, w = arr.shape[:2]
        return np.full((h, w), self.value_m, dtype=np.float32)


# Registry: name -> zero-arg factory. The chosen real model registers here once
# the benchmark selects it (register_depth_backend(name, factory)).
_REGISTRY: dict[str, Callable[[], "MetricDepthBackend"]] = {
    "constant_stub": ConstantDepthBackend,
}


def register_depth_backend(name: str,
                           factory: Callable[[], "MetricDepthBackend"]) -> None:
    """Register a metric-depth backend factory under ``name``."""
    _REGISTRY[name] = factory


def get_depth_backend(name: Optional[str] = None) -> "MetricDepthBackend":
    """Resolve a backend by name, or by env ``PERCEPT_DEPTH_MODEL`` (default stub).

    Unknown names raise so a typo never silently degrades to the stub in an eval.
    """
    if name is None:
        name = os.environ.get(_ENV_VAR, DEFAULT_BACKEND)
    factory = _REGISTRY.get(name)
    if factory is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"unknown metric-depth backend {name!r}; known: {known}. "
            f"Set {_ENV_VAR} or register the chosen model first.")
    return factory()


def is_stub_backend(backend: "MetricDepthBackend") -> bool:
    """True if ``backend`` is a placeholder that measures nothing (honest-eval gate)."""
    return bool(getattr(backend, "is_stub", False))
