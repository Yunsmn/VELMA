"""Falcon-Perception open-vocabulary grounding + segmentation (GPU sidecar-side).

Runs in the percept_gpu_venv (CUDA torch >=2.5, needed for FlexAttention). This is
the TRUE open-vocab replacement for the old colour-seeded FastSAM coarse stage: the
free-text query goes straight to the model as the grounding signal — no colour
threshold, no class list, no object truth. `tiiuae/Falcon-Perception` (0.6B,
Apache-2.0, non-gated) takes an image + a natural-language phrase and returns zero,
one or many matching instances, each with a normalized center/size and a
full-resolution COCO-RLE mask (arXiv:2603.27365; model card:
https://huggingface.co/tiiuae/Falcon-Perception).

Falcon does not emit a confidence score (unlike SAM 3's per-instance `score`), so
candidates are ranked by mask area here — the caller (find_object_depth) applies
its own reachability/size gates on top, which is where the real disambiguation
happens for this pipeline.

DTYPE GOTCHA (measured, not guessed): float16 loads fine and runs without error,
but silently returns ZERO instances for every query, on every image tested —
including the README's own COCO "cat" example and a trivial solid-colour
synthetic shape. bfloat16 (same 2-byte footprint, so no memory cost) fixes it
completely and localises correctly (COCO cats: 2/2 found; a synthetic red square
and blue circle: normalized xy/hw matched the drawn geometry to ~1%). This looks
like fp16 range/overflow in an attention or RoPE component the model was not
validated under — use bfloat16 on this hardware (Ampere, RTX A500, compute
capability 8.6, supports bf16 natively).
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np

_MODEL_ID = "tiiuae/falcon-perception"
_DEFAULT_MAX_NEW_TOKENS = 512
_DEFAULT_MIN_DIM = 256
_DEFAULT_MAX_DIM = 448   # 1024 OOMs the 4GB RTX A500 on the forward pass (activations
                         # >2.2GB); measured fit: 640=2980MB, 448=2390MB, 320=2181MB —
                         # 448 keeps ~1.3GB headroom and detection is pixel-identical.


class FalconGroundingDetector:
    """Loads Falcon-Perception once; grounds a free-text query to mask instances."""

    def __init__(self, model_id: str = _MODEL_ID, device: str = "cuda:0",
                 dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM

        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.model_id = model_id
        self.is_stub = False
        print(f"[falcon] loading {model_id} on {device} ({dtype}) ...",
              file=sys.stderr)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map={"": device},
        )
        self.model.eval()
        print("[falcon] loaded", file=sys.stderr)

    def ground(self, img_rgb: np.ndarray, query: str,
               max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
               min_dimension: int = _DEFAULT_MIN_DIM,
               max_dimension: int = _DEFAULT_MAX_DIM,
               compile: bool = False) -> list[dict]:
        """Free-text `query` -> mask instances, sorted by mask area (desc).

        Each instance: {"uv":[u,v] pixel centroid (from the DECODED mask, not the
        model's normalized xy head, for consistency with the FastSAM candidate
        format the rest of the pipeline expects), "bbox":[x0,y0,x1,y1] pixel,
        "area":int mask px, "fill":float area/bbox_area, "xy_norm":[x,y],
        "hw_norm":[h,w]} (the last two are Falcon's own normalized outputs, kept
        for debugging/visualisation).
        """
        from PIL import Image
        import torch
        from pycocotools import mask as mask_utils

        image = Image.fromarray(np.asarray(img_rgb)).convert("RGB")
        with torch.no_grad():
            preds = self.model.generate(
                image, str(query),
                max_new_tokens=max_new_tokens,
                min_dimension=min_dimension, max_dimension=max_dimension,
                compile=compile,
            )[0]

        out: list[dict] = []
        for p in preds:
            rle = p["mask_rle"]
            counts = rle["counts"]
            if isinstance(counts, str):
                counts = counts.encode("utf-8")
            mask = mask_utils.decode({"size": rle["size"], "counts": counts}).astype(bool)
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            area = int(mask.sum())
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            out.append({
                "uv": [float(xs.mean()), float(ys.mean())],
                "bbox": [x0, y0, x1, y1],
                "area": area,
                "fill": area / max(bbox_area, 1),
                "xy_norm": [float(p["xy"]["x"]), float(p["xy"]["y"])],
                "hw_norm": [float(p["hw"]["h"]), float(p["hw"]["w"])],
            })
        out.sort(key=lambda d: -d["area"])
        return out
