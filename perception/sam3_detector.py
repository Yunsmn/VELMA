"""SAM 3 open-vocabulary, text-prompted instance segmentation (sidecar-side).

Runs in the SCRATCH venv (torch + ultralytics), the same process as
``FastSAMDetector``. A short noun-phrase prompt (e.g. ``"the red cube"``,
``"the container"``) returns a scored instance mask per match — open-vocabulary,
so there is NO colour threshold and NO hard-coded object class. This is the
honest replacement for FastSAM everything-mode + a geometric selector; SAM 3
folds in the Grounding-DINO-style text grounding a separate stage used to need.

Weights + version: uses the ultralytics SAM interface (``from ultralytics import
SAM; SAM("sam3.pt")``) for consistency with ``FastSAMDetector``'s
``from ultralytics import FastSAM``. It requires an ultralytics that ships
``models/sam/build_sam3.py`` AND a local ``sam3.pt``. If SAM 3 is unsupported OR
the weights are absent, this falls back to a CLEARLY-MARKED stub segmenter
(``stub=True``, one centre-of-frame instance) so the pipeline scaffold still runs
end-to-end. It deliberately does NOT force a weight download or an ultralytics
upgrade — upgrading could break the pinned FastSAM path the benchmarks use.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Optional

import numpy as np

_DEFAULT_SCORE_THR = 0.4


def ultralytics_supports_sam3() -> bool:
    """True if the installed ultralytics ships the SAM 3 build module."""
    try:
        importlib.import_module("ultralytics.models.sam.build_sam3")
        return True
    except Exception:
        return False


class SAM3Detector:
    """Text-prompted SAM 3 segmenter with a clearly-marked stub fallback."""

    def __init__(self, weights: str = "sam3.pt", imgsz: int = 1024,
                 score_thr: float = _DEFAULT_SCORE_THR):
        self.weights = weights
        self.imgsz = imgsz
        self.default_score_thr = float(score_thr)
        self.model = None
        self.is_stub = True
        self.reason = ""
        # Check weights FIRST: the common "no sam3.pt" case then stubs without the
        # heavy build_sam3 import (which pulls torch) just to decide to stub.
        if not os.path.exists(weights):
            self.reason = f"weights not found locally: {weights}"
        elif not ultralytics_supports_sam3():
            self.reason = "installed ultralytics has no SAM3 build (build_sam3)"
        else:
            try:
                from ultralytics import SAM
                self.model = SAM(weights)
                self.is_stub = False
            except Exception as exc:  # never force a download / upgrade
                self.reason = f"SAM3 load failed: {type(exc).__name__}: {exc}"
        if self.is_stub:
            print(f"[sam3] STUB segmenter active ({self.reason})", file=sys.stderr)

    def segment(self, img_rgb: np.ndarray, prompt: str,
                score_thr: Optional[float] = None) -> list[dict]:
        """Text noun-phrase -> scored instances:
        [{"uv":[u,v], "bbox":[x0,y0,x1,y1], "mask_area_px":int, "score":float}, ...]
        sorted by descending score. Colour/class-free."""
        thr = self.default_score_thr if score_thr is None else float(score_thr)
        if self.is_stub:
            return self._stub_segment(img_rgb, prompt)
        return self._real_segment(img_rgb, prompt, thr)

    def _stub_segment(self, img_rgb: np.ndarray, prompt: str) -> list[dict]:
        """Placeholder: one centre-of-frame instance flagged stub=True. Not a real
        detection — just enough structure to exercise the depth/deproject plumbing."""
        arr = np.asarray(img_rgb)
        h, w = arr.shape[:2]
        u, v = w / 2.0, h / 2.0
        half = min(h, w) * 0.06
        bbox = [u - half, v - half, u + half, v + half]
        return [{"uv": [float(u), float(v)],
                 "bbox": [float(c) for c in bbox],
                 "mask_area_px": int((2 * half) ** 2),
                 "score": 0.5, "stub": True, "prompt": str(prompt)}]

    def _real_segment(self, img_rgb: np.ndarray, prompt: str,
                      thr: float) -> list[dict]:
        # ultralytics forwards **kwargs from SAM.predict to the SAM3 semantic
        # predictor, which takes a text noun-phrase list. The exact kwarg name has
        # drifted across ultralytics revisions, so try the documented one then
        # aliases. (Untested here: sam3.pt weights are not present in this repo.)
        res = None
        last_err: Optional[Exception] = None
        for kw in ("texts", "text", "prompt"):
            try:
                res = self.model.predict(img_rgb, device="cpu", imgsz=self.imgsz,
                                         retina_masks=True, verbose=False,
                                         **{kw: [str(prompt)]})
                break
            except TypeError as exc:
                last_err = exc
        if res is None:
            raise RuntimeError("SAM3 text-prompt call rejected texts/text/prompt "
                               f"kwargs: {last_err}")
        return self._parse(res[0], thr)

    @staticmethod
    def _parse(r, thr: float) -> list[dict]:
        out: list[dict] = []
        masks = getattr(r, "masks", None)
        if masks is None:
            return out
        data = masks.data.cpu().numpy()
        boxes = getattr(r, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.cpu().numpy()
        else:
            confs = np.ones(len(data), dtype=float)
        for i, m in enumerate(data):
            score = float(confs[i]) if i < len(confs) else 1.0
            if score < thr:
                continue
            mm = m > 0.5
            ys, xs = np.where(mm)
            if xs.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            out.append({"uv": [float(xs.mean()), float(ys.mean())],
                        "bbox": [x0, y0, x1, y1],
                        "mask_area_px": int(mm.sum()),
                        "score": score})
        out.sort(key=lambda d: d["score"], reverse=True)
        return out
