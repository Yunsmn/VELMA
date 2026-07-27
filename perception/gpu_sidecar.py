"""GPU perception sidecar — runs in percept_gpu_venv (CUDA torch >= 2.5).

A SECOND long-lived subprocess, parallel to `sidecar.py`'s CPU one. It exists
because Falcon-Perception needs FlexAttention (torch >= 2.5 with a CUDA build);
the existing sidecar's FastSAM / Depth-Anything / SAM 3 stack is CPU-only and
lives in a separate venv (`percept_venv`) that must not be disturbed by a CUDA
torch install. `PerceptionClient` spawns this process lazily — only the first
time `.ground()` is called — so nothing about the existing CPU pipeline changes
for callers that never use open-vocab grounding.

Protocol (one JSON object per line), same wire shape as sidecar.py:
  {"op":"ping"}
      -> {"ok":true,"pong":true,"cuda":bool,"device":str}
  {"op":"ground","image":<png path>,"query":<str>,"max_new_tokens":<int?>}
      -> {"ok":true,"instances":[{"uv":[u,v],"bbox":[x0,y0,x1,y1],
                                   "area":int,"fill":float}], "backend":"falcon_perception"}
       | {"ok":false,"error":str}
  {"op":"quit"} -> exits
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # import perception.falcon_detector

from perception.falcon_detector import FalconGroundingDetector  # noqa: E402


class GPUSidecar:
    def __init__(self) -> None:
        self._falcon: FalconGroundingDetector | None = None

    @property
    def falcon(self) -> FalconGroundingDetector:
        if self._falcon is None:
            self._falcon = FalconGroundingDetector()
        return self._falcon

    def ground(self, req: dict) -> dict:
        img = np.array(Image.open(req["image"]).convert("RGB"))
        insts = self.falcon.ground(
            img, req["query"],
            max_new_tokens=int(req.get("max_new_tokens", 512)),
        )
        return {"ok": True, "instances": insts, "backend": "falcon_perception"}

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            import torch
            cuda = bool(torch.cuda.is_available())
            return {"ok": True, "pong": True, "cuda": cuda,
                    "device": torch.cuda.get_device_name(0) if cuda else "cpu"}
        if op == "ground":
            return self.ground(req)
        return {"ok": False, "error": f"unknown op {op!r}"}


def main() -> None:
    sc = GPUSidecar()
    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
            sys.stdout.flush()
            continue
        if req.get("op") == "quit":
            break
        try:
            resp = sc.handle(req)
        except Exception as exc:  # keep the server alive on any inference error
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
