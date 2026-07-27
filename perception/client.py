"""Client for the perception sidecar — runs in the ROBOT venv (py3.14).

Spawns `percept_venv/bin/python -m perception.sidecar` as a subprocess and talks
newline-delimited JSON over stdio. The robot venv renders MuJoCo frames + reads
FK camera poses; this ships them to the torch venv for FastSAM / Depth-Anything
inference and reads back pixel centroids / metric depth. Keeps the models warm
across many calls (one process, one load).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent                                  # so101-Models (project root)
_SCRATCH_PY = _EXP / "percept_venv" / "bin" / "python"
_GPU_SCRATCH_PY = _EXP / "percept_gpu_venv" / "bin" / "python"


class SidecarError(RuntimeError):
    pass


class PerceptionClient:
    def __init__(self, python: Optional[str] = None, ready_timeout: float = 180.0,
                 gpu_python: Optional[str] = None):
        self.python = python or str(_SCRATCH_PY)
        if not Path(self.python).exists():
            raise SidecarError(f"scratch interpreter not found: {self.python}")
        self.proc = subprocess.Popen(
            [self.python, "-m", "perception.sidecar"],
            cwd=str(_EXP),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1,
        )
        # Wait for the ready banner (model import can be slow first time).
        line = self.proc.stdout.readline()
        if not line:
            raise SidecarError("sidecar exited before ready")
        banner = json.loads(line)
        if not banner.get("ready"):
            raise SidecarError(f"unexpected banner: {banner}")

        # GPU sidecar (Falcon-Perception open-vocab grounding) — SEPARATE process/venv
        # because it needs a CUDA build of torch>=2.5 (FlexAttention), while the CPU
        # sidecar above must stay on torch-CPU. Spawned LAZILY on first .ground() call
        # (see _gpu_rpc), not here, so nothing pays the CUDA-model load cost unless the
        # caller actually asks for open-vocab grounding.
        self.gpu_python = gpu_python or str(_GPU_SCRATCH_PY)
        self.gpu_proc: Optional[subprocess.Popen] = None
        self._gpu_ready_timeout = ready_timeout

    def _rpc(self, req: dict) -> dict:
        if self.proc.poll() is not None:
            raise SidecarError("sidecar process is dead")
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise SidecarError("sidecar closed the pipe")
        return json.loads(line)

    def _ensure_gpu_proc(self) -> None:
        if self.gpu_proc is not None and self.gpu_proc.poll() is None:
            return
        if not Path(self.gpu_python).exists():
            raise SidecarError(
                f"GPU scratch interpreter not found: {self.gpu_python} — build "
                "percept_gpu_venv (CUDA torch>=2.5) to use open-vocab grounding")
        self.gpu_proc = subprocess.Popen(
            [self.gpu_python, "-m", "perception.gpu_sidecar"],
            cwd=str(_EXP),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1,
        )
        line = self.gpu_proc.stdout.readline()
        if not line:
            raise SidecarError("GPU sidecar exited before ready")
        banner = json.loads(line)
        if not banner.get("ready"):
            raise SidecarError(f"unexpected GPU sidecar banner: {banner}")

    def _gpu_rpc(self, req: dict) -> dict:
        self._ensure_gpu_proc()
        if self.gpu_proc.poll() is not None:
            raise SidecarError("GPU sidecar process is dead")
        self.gpu_proc.stdin.write(json.dumps(req) + "\n")
        self.gpu_proc.stdin.flush()
        line = self.gpu_proc.stdout.readline()
        if not line:
            raise SidecarError("GPU sidecar closed the pipe")
        return json.loads(line)

    def ping(self) -> bool:
        return bool(self._rpc({"op": "ping"}).get("pong"))

    def detect(self, image: str, cam, view: str, name: str = "object",
               z_plane: float = 0.015, near_xy=None, target_rgb=None,
               gates: Optional[dict] = None, debug: bool = False) -> Optional[dict]:
        """FastSAM object detection -> pixel centroid (+ plane/floor deprojections).

        gates: optional mask-filter overrides (area_max_frac / area_min_px / min_fill /
        in_workspace). The side view uses the module defaults; the close-range wrist
        view must pass FastSAMDetector.WRIST_GATES or every mask is rejected as too big.
        debug: also return the surviving candidates and per-gate rejections (tuning).
        """
        f, cx, cy, cam_pos, R_cw = cam
        req = {
            "op": "detect", "image": str(image), "name": name, "view": view,
            "z_plane": z_plane,
            "cam": [float(f), float(cx), float(cy),
                    np.asarray(cam_pos, float).tolist(),
                    np.asarray(R_cw, float).tolist()],
        }
        if near_xy is not None:
            req["near_xy"] = [float(near_xy[0]), float(near_xy[1])]
        if target_rgb is not None:
            req["target_rgb"] = [float(c) for c in target_rgb]
        if gates:
            req.update(gates)
        if debug:
            req["debug"] = True
        r = self._rpc(req)
        if r.get("ok"):
            return r
        return r if debug else None

    def locate(self, image: str, cam, view: str = "side", name: str = "object",
               z_plane: float = 0.015, anchor_z: float = 0.0,
               near_xy=None) -> Optional[dict]:
        """Monocular SAM + metric-depth: returns {uv, depth_m, bbox, xy_plane,
        votes} or None. depth_m is the object's forward-axis depth in metres;
        the caller back-projects it with `cam` to a 3D world point."""
        f, cx, cy, cam_pos, R_cw = cam
        req = {
            "op": "locate", "image": str(image), "name": name, "view": view,
            "z_plane": z_plane, "anchor_z": anchor_z,
            "cam": [float(f), float(cx), float(cy),
                    np.asarray(cam_pos, float).tolist(),
                    np.asarray(R_cw, float).tolist()],
        }
        if near_xy is not None:
            req["near_xy"] = [float(near_xy[0]), float(near_xy[1])]
        r = self._rpc(req)
        return r if r.get("ok") else None

    def depth(self, image: str, uv, anchors) -> Optional[float]:
        req = {"op": "depth", "image": str(image),
               "uv": [float(uv[0]), float(uv[1])],
               "anchors": np.asarray(anchors, float).tolist()}
        r = self._rpc(req)
        return float(r["depth_m"]) if r.get("ok") else None

    def depth_many(self, image: str, uvs, anchors) -> Optional[list]:
        """Calibrated metric depth at several pixels in ONE model pass.

        Returns a list aligned with `uvs` (entries may be None where calibration failed),
        or None if the whole call failed. Used to price every candidate mask so the coarse
        stage can gate on real 3D reachability instead of a table-plane assumption.
        """
        req = {"op": "depth", "image": str(image),
               "uvs": [[float(u), float(v)] for u, v in uvs],
               "anchors": np.asarray(anchors, float).tolist()}
        r = self._rpc(req)
        if not r.get("ok"):
            return None
        return [None if d is None else float(d) for d in r["depths"]]

    def segment(self, image: str, prompt: str = "object",
                score_thr: Optional[float] = None) -> Optional[dict]:
        """SAM 3 open-vocab text-prompt segmentation. Returns
        {"instances":[{uv,bbox,mask_area_px,score,...}], "stub":bool} or None.
        Colour/class-free — the noun-phrase prompt does the grounding."""
        req = {"op": "segment", "image": str(image), "prompt": str(prompt)}
        if score_thr is not None:
            req["score_thr"] = float(score_thr)
        r = self._rpc(req)
        return r if r.get("ok") else None

    def metric_depth(self, image: str) -> Optional[dict]:
        """Monocular metric-depth map for the image. Returns
        {"depth": (H,W) float32 forward-axis metres, "backend":str, "is_stub":bool,
        "shape":[H,W]} or None. The sidecar writes the map to .npy (path in the
        reply) and we load it here so the heavy array never crosses the pipe."""
        r = self._rpc({"op": "metric_depth", "image": str(image)})
        if not r.get("ok"):
            return None
        r["depth"] = np.load(r["npy"])
        return r

    def ground(self, image: str, query: str,
               max_new_tokens: int = 512) -> Optional[dict]:
        """Falcon-Perception open-vocabulary grounding (GPU sidecar). `query` is a
        free-text phrase (e.g. "the red cube") — it is the ONLY object hint, no
        colour threshold and no class list. Returns {"instances":[{uv,bbox,area,
        fill,xy_norm,hw_norm}, ...], "backend":"falcon_perception"} sorted by mask
        area (Falcon emits no confidence score), or None on failure. Spawns the
        GPU sidecar process on first call (see `_ensure_gpu_proc`)."""
        r = self._gpu_rpc({"op": "ground", "image": str(image), "query": str(query),
                           "max_new_tokens": int(max_new_tokens)})
        return r if r.get("ok") else None

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        try:
            if self.gpu_proc is not None and self.gpu_proc.poll() is None:
                self.gpu_proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.gpu_proc.stdin.flush()
                self.gpu_proc.wait(timeout=10)
        except Exception:
            if self.gpu_proc is not None:
                self.gpu_proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
