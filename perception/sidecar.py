"""Perception inference sidecar — runs in the SCRATCH venv (py3.11 + torch).

A long-lived process that loads FastSAM (and Depth-Anything lazily) ONCE and
answers newline-delimited JSON requests on stdin, replying on stdout. This is
the honest cross-venv seam: the robot venv (py3.14, no torch) renders MuJoCo
frames + supplies FK camera poses and calls this for the pixel/metric-depth
inference it cannot run in-process.

Protocol (one JSON object per line):
  {"op":"ping"}
      -> {"ok":true,"pong":true}
  {"op":"detect","image":<png path>,"name":<str>,"view":"side"|"wrist",
   "cam":[f,cx,cy,[cam_pos],[[R_cw]]],"z_plane":<float>}
      -> {"ok":true,"uv":[u,v],"area":int,"fill":float,"votes":int}  | {"ok":false}
  {"op":"depth","image":<png path>,"uv":[u,v],"anchors":[[u,v,depth_m],...]}
      -> {"ok":true,"depth_m":float,"n_anchors":int}                 | {"ok":false}
  {"op":"segment","image":<png path>,"prompt":<str>,"score_thr":<float?>}
      -> {"ok":true,"instances":[{uv,bbox,mask_area_px,score},...],"stub":bool}
  {"op":"metric_depth","image":<png path>}
      -> {"ok":true,"npy":<path>,"shape":[H,W],"backend":str,"is_stub":bool}
  {"op":"quit"} -> exits

Model weights and CLIP cache are resolved relative to this file's scratch dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_SCRATCH = _HERE.parent / "scratch"
sys.path.insert(0, str(_HERE.parent))          # import perception.detector

from perception.detector import FastSAMDetector  # noqa: E402
from perception.sam3_detector import SAM3Detector  # noqa: E402

_FASTSAM_WEIGHTS = str(_SCRATCH / "FastSAM-s.pt")
_SAM3_WEIGHTS = str(_SCRATCH / "sam3.pt")
_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_DEPTH_TMP = _SCRATCH / "depth_tmp"


class Sidecar:
    def __init__(self) -> None:
        self._det: FastSAMDetector | None = None
        self._depth = None
        self._sam3: SAM3Detector | None = None
        self._depth_backend = None

    # ── lazy model loaders ────────────────────────────────────────────────────
    @property
    def det(self) -> FastSAMDetector:
        if self._det is None:
            self._det = FastSAMDetector(_FASTSAM_WEIGHTS)
        return self._det

    @property
    def sam3(self) -> SAM3Detector:
        """SAM 3 open-vocab segmenter (stub-fallback if no sam3.pt / no SAM3 build)."""
        if self._sam3 is None:
            self._sam3 = SAM3Detector(_SAM3_WEIGHTS)
        return self._sam3

    @property
    def depth_backend(self):
        """Pluggable metric-depth backend (env PERCEPT_DEPTH_MODEL; default stub)."""
        if self._depth_backend is None:
            from perception import depth_backends
            self._depth_backend = depth_backends.get_depth_backend()
        return self._depth_backend

    def _depth_pipe(self):
        if self._depth is None:
            from transformers import pipeline
            self._depth = pipeline("depth-estimation", model=_DEPTH_MODEL,
                                   device="cpu")
        return self._depth

    # ── ops ───────────────────────────────────────────────────────────────────
    def detect(self, req: dict) -> dict:
        img = np.array(Image.open(req["image"]).convert("RGB"))
        f, cx, cy, cam_pos, R_cw = req["cam"]
        cam = (f, cx, cy, np.asarray(cam_pos, float), np.asarray(R_cw, float))
        near_xy = req.get("near_xy")
        if near_xy is not None:
            near_xy = (float(near_xy[0]), float(near_xy[1]))
        target_rgb = req.get("target_rgb")
        if target_rgb is not None:
            target_rgb = tuple(float(c) for c in target_rgb)
        # Gate overrides: the close-range wrist view needs a much larger area envelope
        # than the side view (see FastSAMDetector.WRIST_GATES).
        gates = {k: req[k] for k in ("area_max_frac", "area_min_px", "min_fill",
                                     "in_workspace") if k in req}
        debug = bool(req.get("debug"))
        r = self.det.detect(img, cam, z_plane=req.get("z_plane", 0.015),
                            near_xy=near_xy, target_rgb=target_rgb,
                            debug=debug, **gates)
        if r is None or r.get("uv") is None:
            out = {"ok": False}
            if debug and r is not None:
                out["candidates"] = r.get("candidates", [])
                out["rejected"] = r.get("rejected", [])
            return out
        out = {"ok": True, "uv": r["uv"], "area": r["area"],
               "fill": r["fill"], "votes": r.get("votes", 1),
               "bbox": r.get("bbox"),
               "xy_plane": r.get("xy_plane"), "xy_floor": r.get("xy_floor"),
               "color_dist": r.get("color_dist")}
        if debug:
            out["candidates"] = r.get("candidates", [])
            out["rejected"] = r.get("rejected", [])
        return out

    def depth(self, req: dict) -> dict:
        """Calibrated metric depth at one pixel ("uv") or many ("uvs").

        The many-point form exists so the coarse stage can price EVERY candidate mask
        without re-running the model per candidate: one forward pass, one affine fit, then
        a cheap lookup per pixel. That is what makes a depth-based reachability gate
        affordable as a replacement for the table-plane gate.
        """
        pil = Image.open(req["image"]).convert("RGB")
        pred, W, H = self._metric_depth_map(pil)
        anchors = req["anchors"]
        if "uvs" in req:
            depths = [self._calibrate_depth(pred, W, H, uv, anchors) for uv in req["uvs"]]
            return {"ok": any(d is not None for d in depths), "depths": depths,
                    "n_anchors": int(len(anchors))}
        depth_m = self._calibrate_depth(pred, W, H, req["uv"], anchors)
        if depth_m is None:
            return {"ok": False}
        return {"ok": True, "depth_m": depth_m, "n_anchors": int(len(anchors))}

    def _metric_depth_map(self, pil_img):
        """Depth-Anything relative-depth map, resized to image resolution."""
        out = self._depth_pipe()(pil_img)
        pred = np.array(out["predicted_depth"], dtype=float)
        W, H = pil_img.size
        if pred.shape != (H, W):
            pred = np.array(Image.fromarray(pred).resize((W, H)))
        return pred, W, H

    def _calibrate_depth(self, pred, W, H, uv, anchors):
        """Affine-calibrate the up-to-affine disparity against plane anchors and
        read out the metric forward-axis depth at pixel uv. Returns float | None."""
        anchors = np.asarray(anchors, float)
        if len(anchors) < 3:
            return None
        au = np.clip(anchors[:, 0].astype(int), 0, W - 1)
        av = np.clip(anchors[:, 1].astype(int), 0, H - 1)
        ad = anchors[:, 2]
        p_anchor = pred[av, au]
        # Depth-Anything is disparity up to affine: 1/depth = a*pred + b.
        A = np.column_stack([p_anchor, np.ones(len(p_anchor))])
        (a, b), *_ = np.linalg.lstsq(A, 1.0 / ad, rcond=None)
        pv = pred[int(round(uv[1])), int(round(uv[0]))]
        disp = a * pv + b
        if disp <= 1e-6:
            return None
        return float(1.0 / disp)

    def locate(self, req: dict) -> dict:
        """SAM detect (PRIMARY: colour-free centroid -> support-plane xy) plus an
        OPTIONAL monocular metric-depth read.

        Detection success is the `ok` criterion; depth is advisory. The accurate
        tabletop estimate is the plane-intersection `xy_plane` (~1-2 mm, matches
        HSV), NOT the depth back-projection — monocular depth at side-cam standoff
        is cm-scale and cannot resolve a 30 mm object's height (percept EXP3). So
        `depth_m` may be None (anchors/calibration failed) and the caller still
        gets a valid plane xy. depth_m, when present, is an off-plane z-hint only.
        """
        from perception import camera_math as CM
        pil = Image.open(req["image"]).convert("RGB")
        img = np.array(pil)
        f, cx, cy, cam_pos, R_cw = req["cam"]
        cam_pos = np.asarray(cam_pos, float)
        R_cw = np.asarray(R_cw, float)
        cam = (f, cx, cy, cam_pos, R_cw)
        det = self.det.detect(img, cam, z_plane=req.get("z_plane", 0.015),
                              near_xy=req.get("near_xy"))
        if det is None:
            return {"ok": False, "stage": "detect"}
        u, v = det["uv"]
        bbox = det.get("bbox", [u - 20, v - 20, u + 20, v + 20])
        H, W = img.shape[:2]
        resp = {"ok": True, "uv": [u, v], "bbox": bbox,
                "xy_plane": det.get("xy_plane"), "votes": det.get("votes", 1),
                "depth_m": None, "n_anchors": 0}
        # Optional metric depth — off-plane z-hint only; never sets xy.
        anchors = CM.table_anchor_ring(bbox, f, cx, cy, cam_pos, R_cw, W, H,
                                       z_plane=req.get("anchor_z", 0.0))
        if len(anchors) >= 3:
            pred, W2, H2 = self._metric_depth_map(pil)
            depth_m = self._calibrate_depth(pred, W2, H2, (u, v), anchors)
            if depth_m is not None:
                resp["depth_m"] = depth_m
                resp["n_anchors"] = len(anchors)
        return resp

    def segment(self, req: dict) -> dict:
        """SAM 3 open-vocab text-prompt segmentation -> scored instances."""
        img = np.array(Image.open(req["image"]).convert("RGB"))
        insts = self.sam3.segment(img, req.get("prompt", "object"),
                                  score_thr=req.get("score_thr"))
        return {"ok": True, "instances": insts,
                "stub": bool(self.sam3.is_stub), "backend": "sam3"}

    def metric_depth(self, req: dict) -> dict:
        """Monocular metric-depth map (forward-axis metres) for the image. The map
        is written to .npy and the path returned — the caller loads the array, so
        the heavy HxW payload never crosses the JSON stdio pipe."""
        img = np.array(Image.open(req["image"]).convert("RGB"))
        depth = np.asarray(self.depth_backend.infer_depth(img), np.float32)
        _DEPTH_TMP.mkdir(parents=True, exist_ok=True)
        out = _DEPTH_TMP / (Path(req["image"]).stem + "_depth.npy")
        np.save(out, depth)
        return {"ok": True, "npy": str(out),
                "shape": [int(depth.shape[0]), int(depth.shape[1])],
                "backend": self.depth_backend.name,
                "is_stub": bool(getattr(self.depth_backend, "is_stub", False))}

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "pong": True}
        if op == "detect":
            return self.detect(req)
        if op == "depth":
            return self.depth(req)
        if op == "locate":
            return self.locate(req)
        if op == "segment":
            return self.segment(req)
        if op == "metric_depth":
            return self.metric_depth(req)
        return {"ok": False, "error": f"unknown op {op!r}"}


def main() -> None:
    sc = Sidecar()
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
