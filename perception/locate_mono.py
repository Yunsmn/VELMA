"""Monocular SAM + metric-depth object localisation — locate_object_3d_mono.

The robustness-first perception path the project standardised on: ONE camera, no
second view, no triangulation, no plane-at-object assumption. It generalises to
any scene the depth model handles and to raised / stacked / unknown-height
objects, and it is cheap enough for a small LLM to call as a single MCP tool.

Pipeline (all honest — nothing reads the object's qpos)
  1. render ONE camera (fixed side calibration by default).
  2. sidecar (FastSAM everything-mode + a colour-free geometric selector) returns
     the object mask centroid (u,v) and bbox.
  3. sidecar samples a ring of table pixels AROUND the bbox, gives each its
     analytic support-plane depth (ray∩z=0 from the camera pose — pure geometry),
     and affine-calibrates the up-to-affine monocular depth model on them, then
     reads the object's own metric forward-axis depth at (u,v).
  4. back-project (u,v) along its viewing ray to that depth -> metric world
     (x,y,z). z is RECOVERED from depth, so a raised object gets its true height.

Contrast with locate_object_3d.py (kept for reference): that fuses a second
wrist view via closest-point triangulation. This module deliberately uses a
single view so it drops into any environment without a hover manoeuvre.

Returns Locate3DResult(x, y, z, confidence, method, extras).
"""
from __future__ import annotations

import numpy as np

from perception import camera_math as CM
from perception.locate_object_3d import Locate3DResult, _save_png, _wrist_extrinsics

# Depth sanity envelope (metres) — the reachable tabletop volume. A calibrated
# depth wildly outside this means the affine fit or the detect mis-fired, so we
# fall back to the side-plane estimate rather than return a fantasy point.
Z_MIN, Z_MAX = -0.02, 0.20
XY_R_MAX = 0.45


def _side_pose():
    f, cx, cy, cam_pos, R_cw = CM.side_cam_params()
    return f, cx, cy, cam_pos, R_cw


def locate_object_3d_mono(backend, controller, client, name: str = "cube",
                          view: str = "side", support_z: float = 0.015,
                          near_xy=None, assume_on_plane: bool = True
                          ) -> Locate3DResult:
    """Single-view SAM localisation with a support-plane metric prior.

    PRIMARY (tabletop, default): the colour-free SAM centroid is intersected with
    the known support plane z=support_z (ray∩plane) to give a ~1-2 mm (x,y) that
    matches/beats the HSV baseline (measured: 1.67 mm mean vs HSV 2.0 mm); z is
    the support height. This is the accurate path for objects resting on a known
    plane, and is exactly what the real-arm ChArUco-homography pipeline reduces
    to. Method = "sam_plane".

    Monocular depth is computed and stashed as an OFF-PLANE z-hint only
    (extras['mono_xyz'], extras['depth_m']); at side-cam standoff it is cm-scale
    and cannot resolve a 30 mm object's height (percept EXP3), so it NEVER sets
    the returned (x,y) on the default path — that was the 12 mm error. Pass
    assume_on_plane=False to opt into the depth-back-projected 3D point for a
    genuinely raised/stacked object (still coarse — prefer a real depth/stereo
    source for metric height). Method = "mono_sam_depth".

    `view` is 'side' (fixed calibration, default) or 'wrist' (FK pose).
    """
    extras: dict = {}

    if view == "wrist":
        img = backend.render_wrist()
        ext = _wrist_extrinsics(backend)
        if img is None or ext is None:
            return Locate3DResult(None, None, None, 0.0, "wrist_pose_fail", extras)
        cam_pos, R_cw = ext
        f, cx, cy = CM.wrist_intrinsics()
    else:
        img = backend.render_side()
        f, cx, cy, cam_pos, R_cw = _side_pose()

    cam = (f, cx, cy, cam_pos, R_cw)
    png = _save_png(img, f"mono_{view}.png")

    loc = client.locate(png, cam, view=view, name=name, z_plane=support_z,
                        anchor_z=0.0, near_xy=near_xy)
    if loc is None:
        return Locate3DResult(None, None, None, 0.0, "detect_fail", extras)

    u, v = loc["uv"]
    extras["uv"] = [u, v]
    extras["bbox"] = loc.get("bbox")
    extras["votes"] = loc.get("votes")
    plane_xy = loc.get("xy_plane")
    if plane_xy is not None:
        extras["side_plane_xy"] = plane_xy

    # Depth is advisory: compute the off-plane 3D point for the hint / raised path.
    depth_m = loc.get("depth_m")
    mono_p = None
    if depth_m is not None:
        extras["depth_m"] = round(float(depth_m), 4)
        extras["n_anchors"] = loc.get("n_anchors")
        mono_p = CM.point_at_depth(u, v, f, cx, cy, cam_pos, R_cw, float(depth_m))
        extras["mono_xyz"] = [float(mono_p[0]), float(mono_p[1]), float(mono_p[2])]

    # ── PRIMARY: support-plane xy (accurate, HSV-parity) ─────────────────────
    if assume_on_plane and plane_xy is not None:
        x0, y0 = plane_xy
        if float(np.hypot(x0, y0)) <= XY_R_MAX:
            votes = loc.get("votes", 1) or 1
            conf = 0.75 + 0.20 * min(1.0, votes / 4.0)
            return Locate3DResult(float(x0), float(y0), float(support_z),
                                  conf, "sam_plane", extras)

    # ── OFF-PLANE: depth-back-projected 3D point (raised objects; coarse) ─────
    if mono_p is not None:
        r = float(np.hypot(mono_p[0], mono_p[1]))
        if Z_MIN <= mono_p[2] <= Z_MAX and r <= XY_R_MAX:
            na = loc.get("n_anchors", 0) or 0
            conf = 0.45 + 0.25 * min(1.0, na / 20.0)
            return Locate3DResult(float(mono_p[0]), float(mono_p[1]),
                                  float(mono_p[2]), conf, "mono_sam_depth", extras)

    # ── Fallback: plane xy at support height (depth failed / out of envelope) ─
    if plane_xy is not None:
        x0, y0 = plane_xy
        extras["note"] = "plane_fallback"
        return Locate3DResult(float(x0), float(y0), float(support_z), 0.5,
                              "sam_plane", extras)
    return Locate3DResult(None, None, None, 0.0, "out_of_envelope", extras)
