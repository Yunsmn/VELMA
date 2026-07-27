"""Honest locate_object_3d — metric (x,y,z) of a named object from images only.

Runs in the ROBOT venv (renders MuJoCo, drives the arm, reads FK camera poses).
Model inference (FastSAM / Depth-Anything) is delegated to the scratch-venv
sidecar via PerceptionClient. NOTHING here reads the object's qpos.

Pipeline
  1. render the fixed side camera; sidecar detects the object -> (u_s,v_s).
  2. coarse plane-deproject the side centroid to the support plane -> (x0,y0).
  3. state-neutrally hover the WRIST camera over (x0,y0) (snapshot/restore, from
     wrist_refine); render it; read its FK pose (cam_xpos/cam_xmat); sidecar
     detects the object top-down -> (u_w,v_w).
  4. back-project BOTH centroids to world rays (side pose = fixed calibration,
     wrist pose = FK). Closest-point triangulation gives metric (x,y,z) with NO
     flat-plane assumption -> recovers height, generalises to stacked/unknown-z.
  5. fuse: prefer well-conditioned triangulation; else wrist-plane; else
     side-plane. Optional Depth-Anything z cross-check / plane-gate.

Returns Locate3DResult(x, y, z, confidence, method, extras).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco

from perception import camera_math as CM

# Object / plane priors (dimensions, NOT positions).
SUPPORT_Z = 0.015          # cube-centre plane (tabletop cube)
TOP_Z = 0.030              # cube top face (side = 30 mm), used for wrist top-down
CUBE_H = 0.030

# Triangulation conditioning gates.
TRI_MIN_ANGLE_DEG = 18.0   # below this the two rays are too parallel
TRI_MAX_GAP_M = 0.005      # closest-point residual; > this => a detect mis-fired,
                           # the two rays don't meet, so fall back to side-plane.

# Per-PROCESS scratch dir. The frame filenames are fixed ("find_side.png", "wtri_0.png"),
# so two runs sharing one directory would overwrite each other's images between the write
# and the sidecar's read — silently scoring one run against the other's frames. Isolating
# by pid makes concurrent benchmarks safe.
_TMP = (Path(__file__).resolve().parent.parent / "scratch" / "locate3d_tmp"
        / f"p{os.getpid()}")
_TMP.mkdir(parents=True, exist_ok=True)


@dataclass
class Locate3DResult:
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    confidence: float
    method: str
    extras: dict = field(default_factory=dict)

    def xy(self):
        return (self.x, self.y)


def _wrist_extrinsics(backend):
    cid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if cid < 0:
        return None
    backend._mj_forward()
    cam_pos = np.array(backend.data.cam_xpos[cid], float).copy()
    R_cw = np.array(backend.data.cam_xmat[cid], float).reshape(3, 3).copy()
    return cam_pos, R_cw


def _save_png(img: np.ndarray, name: str) -> str:
    from PIL import Image
    p = _TMP / name
    Image.fromarray(img).save(p)
    return str(p)


def _table_anchors(f, cx, cy, cam_pos, R_cw, bbox, W, H, n=24):
    """Honest metric-depth anchors: analytic z=0 table depth at pixels sampled in
    a ring AROUND the object bbox (surrounding table, never the object). Depth is
    pure geometry (ray∩z=0 from the FK pose); no oracle, no object truth."""
    x0, y0, x1, y1 = bbox
    cxp, cyp = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rad = max(x1 - x0, y1 - y0) * 1.8 + 20
    fwd = -R_cw[:, 2]
    anchors = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        u = cxp + rad * np.cos(ang)
        v = cyp + rad * np.sin(ang)
        if not (2 <= u < W - 2 and 2 <= v < H - 2):
            continue
        o, d = CM.back_project(u, v, f, cx, cy, cam_pos, R_cw)
        if abs(d[2]) < 1e-9:
            continue
        t = (0.0 - o[2]) / d[2]
        if t <= 0:
            continue
        hit = o + t * d
        depth = float((hit - o) @ fwd)
        if depth > 0:
            anchors.append([u, v, depth])
    return anchors


# ── Multi-view (N-ray) honest triangulation ────────────────────────────────────
# Wrist hover offsets (m) relative to the running best (x,y). The fixed side ray
# gives the wide baseline; these near-top-down wrist views add redundant rays so
# triangulate_nrays can REJECT a single mis-fired centroid instead of forcing the
# old plane-prior fallback. "Take more than one image" = robustness by redundancy.
WRIST_VIEW_OFFSETS = [(0.0, 0.0), (0.030, 0.0), (0.0, 0.030)]
GATE_Z = 0.015             # detector blob-IDENTITY gate only (not the coordinate)
AIM_Z = 0.0               # wrist-aim uses the calibratable TABLE plane, not object height
TRI_MAX_RESID_M = 0.004    # per-ray closest-point residual budget
TRI_MIN_COND = 0.05        # smallest-eigenvalue/N conditioning floor (rays too parallel)


# Fixed external-camera rig (az, el, dist, lookat) — NO wrist camera. Two cams
# 90° apart in azimuth give a wide, well-conditioned baseline; the steeper third
# adds vertical parallax (better z) and a redundant ray so a single mis-fired
# detection is dropped by triangulate_nrays. This is the real-world default:
# fixed cameras calibrated once, no arm-mounted camera required.
FIXED_CAMS = [
    (135.0, -35.0, 1.15, (0.0, 0.0, 0.04)),   # A — front-right (= render_side)
    (45.0, -35.0, 1.15, (0.0, 0.0, 0.04)),    # B — front-left (90° azimuth baseline)
    (90.0, -60.0, 1.15, (0.0, 0.0, 0.04)),    # C — front-centre, steeper (z baseline)
]
# The closest-point RESIDUAL (how much the surviving rays disagree) is the honest,
# ground-truth-free confidence: correct cells triangulate to <1.6 mm residual, a
# mis-detection makes the rays cross in the wrong place at >2 mm. Gate on it.
FIXED_MAX_RESID_M = 0.0018
FIXED_MIN_COND = 0.05      # parallax sanity only (near-parallel rays are unstable)
FIXED_MIN_INLIERS = 2


def locate_object_3d_fixed(backend, controller, client, name: str = "cube",
                           cams=FIXED_CAMS) -> Locate3DResult:
    """Plane-FREE metric (x,y,z) from a rig of FIXED external cameras — no wrist
    camera, no support-plane / height prior in the returned coordinate. Each cam's
    pose is analytic (real hardware: one ChArUco extrinsic per camera). Renders
    every cam, detects the object, back-projects to a world ray, and robustly
    triangulates (outlier rays dropped). Honest low-confidence miss if fewer than
    two rays survive or the geometry is ill-conditioned. Nothing reads qpos."""
    extras: dict = {"n_rays": 0, "uv": []}
    rays = []
    for j, (az, el, dist, lookat) in enumerate(cams):
        cam = CM.free_cam_params(az, el, dist, lookat)
        f, cx, cy, cpos, R = cam
        img = backend.render_view(az, el, dist, lookat)
        png = _save_png(img, f"fx_cam{j}.png")
        det = client.detect(png, cam, view="side", name=name, z_plane=GATE_Z)
        if det is None:
            continue
        u, v = det["uv"]
        extras["uv"].append([u, v])
        o, d = CM.back_project(u, v, f, cx, cy, cpos, R)
        rays.append((o, d))
    extras["n_rays"] = len(rays)
    if len(rays) < 2:
        return Locate3DResult(None, None, None, 0.0, "insufficient_views", extras)

    p, resid, n_in, cond = CM.triangulate_nrays(rays)
    if p is None:
        return Locate3DResult(None, None, None, 0.0, "tri_singular", extras)
    extras["tri_xyz"] = [float(p[0]), float(p[1]), float(p[2])]
    extras["tri_resid_mm"] = round(resid * 1000, 2)
    extras["tri_n_inliers"] = n_in
    extras["tri_cond"] = round(cond, 3)

    if n_in < FIXED_MIN_INLIERS or cond < FIXED_MIN_COND or resid > FIXED_MAX_RESID_M:
        return Locate3DResult(float(p[0]), float(p[1]), float(p[2]), 0.3,
                              "triangulation_illcond", extras)
    conf = 0.6 + 0.4 * max(0.0, 1.0 - resid / FIXED_MAX_RESID_M)
    return Locate3DResult(float(p[0]), float(p[1]), float(p[2]),
                          conf, "triangulation_fixed", extras)


# ── Single MOVING camera (eye-in-hand / monocular structure-from-motion) ───────
# ONE physical camera, several images at known FK poses, triangulated across its
# OWN motion. No second camera, no height prior. The camera MUST move — a fixed
# single camera on a static scene is scale-ambiguous; this is motion parallax, the
# monocular-SfM analogue of moving your head. A ±5.5 cm lateral orbit at ~0.14 m
# standoff gives ~40° between opposite rays — well-conditioned from one sensor.
MONOCAM_OVERVIEW = (0.23, 0.0, 0.26)      # high wide look to first find the object
MONOCAM_ORBIT = [(0.055, 0.0), (-0.055, 0.0), (0.0, 0.055), (0.0, -0.055)]
MONOCAM_HOVER_Z = 0.14
WS_CENTER = (0.23, 0.0)                    # nominal 'look at the table' aim prior


def locate_object_3d_monocam(backend, controller, client, name: str = "cube",
                             orbit=MONOCAM_ORBIT) -> Locate3DResult:
    """Plane-FREE metric (x,y,z) from ONE moving camera (the wrist/eye-in-hand):
    a wide overview to find the object, then an orbit of N poses around it, every
    frame back-projected to a world ray and robustly triangulated. No second
    camera, no support-plane / height prior in the coordinate. The table plane is
    used ONLY to aim (WS_CENTER prior + z=0), never for the answer. Honest decline
    if fewer than two rays survive or the geometry is ill-conditioned."""
    import wrist_refine as WR
    extras: dict = {"n_rays": 0, "uv": []}
    fw, cxw, cyw = CM.wrist_intrinsics()
    snap = WR.save_state(backend)
    rays = []
    try:
        # 1. overview shot — find the object; aim only (table-plane deproject).
        ox, oy, oz = MONOCAM_OVERVIEW
        WR.hover_top_down(controller, ox, oy, hover_z=oz)
        img = backend.render_wrist(); ext = _wrist_extrinsics(backend)
        if img is None or ext is None:
            return Locate3DResult(None, None, None, 0.0, "overview_fail", extras)
        wpos, wR = ext
        cam = (fw, cxw, cyw, wpos, wR)
        det = client.detect(_save_png(img, "mc_ov.png"), cam, view="wrist",
                            name=name, z_plane=GATE_Z, near_xy=WS_CENTER)
        if det is None:
            return Locate3DResult(None, None, None, 0.0, "overview_detect", extras)
        u, v = det["uv"]; extras["uv"].append([u, v])
        aim = CM.ray_plane(u, v, fw, cxw, cyw, wpos, wR, 0.0)     # table, aim only
        if aim is None:
            return Locate3DResult(None, None, None, 0.0, "aim_fail", extras)
        ax, ay = aim
        o, d = CM.back_project(u, v, fw, cxw, cyw, wpos, wR)
        rays.append((o, d))                                       # overview ray counts

        # 2. orbit the SAME camera around the aim; each pose adds a parallax ray.
        for k, (dx, dy) in enumerate(orbit):
            WR.hover_top_down(controller, ax + dx, ay + dy, hover_z=MONOCAM_HOVER_Z)
            img = backend.render_wrist(); ext = _wrist_extrinsics(backend)
            if img is None or ext is None:
                continue
            wpos, wR = ext
            det = client.detect(_save_png(img, f"mc{k}.png"),
                                (fw, cxw, cyw, wpos, wR), view="wrist",
                                name=name, z_plane=GATE_Z, near_xy=(ax, ay))
            if det is None:
                continue
            u, v = det["uv"]; extras["uv"].append([u, v])
            o, d = CM.back_project(u, v, fw, cxw, cyw, wpos, wR)
            rays.append((o, d))
            p, _, _, _ = CM.triangulate_nrays(rays)
            if p is not None:
                ax, ay = float(p[0]), float(p[1])
    finally:
        WR.restore_state(backend, snap)

    extras["n_rays"] = len(rays)
    if len(rays) < 2:
        return Locate3DResult(None, None, None, 0.0, "insufficient_views", extras)
    p, resid, n_in, cond = CM.triangulate_nrays(rays)
    if p is None:
        return Locate3DResult(None, None, None, 0.0, "tri_singular", extras)
    extras["tri_xyz"] = [float(p[0]), float(p[1]), float(p[2])]
    extras["tri_resid_mm"] = round(resid * 1000, 2)
    extras["tri_n_inliers"] = n_in
    extras["tri_cond"] = round(cond, 3)
    if n_in < FIXED_MIN_INLIERS or cond < FIXED_MIN_COND or resid > FIXED_MAX_RESID_M:
        return Locate3DResult(float(p[0]), float(p[1]), float(p[2]), 0.3,
                              "monocam_illcond", extras)
    conf = 0.6 + 0.4 * max(0.0, 1.0 - resid / FIXED_MAX_RESID_M)
    return Locate3DResult(float(p[0]), float(p[1]), float(p[2]),
                          conf, "monocam_tri", extras)


def locate_object_3d_multiview(backend, controller, client, name: str = "cube",
                               view_offsets=WRIST_VIEW_OFFSETS) -> Locate3DResult:
    """Plane-FREE metric (x,y,z): triangulate the fixed side ray with several
    wrist-camera rays taken at known FK poses. The returned coordinate uses NO
    support-plane and NO object-height prior — z is recovered from ray geometry.

    Honesty: the detector's z_plane (GATE_Z) only selects WHICH mask is the object
    (a coarse identity/workspace gate, ±15 mm irrelevant); the wrist AIM uses the
    calibratable table plane (AIM_Z=0, environmental geometry, not the object's
    size). If fewer than two rays survive or the geometry is ill-conditioned it
    returns an honest low-confidence miss — it NEVER returns a plane estimate at an
    assumed height. Nothing reads object qpos.
    """
    import wrist_refine as WR
    extras: dict = {"n_rays": 0}

    side_cam = CM.side_cam_params()
    fs, cxs, cys, spos, sR = side_cam
    side_img = backend.render_side()
    side_png = _save_png(side_img, "mv_side.png")
    sdet = client.detect(side_png, side_cam, view="side", name=name, z_plane=GATE_Z)
    if sdet is None:
        return Locate3DResult(None, None, None, 0.0, "side_detect_fail", extras)
    us, vs = sdet["uv"]
    extras["side_uv"] = [us, vs]
    o_s, d_s = CM.back_project(us, vs, fs, cxs, cys, spos, sR)
    rays = [(o_s, d_s)]

    aim = CM.ray_plane(us, vs, fs, cxs, cys, spos, sR, AIM_Z)   # table-plane aim only
    if aim is None:
        return Locate3DResult(None, None, None, 0.0, "side_ray_fail", extras)
    ax, ay = aim

    fw, cxw, cyw = CM.wrist_intrinsics()
    snap = WR.save_state(backend)
    wrist_uvs = []
    try:
        for k, (dx, dy) in enumerate(view_offsets):
            WR.hover_top_down(controller, ax + dx, ay + dy)
            wimg = backend.render_wrist()
            ext = _wrist_extrinsics(backend)
            if wimg is None or ext is None:
                continue
            wpos, wR = ext
            wpng = _save_png(wimg, f"mv_wrist{k}.png")
            wdet = client.detect(wpng, (fw, cxw, cyw, wpos, wR), view="wrist",
                                 name=name, z_plane=GATE_Z, near_xy=(ax, ay))
            if wdet is None:
                continue
            uw, vw = wdet["uv"]
            wrist_uvs.append([uw, vw])
            o_w, d_w = CM.back_project(uw, vw, fw, cxw, cyw, wpos, wR)
            rays.append((o_w, d_w))
            # progressive re-aim from the running triangulation (keeps the next
            # wrist view centred on the object -> least oblique-centroid bias).
            p, _, _, _ = CM.triangulate_nrays(rays)
            if p is not None:
                ax, ay = float(p[0]), float(p[1])
    finally:
        WR.restore_state(backend, snap)

    extras["wrist_uv"] = wrist_uvs
    extras["n_rays"] = len(rays)
    if len(rays) < 2:
        return Locate3DResult(None, None, None, 0.0, "insufficient_views", extras)

    p, resid, n_in, cond = CM.triangulate_nrays(rays)
    if p is None:
        return Locate3DResult(None, None, None, 0.0, "tri_singular", extras)
    extras["tri_xyz"] = [float(p[0]), float(p[1]), float(p[2])]
    extras["tri_resid_mm"] = round(resid * 1000, 2)
    extras["tri_n_inliers"] = n_in
    extras["tri_cond"] = round(cond, 3)

    if cond < TRI_MIN_COND or resid > TRI_MAX_RESID_M:
        return Locate3DResult(float(p[0]), float(p[1]), float(p[2]), 0.3,
                              "triangulation_illcond", extras)
    conf = 0.6 + 0.4 * max(0.0, 1.0 - resid / TRI_MAX_RESID_M)
    return Locate3DResult(float(p[0]), float(p[1]), float(p[2]),
                          conf, "triangulation_nview", extras)


def locate_object_3d(backend, controller, client, name: str = "cube",
                     want_depth: bool = False,
                     support_z: float = SUPPORT_Z, top_z: float = TOP_Z
                     ) -> Locate3DResult:
    extras: dict = {}

    # ── 1. side camera ────────────────────────────────────────────────────────
    side_cam = CM.side_cam_params()
    fs, cxs, cys, spos, sR = side_cam
    side_img = backend.render_side()
    side_png = _save_png(side_img, "side.png")
    sdet = client.detect(side_png, side_cam, view="side", name=name,
                         z_plane=support_z)
    if sdet is None:
        return Locate3DResult(None, None, None, 0.0, "side_detect_fail", extras)
    us, vs = sdet["uv"]
    extras["side_uv"] = [us, vs]

    # coarse plane estimate (the honest side-cam-only baseline)
    coarse = CM.ray_plane(us, vs, fs, cxs, cys, spos, sR, support_z)
    if coarse is None:
        return Locate3DResult(None, None, None, 0.0, "side_ray_fail", extras)
    x0, y0 = coarse
    extras["side_plane_xy"] = [x0, y0]

    # side viewing ray (one triangulation leg)
    o_s, d_s = CM.back_project(us, vs, fs, cxs, cys, spos, sR)

    # ── 2-3. wrist hover + detect (state-neutral) ─────────────────────────────
    import wrist_refine as WR
    snap = WR.save_state(backend)
    wdet = None
    wrist_ext = None
    try:
        WR.hover_top_down(controller, x0, y0)
        wrist_img = backend.render_wrist()
        wrist_ext = _wrist_extrinsics(backend)
        if wrist_img is not None and wrist_ext is not None:
            wpos, wR = wrist_ext
            fw, cxw, cyw = CM.wrist_intrinsics()
            wrist_png = _save_png(wrist_img, "wrist.png")
            wcam = (fw, cxw, cyw, wpos, wR)
            # near_xy = coarse side estimate: on-table objects deproject near it,
            # the gripper (well above the table) deprojects far -> rejected.
            wdet = client.detect(wrist_png, wcam, view="wrist", name=name,
                                 z_plane=support_z, near_xy=(x0, y0))
    finally:
        WR.restore_state(backend, snap)

    # ── 4. triangulation (plane-free) ─────────────────────────────────────────
    tri = None
    if wdet is not None and wrist_ext is not None:
        uw, vw = wdet["uv"]
        extras["wrist_uv"] = [uw, vw]
        wpos, wR = wrist_ext
        fw, cxw, cyw = CM.wrist_intrinsics()
        o_w, d_w = CM.back_project(uw, vw, fw, cxw, cyw, wpos, wR)
        p3d, gap, angle = CM.triangulate(o_s, d_s, o_w, d_w)
        extras["tri_xyz"] = [float(p3d[0]), float(p3d[1]), float(p3d[2])]
        extras["tri_gap_mm"] = round(gap * 1000, 2)
        extras["tri_angle_deg"] = round(angle, 1)
        # wrist-plane estimate (top-down deproject to the top face)
        wxy = CM.ray_plane(uw, vw, fw, cxw, cyw, wpos, wR, top_z)
        if wxy is not None:
            extras["wrist_plane_xy"] = [wxy[0], wxy[1]]
        tri = (p3d, gap, angle)

    # ── Optional Depth-Anything z at the side centroid (fallback + cross-check).
    # Computed whenever want_depth so the eval can score depth on every cell.
    if want_depth:
        bh = 20
        bbox = (us - bh, vs - bh, us + bh, vs + bh)
        anchors = _table_anchors(fs, cxs, cys, spos, sR, bbox,
                                 CM.SIDE_W, CM.SIDE_H)
        dz = client.depth(side_png, [us, vs], anchors) if anchors else None
        if dz is not None:
            extras["depth_side_m"] = round(dz, 4)

    # ── 5. fuse ───────────────────────────────────────────────────────────────
    # Prefer a well-conditioned triangulation (metric x,y,z incl. plane-free
    # height). A large closest-point gap means one centroid mis-fired and the
    # rays don't intersect -> fall back to the validated side-cam plane estimate
    # (~1-2 mm for a tabletop cube), which never needs the wrist view.
    if tri is not None:
        p3d, gap, angle = tri
        if angle >= TRI_MIN_ANGLE_DEG and gap <= TRI_MAX_GAP_M:
            conf = 0.6 + 0.4 * max(0.0, 1.0 - gap / TRI_MAX_GAP_M)
            return Locate3DResult(float(p3d[0]), float(p3d[1]), float(p3d[2]),
                                  conf, "triangulation", extras)
        extras["tri_rejected"] = "gap>thresh" if gap > TRI_MAX_GAP_M else "parallel"

    # side-plane fallback (tabletop support-plane assumption, z = support_z)
    return Locate3DResult(x0, y0, support_z, 0.5, "side_plane", extras)
