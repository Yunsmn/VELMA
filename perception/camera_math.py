"""Shared camera geometry for the honest locate_object_3d pipeline.

Pure-numpy so it imports in BOTH the robot venv (py3.14, renders MuJoCo) and the
scratch venv (py3.11, runs the models). Nothing here reads object qpos.

Two honest pose sources:
  * Side camera  -> fixed free-cam calibration (az/el/dist/lookat constants).
                    A known extrinsic; independent of the object.
  * Wrist camera -> read at run time from data.cam_xpos / data.cam_xmat, i.e.
                    the robot's forward kinematics of its own mount (hand-eye).

Both give (cam_pos, R_cw): R_cw columns are the camera-frame axes in world.
MuJoCo camera convention: looks along local -Z, +Y up, +X right (so the pixel
ray in camera frame is [(u-cx)/f, -(v-cy)/f, -1]).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# ── Side camera (matches simulation.py:render_side and honest_camera_grid) ─────
SIDE_FOVY = 45.0
SIDE_W = 640
SIDE_H = 480
SIDE_LOOKAT = np.array([0.0, 0.0, 0.04])
SIDE_DIST = 1.15
SIDE_AZ = 135.0
SIDE_EL = -35.0

# ── Wrist camera intrinsics (so101_new_calib.xml: fovy 70.5, 640x480) ─────────
WRIST_FOVY = 70.5
WRIST_W = 640
WRIST_H = 480


def free_cam_params(az_deg: float, el_deg: float, dist: float, lookat,
                    fovy: float = SIDE_FOVY, W: int = SIDE_W, H: int = SIDE_H
                    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """(f, cx, cy, cam_pos, R_cw) for a MuJoCo free camera at (az,el,dist,lookat).

    Analytic and honest — a known extrinsic (on real hardware: one ChArUco
    extrinsic calibration per fixed camera). Any number of fixed external cameras
    get their pose here; triangulation across them needs no wrist camera.
    """
    f = 0.5 * H / math.tan(math.radians(fovy * 0.5))
    cx = W / 2.0
    cy = H / 2.0

    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cos_az, sin_az = math.cos(az), math.sin(az)
    cos_el, sin_el = math.cos(el), math.sin(el)

    lookat = np.asarray(lookat, float)
    cam_pos = lookat + dist * np.array([
        -cos_el * cos_az,
        -cos_el * sin_az,
        -sin_el,
    ])
    e_x = np.array([sin_az, -cos_az, 0.0])
    e_y = np.array([-sin_el * cos_az, -sin_el * sin_az, cos_el])
    e_z = np.array([-cos_el * cos_az, -cos_el * sin_az, -sin_el])
    R_cw = np.column_stack([e_x, e_y, e_z])
    return f, cx, cy, cam_pos, R_cw


def side_cam_params() -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """(f, cx, cy, cam_pos, R_cw) for the fixed side camera — analytic, honest."""
    return free_cam_params(SIDE_AZ, SIDE_EL, SIDE_DIST, SIDE_LOOKAT)


def wrist_intrinsics() -> tuple[float, float, float]:
    """(f, cx, cy) for the wrist camera (square pixels)."""
    f = 0.5 * WRIST_H / math.tan(math.radians(WRIST_FOVY * 0.5))
    return f, WRIST_W / 2.0, WRIST_H / 2.0


def back_project(u: float, v: float, f: float, cx: float, cy: float,
                 cam_pos: np.ndarray, R_cw: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Pixel (u,v) -> (origin, unit_direction) world-frame ray.

    origin is the camera centre; direction is the outgoing viewing ray.
    """
    d_c = np.array([(u - cx) / f, -(v - cy) / f, -1.0])
    d_w = R_cw @ d_c
    n = np.linalg.norm(d_w)
    if n < 1e-12:
        return cam_pos.astype(float), d_w
    return cam_pos.astype(float), d_w / n


def ray_plane(u: float, v: float, f: float, cx: float, cy: float,
              cam_pos: np.ndarray, R_cw: np.ndarray,
              z_plane: float) -> Optional[tuple[float, float]]:
    """Back-project (u,v) and intersect z=z_plane -> (x,y) or None."""
    o, d = back_project(u, v, f, cx, cy, cam_pos, R_cw)
    if abs(d[2]) < 1e-9:
        return None
    t = (z_plane - o[2]) / d[2]
    if t <= 0.0:
        return None
    hit = o + t * d
    return float(hit[0]), float(hit[1])


def point_at_depth(u: float, v: float, f: float, cx: float, cy: float,
                   cam_pos: np.ndarray, R_cw: np.ndarray,
                   depth_m: float) -> np.ndarray:
    """Metric world point for pixel (u,v) at forward-axis depth `depth_m`.

    depth_m is measured along the camera's forward (-Z) axis, matching how the
    metric-depth anchors below are defined. This is the SINGLE-VIEW deprojection
    that recovers an object's own height from calibrated monocular depth — no
    plane-at-object assumption, no second view, no triangulation.
    """
    o, d = back_project(u, v, f, cx, cy, cam_pos, R_cw)
    fwd = -R_cw[:, 2]
    ddot = float(d @ fwd)
    if abs(ddot) < 1e-9:
        return o.astype(float)
    t = depth_m / ddot
    return (o + t * d).astype(float)


def table_anchor_ring(bbox, f: float, cx: float, cy: float,
                      cam_pos: np.ndarray, R_cw: np.ndarray,
                      W: int, H: int, z_plane: float = 0.0, n: int = 24):
    """Honest metric-depth calibration anchors around an object bbox.

    Samples a ring of pixels AROUND (never on) the object and gives each the
    analytic forward-axis depth of the support plane z=z_plane at that pixel
    (ray∩plane from the camera pose). Pure geometry — no object truth, no oracle.
    These pin the affine scale of the (up-to-affine) monocular depth model so the
    object's own depth reads out in metres. Returns [[u,v,depth_m],...].
    """
    x0, y0, x1, y1 = bbox
    cxp, cyp = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rad = max(x1 - x0, y1 - y0) * 1.8 + 20
    fwd = -R_cw[:, 2]
    cam_pos = np.asarray(cam_pos, float)
    R_cw = np.asarray(R_cw, float)
    anchors = []
    for k in range(n):
        ang = 2 * math.pi * k / n
        u = cxp + rad * math.cos(ang)
        v = cyp + rad * math.sin(ang)
        if not (2 <= u < W - 2 and 2 <= v < H - 2):
            continue
        o, d = back_project(u, v, f, cx, cy, cam_pos, R_cw)
        if abs(d[2]) < 1e-9:
            continue
        t = (z_plane - o[2]) / d[2]
        if t <= 0:
            continue
        hit = o + t * d
        depth = float((hit - o) @ fwd)
        if depth > 0:
            anchors.append([float(u), float(v), depth])
    return anchors


def triangulate_nrays(rays, max_residual_m: float = 0.004, min_rays: int = 2):
    """Robust least-squares intersection of N world rays — plane-free metric 3D.

    rays: iterable of (origin, direction) world-frame pairs (directions need not
    be unit). Solves argmin_p Σ dist(p, line_i)² via the normal equations
        A p = b,  A = Σ (I - d_i d_iᵀ),  b = Σ (I - d_i d_iᵀ) o_i
    then RANSAC-lite: if the worst ray's perpendicular residual exceeds
    max_residual_m and > min_rays remain, drop that ray (a mis-fired centroid
    gives an outlier ray) and refit. This is how extra views buy robustness — one
    bad detection is rejected instead of poisoning the estimate.

    Returns (point3d, mean_residual_m, n_inliers, cond) where cond ∈ [0,1] is the
    smallest eigenvalue of A / n_inliers (≈0 ⇒ rays near-parallel/ill-conditioned,
    →1 ⇒ well spread). Returns (None, None, 0, 0.0) if it cannot solve.
    """
    R = [(np.asarray(o, float), np.asarray(d, float) / (np.linalg.norm(d) + 1e-12))
         for o, d in rays]
    active = list(range(len(R)))
    if len(active) < 2:
        return None, None, 0, 0.0
    while True:
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in active:
            o, d = R[i]
            M = np.eye(3) - np.outer(d, d)
            A += M
            b += M @ o
        try:
            p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None, None, 0, 0.0
        resid = [float(np.linalg.norm((np.eye(3) - np.outer(R[i][1], R[i][1]))
                                      @ (p - R[i][0]))) for i in active]
        worst = int(np.argmax(resid))
        if resid[worst] <= max_residual_m or len(active) <= min_rays:
            eig = float(np.linalg.eigvalsh(A)[0])   # smallest eigenvalue
            cond = eig / max(len(active), 1)
            return p, float(np.mean(resid)), len(active), cond
        active.pop(worst)


def triangulate(o1: np.ndarray, d1: np.ndarray,
                o2: np.ndarray, d2: np.ndarray
                ) -> tuple[np.ndarray, float, float]:
    """Closest-point (least-squares intersection) of two world rays.

    Returns (point3d, gap_m, angle_deg):
      point3d : midpoint of the mutual closest points (metric world xyz)
      gap_m   : distance between the two closest points (triangulation residual)
      angle_deg : angle between the rays; near 0 => ill-conditioned (parallel).

    This is PLANE-FREE: z falls out of the geometry, so it generalises to
    unknown-height / stacked objects the ray_plane path cannot handle.
    """
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    w0 = o1 - o2
    a = float(d1 @ d1)          # = 1
    b = float(d1 @ d2)
    c = float(d2 @ d2)          # = 1
    d = float(d1 @ w0)
    e = float(d2 @ w0)
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        # Parallel: fall back to projecting o2 onto ray1.
        s = -d / a
        p = o1 + s * d1
        return p, float(np.linalg.norm(o1 - o2)), 0.0
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    mid = 0.5 * (p1 + p2)
    gap = float(np.linalg.norm(p1 - p2))
    cos_ang = max(-1.0, min(1.0, b))
    angle = math.degrees(math.acos(abs(cos_ang)))
    return mid, gap, angle
