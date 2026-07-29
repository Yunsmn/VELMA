"""Locate an object in BASE coordinates from a single wrist-camera view.

This is the hardware counterpart of the simulation's find_object -> grasp path.
It became possible only once the joint offsets were measured: the wrist camera is
rigid to the gripper, so forward kinematics now gives the camera's pose in base
coordinates for free, exactly as `side_cam_params` did in the sim.

Two independent estimates are produced from the same detection, because on this
rig each checks the other:

  plane  - intersect the pixel's ray with the table (z = 0). Exact for anything
           resting on the table, and depends on no learned model.
  depth  - a monocular depth network. Generalises to objects off the table, but
           the network is RELATIVE (inverse depth up to an affine transform), so
           it is anchored against the table pixels whose true distance the
           geometry already knows.

Agreement between the two is the confidence signal. Disagreement means either the
intrinsics are wrong or the depth model is out of its depth, and the caller
should decline rather than grasp.

INTRINSICS: the camera is an XZ-20250413-L2.0, 1920x1080, 137 deg DIAGONAL field
of view. That is a fisheye, and the MuJoCo model's wrist_cam (fovy 70.5, i.e.
110 deg diagonal) describes a different part entirely — using it mis-aims a
corner ray by about 13 deg. Two projection models are supported because a 137 deg
lens is nowhere near rectilinear and the truth sits between them; keep the target
near the image centre, where they agree, until the fit is measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

# Camera: XZ-20250413-L2.0, 137 deg diagonal FOV.
CAMERA_DFOV_DEG = 137.0
PROJECTION: Literal["pinhole", "equidistant"] = "equidistant"

TABLE_Z_M = 0.0            # base frame; validated to 1.6 mm across three poses
GRASP_Z_M = 0.015          # gripper height for an object resting on the table

# Table pixels used to anchor the relative depth network. Sampled on a grid and
# filtered to those whose ray actually meets the table in front of the camera.
ANCHOR_GRID = 24
ANCHOR_MIN_POINTS = 40
ANCHOR_MAX_RESIDUAL = 0.25  # fraction; above this the affine fit is not trusted


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray      # (3,) camera centre in base frame, metres
    rotation: np.ndarray      # (3,3) camera->base; columns are the camera axes
    width: int
    height: int

    @property
    def focal_px(self) -> float:
        return focal_for(self.width, self.height, PROJECTION)


def focal_for(width: int, height: int,
              projection: str = PROJECTION, dfov_deg: float = CAMERA_DFOV_DEG) -> float:
    """Focal length in pixels implied by the diagonal field of view."""
    half_diag = np.hypot(width, height) / 2.0
    half_angle = np.radians(dfov_deg / 2.0)
    if projection == "pinhole":
        return half_diag / np.tan(half_angle)
    return half_diag / half_angle          # equidistant: r = f * theta


def ray_direction(u: float, v: float, cam: CameraPose) -> np.ndarray:
    """Unit ray through pixel (u, v), in BASE coordinates.

    MuJoCo's camera convention: +x right, +y up, and the camera looks down its
    own -z. Both projection models are handled because the lens is wide enough
    that the choice matters away from the centre.
    """
    f = cam.focal_px
    dx = u - cam.width / 2.0
    dy = -(v - cam.height / 2.0)          # image v grows downward, camera y grows up

    if PROJECTION == "pinhole":
        d_cam = np.array([dx, dy, -f], dtype=float)
    else:
        r = float(np.hypot(dx, dy))
        theta = r / f                      # equidistant
        if r < 1e-9:
            d_cam = np.array([0.0, 0.0, -1.0])
        else:
            d_cam = np.array([np.sin(theta) * dx / r,
                              np.sin(theta) * dy / r,
                              -np.cos(theta)])
    d_cam /= np.linalg.norm(d_cam)
    return cam.rotation @ d_cam


def intersect_plane(origin: np.ndarray, direction: np.ndarray,
                    plane_z: float = TABLE_Z_M) -> Optional[np.ndarray]:
    """Where a ray meets a horizontal plane. None if it points away from it."""
    if abs(direction[2]) < 1e-9:
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t <= 0:
        return None                        # plane is behind the camera
    return origin + t * direction


def locate_via_plane(u: float, v: float, cam: CameraPose,
                     plane_z: float = TABLE_Z_M) -> Optional[np.ndarray]:
    """Object position assuming it rests on the table. No learned model involved."""
    return intersect_plane(cam.position, ray_direction(u, v, cam), plane_z)


def anchor_depth_map(depth_rel: np.ndarray, cam: CameraPose,
                     exclude_box: Optional[tuple] = None) -> Optional[dict]:
    """Fit relative (inverse) depth to metres using the table as ground truth.

    Depth-Anything predicts inverse depth up to an unknown affine transform:
    1/Z = a * pred + b. Every pixel whose ray meets the table has a distance the
    geometry already knows exactly, which gives as many (pred, 1/Z) pairs as we
    care to sample — no external measurement and no assumed scale.
    """
    height, width = depth_rel.shape[:2]
    preds, inv_true = [], []

    for v in np.linspace(0, height - 1, ANCHOR_GRID):
        for u in np.linspace(0, width - 1, ANCHOR_GRID):
            if exclude_box is not None:
                x0, y0, x1, y1 = exclude_box
                if x0 <= u <= x1 and y0 <= v <= y1:
                    continue               # do not anchor on the object itself
            direction = ray_direction(u, v, cam)
            hit = intersect_plane(cam.position, direction)
            if hit is None:
                continue
            distance = float(np.linalg.norm(hit - cam.position))
            if not (0.03 < distance < 3.0):
                continue
            preds.append(float(depth_rel[int(v), int(u)]))
            inv_true.append(1.0 / distance)

    if len(preds) < ANCHOR_MIN_POINTS:
        return None

    preds, inv_true = np.asarray(preds), np.asarray(inv_true)
    a, b = np.polyfit(preds, inv_true, 1)
    residual = float(np.median(np.abs((a * preds + b) - inv_true) / np.maximum(inv_true, 1e-6)))
    return {"a": float(a), "b": float(b), "n_anchors": len(preds),
            "median_rel_residual": residual}


def locate_via_depth(u: float, v: float, depth_rel: np.ndarray, cam: CameraPose,
                     fit: dict) -> Optional[np.ndarray]:
    """Object position from the anchored monocular depth at its pixel."""
    inv_z = fit["a"] * float(depth_rel[int(v), int(u)]) + fit["b"]
    if inv_z <= 1e-6:
        return None                        # non-physical, behind the camera
    distance = 1.0 / inv_z
    if not (0.02 < distance < 3.0):
        return None
    return cam.position + distance * ray_direction(u, v, cam)
