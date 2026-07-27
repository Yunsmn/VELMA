"""HONEST wrist-camera coarse-to-fine refinement of a tabletop object's (x,y).

Stage 2 of the perception pipeline (docs/perception_pipeline_ellmer.md §3.5):
after a gross side-camera estimate (x0,y0), hover the wrist camera above it and
re-localize at a near-top-down view. Two honest wins over the side cam:
  * ~7x finer pixel scale at ~0.10 m standoff vs 1.15 m,
  * near-top-down view removes the oblique silhouette-centroid bias that biases
    the side-cam ray-plane estimate for far/oblique cubes.

HONESTY
  * The wrist-camera pose is READ from the sim (`cam_xpos`/`cam_xmat`) — that is
    the robot's own forward kinematics of its camera mount (hand-eye on real
    hardware), NOT the object position.
  * The refined correction is derived purely from the RENDERED wrist image.
  * z_plane = table + half-cube-height (0.015) is the known support-plane prior,
    the same one the side cam already uses. Ground truth (object xy) is NEVER
    read here.

Nothing in this module reads `cube_position_m` or object qpos.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import mujoco


# ── Wrist camera intrinsics (so101_new_calib.xml: fovy=70.5, 640x480) ──────────
WRIST_FOVY = 70.5
WRIST_W = 640
WRIST_H = 480


def wrist_intrinsics() -> tuple[float, float, float]:
    """(f_px, cx_px, cy_px) for the wrist camera (square pixels)."""
    f = 0.5 * WRIST_H / math.tan(math.radians(WRIST_FOVY * 0.5))
    return f, WRIST_W / 2.0, WRIST_H / 2.0


def wrist_extrinsics(backend) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(cam_pos, R_cw) for the wrist camera, read from the sim at the CURRENT pose.

    cam_pos: camera position in world (from data.cam_xpos).
    R_cw:    3x3 camera->world rotation (from data.cam_xmat). Columns are the
             camera-frame axes expressed in world. MuJoCo cameras look along
             local -Z with +Y up, +X right (same convention as the side-cam code).
    Reading the camera's own pose is FK of the robot's mount — honest.
    """
    cid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if cid < 0:
        return None
    backend._mj_forward()
    cam_pos = np.array(backend.data.cam_xpos[cid], dtype=float).copy()
    R_cw = np.array(backend.data.cam_xmat[cid], dtype=float).reshape(3, 3).copy()
    return cam_pos, R_cw


def ray_plane(u: float, v: float, f: float, cx: float, cy: float,
              cam_pos: np.ndarray, R_cw: np.ndarray,
              z_plane: float) -> Optional[tuple[float, float]]:
    """Back-project pixel (u,v) and intersect with z=z_plane. Returns (x,y)."""
    d_c = np.array([(u - cx) / f, -(v - cy) / f, -1.0])
    d_w = R_cw @ d_c
    if abs(d_w[2]) < 1e-9:
        return None
    t = (z_plane - cam_pos[2]) / d_w[2]
    if t <= 0.0:
        return None
    hit = cam_pos + t * d_w
    return float(hit[0]), float(hit[1])


def optical_axis_world(R_cw: np.ndarray) -> np.ndarray:
    """World-frame forward (viewing) direction of the camera (local -Z)."""
    return -R_cw[:, 2]


# ── Coarse-to-fine refine ──────────────────────────────────────────────────────
HOVER_Z = 0.14          # EE world-z for the top-down wrist look (~0.11 m standoff)
HOVER_WF = 90.0         # wrist_flex that points the gripper (and cam) straight down
REJECT_MM = 30.0        # if refine disagrees with coarse by more than this, keep coarse


# Minimum transit height over the target before descending to hover_z. This is a FLOOR, not
# a fixed altitude: the transit must never be lower than the observation height itself.
# With a hardcoded 0.20 an object requiring a 0.263 hover was approached by first descending
# to 0.20 — i.e. BELOW the intended hover — and then translating across, which swept the
# gripper through a cube standing on a 70 mm pedestal and knocked it 137 mm onto the floor
# before a single frame was taken. The perception then correctly measured a cube that the
# approach had displaced, which read as a 134 mm "localisation error" against the restored
# ground truth. Another tabletop assumption hiding in a constant.
CLEAR_Z = 0.20
CLEAR_MARGIN = 0.06     # transit at least this far above the hover height


def save_state(backend) -> dict:
    """Snapshot everything grasp() depends on: full physics (time, qpos, qvel,
    act, plugins) PLUS the actuator ctrl targets (ctrl is an INPUT, not part of
    mjSTATE_FULLPHYSICS, but grasp's gripper ramp reads data.ctrl)."""
    d = backend.data
    return {
        "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "act": d.act.copy(),
        "ctrl": d.ctrl.copy(), "time": float(d.time),
    }


def restore_state(backend, snap: dict) -> None:
    """Restore a snapshot so grasp() starts byte-identical to the no-refine path."""
    d = backend.data
    d.qpos[:] = snap["qpos"]
    d.qvel[:] = snap["qvel"]
    d.act[:] = snap["act"]
    d.ctrl[:] = snap["ctrl"]
    d.time = snap["time"]
    mujoco.mj_forward(backend.model, backend.data)


def hover_top_down(controller, x0: float, y0: float,
                   hover_z: float = HOVER_Z, wf: float = HOVER_WF) -> None:
    """Move the EE above (x0,y0) at hover_z with a straight-down wrist pose, jaws
    open, APPROACHING FROM ABOVE (rehome high -> over target -> descend). Aims from
    the CAMERA ESTIMATE (x0,y0) — never ground truth. The from-above path keeps the
    forearm clear of the table/cube and gives the cleanest top-down wrist shot."""
    clear_z = max(CLEAR_Z, hover_z + CLEAR_MARGIN)
    pan = math.degrees(math.atan2(-y0, x0))

    if hover_z > HOVER_Z + 1e-6:
        # TALL TARGET -> pure Cartesian transit, skipping the joint-space rehome below.
        # MEASURED: that rehome's posture (shoulder_lift=0, elbow_flex=0) places the
        # end-effector at z=0.075 — BELOW the top of a cube standing on a 70 mm pedestal
        # (0.10) — and set_joint_angles interpolates in JOINT space, so panning to face the
        # target drags the gripper straight through the object. It knocked the cube 153 mm
        # off its pedestal before any frame was captured; raising the transit height and
        # lifting before the pan both failed, because the collision is in the joint path,
        # not the end-effector path. The rehome posture is itself a tabletop assumption.
        # Going up first, rotating only the WRIST joints (which do not swing the arm), then
        # translating at altitude keeps every intermediate pose above the obstacle.
        ee = controller.ik.get_ee_position()
        if float(ee[2]) < clear_z:
            controller.move_to_cartesian(float(ee[0]), float(ee[1]), clear_z,
                                         lock_wrist=True, gain=0.2)
        controller.set_joint_angles({"wrist_flex": wf, "wrist_roll": 90.0 + pan},
                                    gripper_pct=65.0)
        controller.move_to_cartesian(x0, y0, clear_z, lock_wrist=True, gain=0.2)
        controller.move_to_cartesian(x0, y0, hover_z, lock_wrist=True, gain=0.2)
        return

    controller.set_joint_angles({
        "shoulder_pan": pan, "shoulder_lift": 0.0, "elbow_flex": 0.0,
        "wrist_flex": wf, "wrist_roll": 90.0 + pan,
    }, gripper_pct=65.0)
    clear_z = max(CLEAR_Z, hover_z + CLEAR_MARGIN)
    controller.move_to_cartesian(x0, y0, clear_z, lock_wrist=True, gain=0.2)
    controller.move_to_cartesian(x0, y0, hover_z, lock_wrist=True, gain=0.2)


def refine_xy(backend, controller, detect_fn, x0: float, y0: float,
              top_z: float, hover_z: float = HOVER_Z, wf: float = HOVER_WF,
              reject_mm: float = REJECT_MM):
    """Wrist coarse-to-fine refine of a tabletop object's (x,y).

    detect_fn(rgb) -> (u,v) pixel centroid or None (colour/mask segmentation).
    top_z: deprojection plane = the object's VISIBLE-face height from above. For a
           cube resting on the table this is the top face z = table + height, a
           known object DIMENSION (not its position). Deprojecting the top-down
           centroid to the top-face plane removes the constant lateral-offset bias
           that deprojecting to the centre plane injects.

    Returns (x1, y1, info). Falls back to (x0,y0) if the wrist view can't localize
    or disagrees wildly with the coarse seed (info['source'] tells which).

    STATE-NEUTRAL: the full physics state is snapshotted on entry and restored on
    exit, so the hover motion changes ONLY the returned estimate — never the arm
    or cube state the caller's grasp() then starts from. This makes the refine a
    pure perception upgrade: identical kinematics to the no-refine path, only a
    better (x,y).
    """
    info = {"source": "coarse", "wrist_px": 0}
    snap = save_state(backend)
    try:
        hover_top_down(controller, x0, y0, hover_z, wf)
        img = backend.render_wrist()
        if img is None:
            return x0, y0, info
        ext = wrist_extrinsics(backend)
        if ext is None:
            return x0, y0, info
        pix = detect_fn(img)
        if pix is None:
            info["source"] = "coarse_wrist_nodetect"
            return x0, y0, info
        cam_pos, R_cw = ext
        f, cx, cy = wrist_intrinsics()
        xy = ray_plane(pix[0], pix[1], f, cx, cy, cam_pos, R_cw, top_z)
        if xy is None:
            info["source"] = "coarse_ray_fail"
            return x0, y0, info
        disagree = math.hypot(xy[0] - x0, xy[1] - y0) * 1000.0
        info["wrist_px"] = pix
        info["disagree_mm"] = round(disagree, 2)
        if disagree > reject_mm:
            info["source"] = "coarse_reject"
            return x0, y0, info
        info["source"] = "wrist"
        return float(xy[0]), float(xy[1]), info
    finally:
        restore_state(backend, snap)


# ── Aiming the CAMERA rather than the end effector ────────────────────────────────
# The wrist camera is bolted to the end effector, but its WORLD orientation is not
# fixed: the pointing angle is the sum of the three parallel-axis joints, measured as
#     tilt ~= 90 deg - (shoulder_lift + elbow_flex + wrist_flex)
# so as the IK folds the arm to hover higher, the camera swings over. Measured over a
# hover sweep: 28.8 deg at hover 0.14 rising to 69.0 deg at 0.30. `lock_wrist=True`
# does NOT prevent this — it pins the wrist JOINT ANGLE, not the camera's world pose.
#
# The consequence is that "park the EE above the target" does not mean "look at the
# target". Where the optical axis actually lands, versus the point aimed at:
#     hover 0.14 -> 39 mm off      0.22 -> 139 mm      0.28 -> 460 mm, OUT OF FRAME
# The validated tabletop result survives only because at hover 0.14 a ~40 mm miss still
# sits inside a 318 mm-wide footprint. It degrades badly with height, which is why
# raised objects return `no_colour_match` on most views.
#
# It cannot be fixed by pointing the wrist further down: holding the camera vertical
# needs wrist_flex = 90 - lift - elbow, about 167 deg at hover 0.30, against a +-95 deg
# joint limit. So instead we CLOSE THE LOOP on the projection: move, read the camera
# pose from FK, project the target, shift the EE by the lateral error, repeat.
AIM_ITERS = 7
AIM_GAIN = 0.4            # undamped this diverges — see the clamp note below
AIM_MAX_STEP_M = 0.04     # one undamped step once flew the camera from 0.28 m standoff
                          # to 0.16 m and threw the target 820 px off frame
AIM_TOL_PX = 40.0


def aim_camera_at(backend, controller, target_xyz, hover_z: float,
                  iters: int = AIM_ITERS, gain: float = AIM_GAIN,
                  max_step_m: float = AIM_MAX_STEP_M, tol_px: float = AIM_TOL_PX):
    """Drive the EE so the wrist camera's centre lands on `target_xyz`.

    `target_xyz` MUST be a perceived estimate, never ground truth — it is what we are
    trying to look at, and feeding it truth would make the stage look better than it is.
    Uses only the FK camera pose, so it is honest hand-eye on hardware too.

    Returns (ok, info). `ok` is False when the target could not be brought into frame,
    which the caller should treat as "do not use this view" rather than detecting
    whatever else happens to be visible.

    Keeps the BEST iterate, not the last: the linearisation "the camera translates with
    the EE" ignores the orientation swing, so the final step is sometimes the one that
    overshot. Convergence is not guaranteed — at a small radius with a folded arm the
    Jacobian from EE-xy to image-v collapses and the vertical error stops responding.
    """
    from perception import camera_math as CM

    f, cx, cy = CM.wrist_intrinsics()
    P = np.asarray(target_xyz, float)
    ee = np.array([P[0], P[1], hover_z], float)
    best = None

    for k in range(iters):
        controller.move_to_cartesian(float(ee[0]), float(ee[1]), float(ee[2]),
                                     lock_wrist=True, gain=0.5)
        ext = wrist_extrinsics(backend)
        if ext is None:
            break
        cam_pos, R_cw = ext
        xc, yc, zc = R_cw.T @ (P - cam_pos)
        depth = -zc
        if depth <= 1e-6:                       # target behind the camera
            break
        u, v = cx + f * xc / depth, cy - f * yc / depth
        du, dv = u - cx, v - cy
        off = math.hypot(du, dv)
        if best is None or off < best[0]:
            best = (off, ee.copy(), u, v, k + 1)
        if off < tol_px:
            break
        # A target at u > cx lies on the camera's +X side, so the camera moves +X to
        # centre it. v > cy means a NEGATIVE y_c (v = cy - f*y_c/depth), so it moves -Y.
        shift = (du / f * depth) * R_cw[:, 0] - (dv / f * depth) * R_cw[:, 1]
        step = gain * shift[:2]
        n = float(np.linalg.norm(step))
        if n > max_step_m:
            step = step * (max_step_m / n)
        ee[:2] += step

    if best is None:
        return False, {"stage": "aim_failed"}
    off, ee_best, u, v, iters_used = best
    controller.move_to_cartesian(float(ee_best[0]), float(ee_best[1]), float(ee_best[2]),
                                 lock_wrist=True, gain=0.5)
    in_frame = (0 <= u < CM.WRIST_W) and (0 <= v < CM.WRIST_H)
    return bool(in_frame), {"off_centre_px": round(float(off), 1),
                            "uv": [round(float(u), 1), round(float(v), 1)],
                            "iters": iters_used, "in_frame": bool(in_frame)}
