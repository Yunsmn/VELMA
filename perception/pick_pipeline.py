"""End-to-end perceived pick: coarse side view -> wrist triangulation -> grasp.

    find_object(prompt)  ->  coarse (x,y)      [one fixed view, table-plane aided]
    triangulate_object() ->  precise (x,y,z)   [wrist orbit, plane-free]
    grasp(trust_coords)  ->  the pick          [driven by the perceived coordinate]

The two stages are deliberately separate: the coarse one answers "where should I look?"
from a single wide view, and the fine one answers "where exactly do I close the jaws?"
from close range. Coarse alone is not enough — driving grasp straight off it bumps the
object and declines at ~2 cm.

HONESTY BOUNDARY (stated plainly). What is measured: the grasp coordinate (x,y,z) and the
jaw pre-open width. Nothing in THIS module, or in the wrist stage it calls, reads object
qpos — the descent is driven by the triangulated z.

What still reads ground truth, inside `controller.grasp`:
  * `backend.inspect_object()` — shape/width/height for the never-bat declines (sphere,
    short cylinder, wide box) and the tall-object grip height.
  * `_migration_mm()` — the object's true position, polled as the drift-abort guard that
    makes the grasp "never-bat".
Both are safety / "what is this" questions rather than "where is it", and on real hardware
they would need a perceptual or force-feedback equivalent. So: this pipeline is honest
about WHERE the object is; it is not yet honest about WHAT it is or whether it moved.
Do not describe it as a fully perceptual pick.
"""
from __future__ import annotations

from typing import Optional

from perception.find_object import find_object, _prompt_rgb
from perception.find_object_depth import find_object_depth
from perception.locate_object_3d import Locate3DResult
from perception import wrist_triangulate as WT

# Jaw pre-open = measured object width + this margin (the validated grasp recipe's
# clearance), clamped to the physical jaw range.
GRIP_MARGIN_MM = 10.0
JAW_MAX_MM = 103.0
MIN_CONFIDENCE = 0.6      # below this the triangulation is ill-conditioned -> decline


def locate_for_grasp(backend, controller, client, prompt: str,
                     hover_z: Optional[float] = None, orbit=WT.ORBIT,
                     coarse_fn=None) -> Locate3DResult:
    """Coarse find + wrist triangulation. Returns the fine result with the coarse
    estimate preserved in `extras['coarse']` for diagnostics.

    coarse_fn selects the coarse stage. The default `find_object_depth` measures a real
    height, so it works for objects that are not on the table AND lets the orbit pick a safe
    altitude; the older `find_object` is more accurate on a tabletop but assumes the object
    rests on the known plane, which is wrong by 58-73 mm for a raised object and often fails
    to detect one at all. Coarse accuracy barely matters here — the estimate only has to put
    the object inside the wrist camera's field of view, after which triangulation measures it.
    """
    if coarse_fn is None:
        coarse_fn = find_object_depth
    coarse = coarse_fn(backend, controller, client, prompt)
    if coarse.x is None:
        coarse.extras["stage"] = "coarse_failed"
        return coarse

    fine = WT.triangulate_object(backend, controller, client, coarse.x, coarse.y,
                                 z0=coarse.z, target_rgb=_prompt_rgb(prompt),
                                 hover_z=hover_z, orbit=orbit)
    fine.extras["coarse"] = {"x": round(coarse.x, 4), "y": round(coarse.y, 4),
                             "z": None if coarse.z is None else round(coarse.z, 4),
                             "method": coarse.method,
                             "confidence": coarse.confidence}
    return fine


def pick(backend, controller, client, prompt: str,
         min_confidence: float = MIN_CONFIDENCE,
         grip_width_mm: Optional[float] = None,
         hover_z: Optional[float] = None, orbit=WT.ORBIT, coarse_fn=None):
    """Locate `prompt` by perception and grasp it. Returns (MoveResult, Locate3DResult).

    Declines without touching the object if perception cannot place it confidently —
    an honest miss is better than a blind close at a guessed coordinate.

    hover_z None (default) derives the orbit altitude from the coarse height, so this works
    for objects that are not on the table; pass a value only to override it.
    """
    r = locate_for_grasp(backend, controller, client, prompt, hover_z=hover_z,
                         orbit=orbit, coarse_fn=coarse_fn)
    if r.x is None:
        from robot.controller import MoveResult
        return MoveResult(False, f"Perception could not locate '{prompt}' "
                                 f"({r.method}) — declining without touching it.",
                          backend.get_state()), r
    if r.confidence < min_confidence:
        from robot.controller import MoveResult
        return MoveResult(False, f"Perception located '{prompt}' but the views disagree "
                                 f"(confidence {r.confidence}, method {r.method}) — "
                                 f"declining rather than closing on an unreliable point.",
                          backend.get_state()), r

    if grip_width_mm is None:
        measured = r.extras.get("width_mm")
        if measured is not None:
            grip_width_mm = min(float(measured) + GRIP_MARGIN_MM, JAW_MAX_MM)

    res = controller.grasp(r.x, r.y, r.z, trust_coords=True, grip_width_mm=grip_width_mm)
    return res, r


def place(backend, controller, client, prompt: str,
          min_confidence: float = 0.2) -> tuple:
    """Locate container by coarse perception (open-vocab grounding only, no triangulation),
    then place the held object. Returns (MoveResult, Locate3DResult).

    For containers: coarse detection is sufficient. The opening inner half-width is 0.07 m,
    cube half-width is 0.015 m, so the drop zone tolerates ~55 mm lateral error. Coarse XY
    is good enough; Z is derived from controller's placed object physics, not perception
    (Depth-Anything reads ~10% short at container distance).

    No triangulation: for large objects with open top, the silhouette centroid is viewpoint-
    dependent (tens of mm), making 2-ray RANSAC unreliable. Coarse is enough.
    """
    coarse = find_object_depth(backend, controller, client, prompt)
    if coarse.x is None:
        from robot.controller import MoveResult
        return MoveResult(False, f"Perception could not locate '{prompt}' container "
                                 f"({coarse.method}) — declining to place.",
                          backend.get_state()), coarse
    if coarse.confidence < min_confidence:
        from robot.controller import MoveResult
        return MoveResult(False, f"Perception located '{prompt}' container but confidence "
                                 f"is low ({coarse.confidence}, method {coarse.method}) — "
                                 f"declining rather than risk a miss.",
                          backend.get_state()), coarse

    # Use coarse XY; place at a safe fixed drop height (don't chase perceived Z).
    # The controller handles the staged carry, drop from rim height, and settle.
    drop_z = 0.05  # Safe height above the container bottom for drop physics
    res = controller.place_object(coarse.x, coarse.y, drop_z)
    return res, coarse


def refine_container_place(backend, controller, client, prompt: str,
                           min_confidence: float = 0.2) -> tuple:
    """Locate and REFINE container using wrist-camera multi-view triangulation,
    then place the held object. Returns (MoveResult, Locate3DResult).

    FLOW:
      1. Coarse Falcon "container" detection (side camera, ~78mm error baseline)
      2. Move wrist camera to near-vertical over coarse position (16-18° tilt)
      3. Collect 4 views from different angles, detect container centroid in each
      4. Multi-view triangulation → refined XY (target: <55mm error)
      5. Place at refined (x, y), drop from safe fixed height

    HONESTY: The wrist-camera centroid is subject to silhouette variability, but
    multi-view from near-vertical stabilizes it; rays converge and residual is low.
    The refined XY is a real measurement, not a guess.
    """
    import numpy as np
    import wrist_refine
    from perception.locate_object_3d import _save_png
    import perception.camera_math as CM

    # Step 1: Coarse detection
    coarse = find_object_depth(backend, controller, client, prompt)
    if coarse.x is None:
        from robot.controller import MoveResult
        return MoveResult(False, f"Coarse detection failed at '{coarse.extras.get('stage')}'",
                          backend.get_state()), coarse
    if coarse.confidence < min_confidence:
        from robot.controller import MoveResult
        return MoveResult(False, f"Coarse confidence {coarse.confidence} < {min_confidence}",
                          backend.get_state()), coarse

    coarse_xy = np.array([coarse.x, coarse.y])

    # Step 2: Move wrist camera to near-vertical over coarse position
    target_ee = np.array([coarse.x, coarse.y, 0.16])  # High enough to hover safely
    for step in range(300):
        ctrl = controller.ik.step_toward_target(target_ee, gripper_action=0.5,
                                                gain=0.3, locked_joints=[3, 4])
        backend.apply_control(ctrl)
        backend.step()
        if step % 50 == 0:
            current = controller.ik.get_ee_position()
            if np.linalg.norm(current - target_ee) < 0.015:
                break

    # Step 3: Collect views from 4 angles around the coarse position
    rays = []
    n_views = 4
    radius = 0.06
    f_w, cx_w, cy_w = wrist_refine.wrist_intrinsics()

    for view_idx in range(n_views):
        angle_rad = 2 * np.pi * view_idx / n_views
        ox = radius * np.cos(angle_rad)
        oy = radius * np.sin(angle_rad)

        # Move to view position
        target_view = np.array([coarse.x + ox, coarse.y + oy, 0.16])
        for _ in range(150):
            ctrl = controller.ik.step_toward_target(target_view, gripper_action=0.5,
                                                    gain=0.5, locked_joints=[3, 4])
            backend.apply_control(ctrl)
            backend.step()

        # Render and detect
        img_wrist = backend.render_wrist()
        if img_wrist is None:
            continue

        png_wrist = _save_png(img_wrist, f"refine_container_view{view_idx}.png")
        ground_wrist = client.ground(png_wrist, prompt)

        if not ground_wrist or not ground_wrist.get('instances'):
            continue

        inst = ground_wrist['instances'][0]
        uv = np.array(inst.get('uv', []))

        # Build ray from wrist camera
        extrinsics = wrist_refine.wrist_extrinsics(backend)
        if not extrinsics:
            continue

        cam_pos, R_cw = extrinsics
        d_c = np.array([(uv[0] - cx_w) / f_w, -(uv[1] - cy_w) / f_w, -1.0])
        d_w = R_cw @ d_c
        d_w_norm = d_w / np.linalg.norm(d_w)
        rays.append((cam_pos.copy(), d_w_norm))

    # Step 4: Triangulate refined position
    if len(rays) < 2:
        # Fallback to coarse if not enough views
        refined_xy = coarse_xy
        residual = None
        converged = False
    else:
        try:
            p, resid, n_in, cond = CM.triangulate_nrays(rays, max_residual_m=0.050)
            if p is not None:
                refined_xy = np.array([p[0], p[1]])
                residual = resid
                converged = True
            else:
                refined_xy = coarse_xy
                residual = None
                converged = False
        except Exception:
            refined_xy = coarse_xy
            residual = None
            converged = False

    # Step 5: Place at refined (x, y)
    drop_z = 0.05
    res = controller.place_object(refined_xy[0], refined_xy[1], drop_z)

    # Augment the result with refinement metadata
    if converged:
        coarse.extras["refined"] = {
            "x": float(refined_xy[0]),
            "y": float(refined_xy[1]),
            "residual_m": float(residual) if residual else None,
            "n_views": len(rays),
        }

    return res, coarse
