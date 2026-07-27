"""Coarse localisation WITHOUT a table-plane assumption — open-vocab grounding +
monocular depth + robot self-anchors.

Replaces the floor-contact coarse stage. That stage found (x, y) by intersecting an object's
bottom edge with the plane z = 0, which is accurate on a tabletop and simply WRONG for anything
raised: measured 58-73 mm error for a cube standing on a 70 mm box, because the error is the
object's height divided by the viewing angle. It also could not detect such an object at all,
since its candidate filter demanded that masks deproject onto the table inside the reachable
envelope.

The key realisation is that the coarse stage does not need to be accurate. It only has to put
the object inside the wrist camera's field of view — about +/-15 cm at the 0.22 m orbit standoff
— after which triangulation measures the real position to ~1 mm. So a depth estimate good to a
few centimetres is more than sufficient, and unlike floor contact it holds at ANY height.

WHO PICKS THE OBJECT (2026-07-25, Falcon-Perception with FastSAM fallback). PRIMARY: Falcon-
Perception open-vocab grounding — takes the prompt PHRASE directly (e.g. "the red cube", "a small
yellow cylinder") and returns matching instances with masks, so the free-text query is the ONLY
object hint: no colour threshold, no class list, no object truth. This is strictly more honest and
more general than colour matching. FALLBACK: if Falcon fails/unavailable, falls back to FastSAM +
colour selection. Falcon inference fits on 4GB GPU with max_dim=448 tuned for this hardware.

HOW THE SCALE IS FIXED, without a table. Depth-Anything is relative (1/depth = a*pred + b), so
it needs anchors of known depth. Sampling the table plane for those would smuggle the assumption
back in. Instead we anchor on the ROBOT'S OWN LINKS: forward kinematics gives each link's exact
3D position, hence its exact depth from the calibrated camera. The links span 0.89-1.17 m and the
workspace sits inside that range, so the object's depth is interpolated, not extrapolated. This
is information the robot genuinely has about itself — no table, no object prior.

Measured against the old floor-contact stage (4 cells each):
    tabletop  depth 8-45 mm   vs floor-contact 18-19 mm
    RAISED    depth 8-67 mm   vs floor-contact 58-73 mm  (and often no detection at all)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import mujoco

from perception import camera_math as CM
from perception import reach as REACH
from perception.find_object import _wants_container
from perception.locate_object_3d import Locate3DResult, _save_png

_METHOD = "depth_fk_anchored"

# Robot links used as depth anchors — body ORIGINS projected through the known camera pose.
# Chosen to spread in depth (0.89-1.17 m) so the affine fit is conditioned across the
# workspace rather than extrapolating from one point.
ANCHOR_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
MIN_ANCHORS = 3

# Largest thing we will treat as an object, from its deprojected size. The arm's reachable
# diameter is ~0.75 m, so anything wider than this is scenery (table, wall, floor slab), never
# something the robot could pick or place into. Size-agnostic within that bound: a 30 mm cube and
# a 160 mm container both pass, which is the point.
MAX_OBJECT_M = 0.50


def fk_depth_anchors(backend, cam, W: int = CM.SIDE_W, H: int = CM.SIDE_H) -> list:
    """[[u, v, depth_m], ...] from the robot's own links. Pure self-knowledge."""
    f, cx, cy, cam_pos, R_cw = cam
    backend._mj_forward()
    anchors = []
    for name in ANCHOR_BODIES:
        bid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        P = np.asarray(backend.data.xpos[bid], float)
        xc, yc, zc = R_cw.T @ (P - cam_pos)
        depth = -zc
        if depth <= 0:
            continue
        u, v = cx + f * xc / depth, cy - f * yc / depth
        if 2 <= u < W - 2 and 2 <= v < H - 2:
            anchors.append([float(u), float(v), float(depth)])
    return anchors


def _apparent_size_m(area_px: float, depth_m: float, f: float) -> float:
    """Rough physical width of a mask, from its area. sqrt(area) rather than a bbox span
    because a square's bounding box grows 41% under in-plane rotation."""
    return math.sqrt(max(area_px, 1.0)) * depth_m / f


def find_object_depth(backend, controller, client, prompt: str) -> Locate3DResult:
    """Coarse (x, y, z) of `prompt` from the fixed side camera, with NO plane assumption.

    Every returned coordinate — including z — comes from a depth reading, so the result is
    valid for objects at any height. Accuracy is centimetre-scale by design; feed it to
    `wrist_triangulate.triangulate_object` for the millimetre answer.

    Object selection is entirely `prompt`-driven (Falcon-Perception open-vocab grounding):
    the phrase is the only signal used anywhere below, so a mis-worded or unusual query
    fails honestly (no_candidates / no reachable instance) rather than falling back to a
    guess.
    """
    extras: dict = {"prompt": prompt, "method": _METHOD}
    cam = CM.side_cam_params()
    f, cx, cy, cam_pos, R_cw = cam

    img = backend.render_side()
    if img is None:
        return Locate3DResult(None, None, None, 0.0, _METHOD, {**extras, "stage": "render_fail"})
    png = _save_png(img, "find_depth_side.png")

    # 1. Try Falcon-Perception open-vocab grounding (PRIMARY), fall back to FastSAM+colour.
    ground = client.ground(png, prompt)
    if ground is not None:
        # Falcon succeeded: use open-vocab grounding
        cands = ground.get("instances") or []
        extras["backend"] = ground.get("backend")
        if not cands:
            # Try fallback
            ground = None

    if ground is None:
        # Fallback: FastSAM with colour selection
        from perception.find_object import _prompt_rgb
        target_rgb = _prompt_rgb(prompt)
        extras["target_rgb"] = target_rgb
        if target_rgb is None:
            extras["note"] = "no colour in prompt -> consensus object pick"

        r = client.detect(png, (f, cx, cy, cam_pos, R_cw), view="side", target_rgb=target_rgb)
        if r is None:
            return Locate3DResult(None, None, None, 0.0, _METHOD,
                                  {**extras, "stage": "detect_failed"})

        cands = [r]  # Adapt detect result to candidate list format
        extras["backend"] = "fastsam_colour"

    # 2. Get depths for all candidates, scaled by robot-link anchors.
    anchors = fk_depth_anchors(backend, cam)
    extras["n_anchors"] = len(anchors)
    if len(anchors) < MIN_ANCHORS:
        return Locate3DResult(None, None, None, 0.0, _METHOD,
                              {**extras, "stage": "too_few_anchors"})

    uvs = [c["uv"] for c in cands]
    depths = client.depth_many(png, uvs, anchors)
    if depths is None:
        return Locate3DResult(None, None, None, 0.0, _METHOD,
                              {**extras, "stage": "depth_failed"})

    # 3. Deproject and keep what the arm can reach.
    az_max = REACH.pan_limit_deg(backend.model)
    extras["az_max_deg"] = round(az_max, 1)
    scored, n_unreachable, n_too_big = [], 0, 0
    for c, d in zip(cands, depths):
        if d is None or d <= 0:
            continue
        P = CM.point_at_depth(c["uv"][0], c["uv"][1], f, cx, cy, cam_pos, R_cw, d)
        if not REACH.reachable(P, az_max_deg=az_max):
            n_unreachable += 1
            continue
        if _apparent_size_m(c.get("area", 0), d, f) > MAX_OBJECT_M:
            n_too_big += 1
            continue
        scored.append((c, P))

    extras["n_candidates"] = len(cands)
    extras["n_reachable"] = len(scored)
    extras["n_unreachable"] = n_unreachable
    extras["n_too_big"] = n_too_big
    if not scored:
        return Locate3DResult(None, None, None, 0.0, _METHOD,
                              {**extras, "stage": "none_reachable"})

    # 4. Pick the largest reachable instance (candidates are sorted by area desc).
    c, P = scored[0]
    extras.update({
        "uv": c["uv"],
        "area": c.get("area"),
        "fill": c.get("fill"),
        "kind": "container" if _wants_container(prompt) else "graspable",
    })

    # Confidence: modest since this stage only aims the camera, wrist measures precisely.
    # More candidates = more ambiguity -> lower confidence.
    conf = max(0.2, 0.85 - 0.15 * (len(scored) - 1))

    return Locate3DResult(float(P[0]), float(P[1]), float(P[2]),
                          round(conf, 3), _METHOD, extras)
