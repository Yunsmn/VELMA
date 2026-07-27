"""Stage 2: PLANE-FREE grasp point from the wrist camera by N-ray triangulation.

The coarse stage (`find_object`) locates the object from one fixed side view. Its (x,y)
leans on the table plane and its z is a placeholder — good enough to aim the arm, not good
enough to grasp (validated: a coarse-only grasp bumps and declines at ~2 cm).

This stage removes both priors. The wrist camera orbits the coarse point at a fixed height,
taking a frame at each pose. Every frame's mask centroid back-projects to a world ray whose
origin and direction come from the robot's OWN forward kinematics (`cam_xpos`/`cam_xmat` =
hand-eye on real hardware). Intersecting those rays recovers (x, y, z) from geometry alone —
no support plane, no assumed object height, no depth model. Redundant rays let
`triangulate_nrays` DROP a mis-fired centroid instead of averaging it in, and the residual
between the surviving rays is a ground-truth-free confidence signal.

WHAT THE POINT MEANS: near-top-down silhouettes all centre on the object's visible TOP
surface, so the rays converge on the top-face centre, not the body centre. That is the more
useful quantity for a top grasp — it is where the jaws must descend, and its z is the
object's MEASURED top height (what `grasp(grasp_height_m=...)` otherwise has to assume).

STATE-NEUTRAL: full physics is snapshotted on entry and restored on exit, so this is a pure
perception upgrade — the caller's grasp starts from byte-identical kinematics.

Nothing here reads object qpos.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import wrist_refine as WR
from perception import camera_math as CM
from perception.detector import COLOR_MAX_RGB, COLOR_TIE_RGB
from perception.locate_object_3d import Locate3DResult, _save_png

_METHOD = "wrist_triangulation"

# ── Orbit geometry ────────────────────────────────────────────────────────────────
# The wrist camera sits ~0.11 m BEHIND the end-effector, so an EE hover at HOVER_Z puts
# the lens ~0.22 m above a tabletop object. A lateral orbit of ORBIT_R then opens
# ~atan(R/standoff) of parallax per view (~22° at R=0.09), i.e. ~44° between opposed
# views — well-conditioned for triangulation from a single moving sensor. Larger R is
# better geometry but risks the object leaving the 70.5° frame and the pose leaving the
# arm's comfortable reach.
HOVER_Z = 0.14
# Clearance held above the OBJECT when a coarse height is known. 0.125 m reproduces the
# validated tabletop geometry (end-effector 0.14 over an object centred at 0.015) and carries
# it to any height, instead of hardcoding an altitude that only suits the table.
HOVER_CLEARANCE = 0.125
ORBIT_R = 0.10
ORBIT = [(0.0, 0.0), (ORBIT_R, 0.0), (-ORBIT_R, 0.0), (0.0, ORBIT_R), (0.0, -ORBIT_R)]

# GATE_Z (was 0.015) is deliberately GONE. It was the centre height of a 30 mm cube resting
# on the table, used as a "fallback plane" whenever no coarse height was available — i.e. it
# silently asserted that an unmeasured object is cube-sized and sitting on the table, which is
# exactly the assumption this stage exists to remove. There is no honest fallback height, so
# there is none: without a measured z there is no 3D aim point, the first frame is
# disambiguated by colour alone, and from the second frame the running triangulation supplies
# a real 3D aim.
# Two DIFFERENT thresholds, deliberately: RANSAC_DROP_M is loose so only a genuinely
# mis-fired centroid is discarded (pruning at the accept threshold threw away good rays
# and cost us redundancy), while MAX_RESID_M is the tight accept gate on the mean
# residual of whatever survived.
RANSAC_DROP_M = 0.004
MAX_RESID_M = 0.0025      # mean inter-ray disagreement budget (honest confidence signal)
MIN_COND = 0.02           # smallest-eigenvalue/N floor: rays too parallel to trust
MIN_RAYS = 2

# Close-range mask gates: at ~0.22 m a 30 mm object is ~45 px (~2 kpx), and a near-top-down
# silhouette fills its bbox less than a side-on one.
# in_workspace is OFF: that test deprojects each mask onto a table plane and demands it land
# in the reachable envelope, which silently discards anything NOT resting on the table — the
# same assumption this stage exists to remove. Selection is instead by colour plus proximity
# to the running estimate, neither of which cares how high the object is.
WRIST_GATES = {"area_max_frac": 0.25, "area_min_px": 300, "min_fill": 0.55,
               "in_workspace": False}


def _min_area_rect_angle_deg(rows: np.ndarray, cols: np.ndarray) -> Optional[float]:
    """Rotation (image-plane degrees, in [0, 90)) that minimises the axis-aligned
    bounding-box area of a 2D point set — the standard minimum-area-rectangle
    heuristic for recovering a rotated rectangle's true edge direction.

    NOT PCA/second-moment orientation: a filled SQUARE's second-moment tensor is
    isotropic (equal eigenvalues in every basis), so PCA has no signal to lock
    onto for the shape this pipeline cares about most. The bounding-box-area
    criterion still works — it is minimised exactly at the true edge angle for
    any rectangle, square included. Coarse 1-degree brute force over the point
    set is cheap at silhouette scale (tens to a couple thousand px)."""
    if rows.size < 20:
        return None
    x = cols.astype(float) - float(cols.mean())
    y = rows.astype(float) - float(rows.mean())
    best_deg, best_area = 0.0, math.inf
    for deg in range(0, 90):
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        rx = x * c + y * s
        ry = -x * s + y * c
        area = float((rx.max() - rx.min()) * (ry.max() - ry.min()))
        if area < best_area:
            best_area, best_deg = area, float(deg)
    return best_deg


def _mask_edge_angle_deg(img: np.ndarray, bbox, rgb_ref, pad: int = 4) -> Optional[float]:
    """Minimum-area-rectangle angle of the object's silhouette within its own
    detected bbox (+pad), image-plane degrees in [0, 90).

    Self-contained (no sidecar round-trip): thresholds the ALREADY-RENDERED wrist
    frame against the candidate's own measured mean colour (`rgb`, returned by the
    real detector alongside the bbox we are cropping to), so this only looks where
    the real FastSAM detection already said the object is — low risk of picking up
    arm/background the way a full-frame colour threshold would (measured
    elsewhere: "yellow cylinder" caught the yellow ARM on an unconstrained scan).
    """
    if bbox is None or rgb_ref is None:
        return None
    x0, y0, x1, y1 = bbox
    H, W = img.shape[:2]
    x0 = max(0, int(x0) - pad); y0 = max(0, int(y0) - pad)
    x1 = min(W, int(x1) + pad); y1 = min(H, int(y1) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = img[y0:y1, x0:x1].astype(float)
    ref = np.asarray(rgb_ref, float)
    dist = np.linalg.norm(crop - ref, axis=-1)
    mask = dist < max(30.0, COLOR_MAX_RGB * 0.6)
    rows, cols = np.nonzero(mask)
    return _min_area_rect_angle_deg(rows, cols)


def _project(P, f, cx, cy, cam_pos, R_cw):
    """World point -> pixel in the current wrist view (None if behind the camera)."""
    xc, yc, zc = R_cw.T @ (np.asarray(P, float) - np.asarray(cam_pos, float))
    depth = -zc
    if depth <= 1e-6:
        return None
    return cx + f * xc / depth, cy - f * yc / depth


def _view_ray(backend, client, png_name: str, aim_xyz, target_rgb, gates):
    """Render the wrist camera at the CURRENT pose and back-project the object centroid.

    Returns (view, info) where view carries the world ray plus the pose and silhouette
    bbox needed for the metric extent estimate, or (None, info) if the frame could not
    be rendered or nothing was detected.
    """
    img = backend.render_wrist()
    if img is None:
        return None, {"stage": "render_fail"}
    ext = WR.wrist_extrinsics(backend)
    if ext is None:
        return None, {"stage": "no_extrinsics"}
    cam_pos, R_cw = ext
    f, cx, cy = CM.wrist_intrinsics()
    # Selection is PIXEL-space: colour narrows to the right object, then we take the
    # candidate closest to where the running 3D estimate PROJECTS into this view. No plane
    # and no height enters the choice — projecting a 3D point through a known camera pose is
    # exact at any altitude, whereas the old "deproject every candidate onto a plane at
    # height z and compare" both reintroduced a plane and made the estimate's z influence
    # which blob was picked.
    # No z_plane is passed: with in_workspace off and near_xy unused, the detector's plane
    # deprojection feeds nothing this function reads, so supplying a height here would be a
    # dead assumption sitting in the call signature waiting to be believed.
    det = client.detect(_save_png(img, png_name), (f, cx, cy, cam_pos, R_cw),
                        view="wrist", near_xy=None,
                        target_rgb=target_rgb, gates=gates, debug=True)
    cands = (det or {}).get("candidates") or []
    if not cands:
        return None, {"stage": "no_detection",
                      "cam_z": round(float(cam_pos[2]), 4)}

    # COLOUR FIRST, then proximity. `candidates` in debug mode is every mask that passed the
    # geometric gates, of ANY colour — selecting purely by distance to the aim point lets a
    # green sphere win simply by sitting nearer the projected estimate (measured: 85 mm error
    # on a cell that otherwise reads 1.4 mm). Colour narrows to the right OBJECT; proximity
    # then picks the right INSTANCE.
    if target_rgb is not None:
        t = np.asarray(target_rgb, float)
        def _cd(c):
            return float(np.linalg.norm(np.asarray(c["rgb"], float) - t))
        cands = sorted(cands, key=_cd)
        best_d = _cd(cands[0])
        if best_d > COLOR_MAX_RGB:
            return None, {"stage": "no_colour_match", "color_dist": round(best_d, 1)}
        cands = [c for c in cands if _cd(c) < best_d + COLOR_TIE_RGB]

    aim_px = None if aim_xyz is None else _project(aim_xyz, f, cx, cy, cam_pos, R_cw)
    if aim_px is not None and len(cands) > 1:
        pick = min(cands, key=lambda c: (c["uv"][0] - aim_px[0]) ** 2
                   + (c["uv"][1] - aim_px[1]) ** 2)
    else:
        pick = cands[0]
    det = {**det, **pick}
    u, v = det["uv"]
    info = {"uv": [round(u, 1), round(v, 1)], "area": det.get("area"),
            "fill": None if det.get("fill") is None else round(det["fill"], 2),
            "color_dist": det.get("color_dist"),
            "cam_pos": [round(float(c), 4) for c in cam_pos]}
    o, d = CM.back_project(u, v, f, cx, cy, cam_pos, R_cw)
    edge_deg = _mask_edge_angle_deg(img, det.get("bbox"), det.get("rgb"))
    view = {"ray": (o, d), "cam_pos": cam_pos, "R_cw": R_cw,
            "bbox": det.get("bbox"), "area": det.get("area"), "f": f,
            "edge_angle_deg": edge_deg}
    return view, info


def measure_extent(views, point: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Metric width (mm) of the object's silhouette, DEBIASED for each view's known
    obliquity instead of picking a single "least-bad" view.

    THE OLD APPROACH (nadir-only, then "2nd-smallest of >=4 views") wasn't good enough:
    even the "nadir" orbit pose isn't exactly overhead (the wrist lens sits ~0.11 m
    BEHIND the end-effector, so every pose has some tilt from vertical), and picking a
    single order statistic across views threw away the information that tells us HOW
    contaminated each reading is. Measured on the real deploy grid: a 30 mm cube still
    read 37-41 mm at several front-zone poses even with the 2nd-smallest rule.

    THE FIX uses geometry we already have for free. sqrt(area_px)*depth/f converts a
    view's pixel silhouette into a real-world "shadow area" (the area of the mask's
    footprint in the image plane, at the object's known distance). For a box of true
    width w and height h, the classic convex-polyhedron projected-shadow-area identity
    says that shadow area, as a function of the view direction's angle theta from
    straight-down, is (to the isotropic approximation that drops the object's unknown
    yaw):
        A(theta) ~= w^2 * cos(theta)  +  w*h * sin(theta)
                     ^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^
                     top face, foreshortened   visible side-face strip, which is what
                     by the tilt (0 at theta=0)  makes an oblique view over-read w

    theta is NOT assumed — it comes straight from the per-view FK camera pose (the
    already-triangulated 3D point minus the camera position, both honest quantities),
    so nothing here is guessed or hand-tuned per shape. With >=2 views at different
    theta — guaranteed by construction, since the same theta spread is exactly what
    let triangulation succeed — this is a 2-unknown (X=w^2, Y=w*h) LINEAR least-squares
    fit, solved directly for width = sqrt(X). Unlike a "nadir-only" or "assume h~=w"
    shortcut, this does not bias wide-short boxes (small h) or narrow-tall ones
    (large h) toward the wrong side of the never-bat width thresholds, because h is
    fit from the data rather than assumed.
    """
    samples: list[tuple[float, float, float]] = []  # (cos_theta, sin_theta, area_m2)
    for v in views:
        if not v.get("area"):
            continue
        cam_pos = np.asarray(v["cam_pos"], float)
        fwd = -v["R_cw"][:, 2]
        depth = float((point - cam_pos) @ fwd)
        if depth <= 0:
            continue
        dirvec = np.asarray(point, float) - cam_pos
        dist = float(np.linalg.norm(dirvec))
        if dist <= 1e-6:
            continue
        # theta = angle between this view's camera->object ray and straight-down.
        # cos(theta)=1 at true nadir; grows toward 0 as the view tilts to the side.
        cos_theta = float(np.clip(-dirvec[2] / dist, 0.0, 1.0))
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        area_m2 = float(v["area"]) * (depth / v["f"]) ** 2
        samples.append((cos_theta, sin_theta, area_m2))
    if not samples:
        return None, None

    cos_thetas = [c for c, _, _ in samples]
    theta_spread = max(cos_thetas) - min(cos_thetas)
    # theta_spread in cos-space; ~0.02 is a couple of degrees at near-nadir — below
    # that the two unknowns (top-face term, side-face term) are too close to
    # collinear to separate safely, so fall back rather than trust an ill-conditioned
    # solve. The orbit's lateral offset (ORBIT_R) is what normally guarantees spread.
    if len(samples) >= 2 and theta_spread > 0.02:
        M = np.array([[c, s] for c, s, _ in samples])
        b = np.array([a for _, _, a in samples])
        (X, Y), _res, rank, _sv = np.linalg.lstsq(M, b, rcond=None)
        if rank >= 2 and X > 0:
            width_m = math.sqrt(float(X))
            # Per-view spread under the FITTED model (solve the same quadratic per
            # view using the shared Y) gives an honest uncertainty: it is small when
            # the views agree with the fit and grows when one view disagrees, the
            # same role std(widths) played for the old order-statistic estimate.
            per_view_w = []
            for cos_theta, sin_theta, area_m2 in samples:
                disc = (Y * sin_theta) ** 2 + 4.0 * cos_theta * area_m2
                if disc < 0 or cos_theta <= 1e-6:
                    continue
                w_v = (-Y * sin_theta + math.sqrt(disc)) / (2.0 * cos_theta)
                if w_v > 0:
                    per_view_w.append(w_v)
            unc = (max(3.0, float(np.std(per_view_w)) * 1000.0)
                   if len(per_view_w) >= 2 else 3.0)
            return width_m * 1000.0, unc

    # Fallback (a single usable view, or the orbit gave almost no theta spread to fit
    # from — rare, but must degrade gracefully rather than raise). Without a second
    # independent equation h cannot be separated from w, so this assumes h~=w (only
    # defensible when there's no data to do better) and, as before, prefers the
    # least-contaminated (smallest) reading since the raw over-read is one-sided.
    widths = []
    for cos_theta, sin_theta, area_m2 in samples:
        denom = max(1e-6, cos_theta + sin_theta)
        widths.append(math.sqrt(area_m2 / denom) * 1000.0)
    widths.sort()
    width = widths[1] if len(widths) >= 4 else widths[0]
    unc = max(3.0, float(np.std(widths)))
    return width, unc


# Minimum cos(theta) (theta = angle off straight-down) a view must clear before its
# edge angle is trusted for yaw. Below ~45 deg off nadir the silhouette increasingly
# shows side face rather than the top face's true outline (the same effect that
# drives the width over-read), so a badly oblique "edge" would misdirect wrist_roll
# instead of correcting it.
MIN_NADIR_COS_FOR_YAW = 0.7


def estimate_face_yaw_deg(views, point: np.ndarray) -> Optional[float]:
    """Object yaw about world z (degrees, wrapped to [-45, 45) so it always names
    the offset to the NEAREST face), from the wrist silhouette's own edge
    direction in the LEAST OBLIQUE available view.

    ADOPTED IDEA (QuickGrasp, arXiv 2504.19716 — lightweight analytical parallel-
    jaw antipodal grasp planning): choose the approach from the object's own
    geometry (its principal axis / face normal) rather than assuming a fixed
    world-frame pose. The existing wrist_roll formula (`90+pan` / `1.24*pan+92`,
    see controller._grasp_wrist_roll) silently assumes the object's top face is
    WORLD-AXIS-ALIGNED — true only because the validated benchmark scenes never
    randomize object yaw (a latent bug: a rotated object would be gripped off-
    face). This recovers the object's ACTUAL in-plane rotation instead.

    Only the LEAST OBLIQUE view is used. Its image plane is closest to parallel
    with the object's own top face, so `_mask_edge_angle_deg`'s reading maps to
    world yaw with the least distortion; a badly oblique view's apparent "edge" is
    contaminated by the visible side face — the same foreshortening that drives
    the width over-read `measure_extent` exists to correct, not a second
    independent measurement to average in. Below MIN_NADIR_COS_FOR_YAW the
    reading is not trusted and this returns None (caller keeps the axis-aligned
    default rather than act on a misleading angle).
    """
    best = None  # (cos_theta, view)
    for v in views:
        if v.get("edge_angle_deg") is None:
            continue
        cam_pos = np.asarray(v["cam_pos"], float)
        dirvec = np.asarray(point, float) - cam_pos
        dist = float(np.linalg.norm(dirvec))
        if dist <= 1e-6:
            continue
        cos_theta = float(np.clip(-dirvec[2] / dist, 0.0, 1.0))
        if best is None or cos_theta > best[0]:
            best = (cos_theta, v)
    if best is None or best[0] < MIN_NADIR_COS_FOR_YAW:
        return None
    _, v = best
    rad = math.radians(v["edge_angle_deg"])
    R_cw = v["R_cw"]
    # Image-plane basis expressed in world: +u (right) is the camera's local +x
    # axis; +v (down) is the camera's local -y axis (pixel v grows as image-yc
    # SHRINKS in the projection `_project` uses — cy - f*yc/depth).
    e_u = R_cw[:, 0]
    e_v = -R_cw[:, 1]
    edge_world = math.cos(rad) * e_u + math.sin(rad) * e_v
    dx, dy = float(edge_world[0]), float(edge_world[1])
    if math.hypot(dx, dy) < 1e-6:
        return None
    yaw = math.degrees(math.atan2(dy, dx))
    return ((yaw + 45.0) % 90.0) - 45.0


def triangulate_object(backend, controller, client,
                       x0: float, y0: float,
                       z0: Optional[float] = None,
                       target_rgb: Optional[tuple] = None,
                       hover_z: Optional[float] = None,
                       orbit=ORBIT,
                       gates: Optional[dict] = None,
                       max_resid_m: float = MAX_RESID_M,
                       min_cond: float = MIN_COND,
                       restore: bool = True) -> Locate3DResult:
    """Plane-free (x, y, z) of the object near the coarse seed (x0, y0).

    x0, y0: coarse estimate from `find_object` — used ONLY to aim the camera and to
            disambiguate which mask is the target. It does not enter the answer.
    z0:     coarse HEIGHT estimate. MUST be a perceived value (find_object_depth), never a
            true height — it is a seed, and feeding it ground truth would make the stage
            look better than it is.

            It has exactly two uses, and NEITHER can reach the returned coordinate:
              1. orbit altitude — keeps the validated clearance above the OBJECT rather than
                 above the table, so the gripper does not fly into anything standing taller
                 (a cube on a 70 mm box sits at z=0.085, while the fingers hang below an
                 end-effector at 0.14). This is a motion decision.
              2. the initial 3D aim point, projected into the first frame to pick which blob
                 is the target. After two rays it is replaced by the running triangulation.
            The output comes from intersecting rays built out of pixels and FK poses. z0 is
            not an input to that. An earlier version also fed z0 to the detector as a
            deprojection plane, which DID let it influence mask selection through a
            reintroduced plane; that path is gone — selection is now pixel-space.
    target_rgb: which object to keep when several are in frame (colour selects among
            clean SAM object-masks; it never thresholds the image).

    Returns Locate3DResult; `confidence` is derived from the inter-ray residual, so a
    mis-detection degrades the confidence instead of silently corrupting the coordinate.
    """
    # Orbit altitude: keep the validated clearance ABOVE the object rather than a fixed
    # height above the table, so a raised object neither gets struck nor drifts out of frame.
    if hover_z is None:
        hover_z = HOVER_Z if z0 is None else max(HOVER_Z, float(z0) + HOVER_CLEARANCE)
    gates = dict(WRIST_GATES if gates is None else gates)
    extras: dict = {"seed": [round(x0, 4), round(y0, 4)], "hover_z": round(hover_z, 4),
                    "views": [], "n_rays": 0}
    rays: list[tuple[np.ndarray, np.ndarray]] = []
    views: list[dict] = []
    # 3D aim point, projected into each view to disambiguate which blob is the target.
    # Seeded from the coarse estimate (all three components PERCEIVED, never ground truth);
    # once two rays exist it is replaced by the running triangulation. It only ever selects a
    # mask — it cannot enter the returned coordinate, which comes from ray intersection.
    aim = None if z0 is None else np.array([x0, y0, float(z0)], float)

    snap = WR.save_state(backend)
    try:
        for k, (dx, dy) in enumerate(orbit):
            # Orbit around the BEST CURRENT estimate, not the original seed. Every pose used
            # to be computed from (x0, y0), so a biased seed mis-centred the whole orbit and
            # stayed mis-centred no matter what the rays said. Measured on raised objects,
            # where the coarse stage is systematically ~60 mm out: the object drifted to the
            # frame edge and 7 of 15 views returned no detection at all. Re-centring lets the
            # orbit correct itself as evidence arrives — the first ray or two pull it in, and
            # the remaining views are taken over the object rather than beside it.
            centre = (x0, y0) if aim is None else (float(aim[0]), float(aim[1]))
            tx, ty = centre[0] + dx, centre[1] + dy
            # Altitude tracks the estimate for the same reason: it was fixed from the coarse
            # z, which over-read by ~50 mm on raised objects and flew the orbit that much too
            # high. Once rays exist, hold the clearance above the TRIANGULATED height.
            hover_now = (hover_z if aim is None or len(rays) < MIN_RAYS
                         else max(HOVER_Z, float(aim[2]) + HOVER_CLEARANCE))
            if k == 0:
                # First pose: rehome high and descend from above, so the forearm never
                # sweeps across the object on the way in.
                WR.hover_top_down(controller, tx, ty, hover_z=hover_z)
            else:
                # Subsequent poses: translate laterally at height. Cheaper than a rehome
                # per view and it keeps the arm well clear of the tabletop.
                ee = controller.ik.get_ee_position()
                controller.servo_relative(dx=tx - float(ee[0]), dy=ty - float(ee[1]),
                                          dz=hover_now - float(ee[2]), lock_wrist=True)
            view, info = _view_ray(backend, client, f"wtri_{k}.png", aim, target_rgb, gates)
            info["target"] = [round(tx, 3), round(ty, 3)]
            extras["views"].append(info)
            if view is None:
                continue
            views.append(view)
            rays.append(view["ray"])
            # Re-aim at the running estimate: each new ray sharpens where to expect the
            # object in the NEXT frame, which tightens the near_xy disambiguation.
            if len(rays) >= 2:
                p, _, _, _ = CM.triangulate_nrays(rays, max_residual_m=RANSAC_DROP_M)
                if p is not None:
                    aim = np.asarray(p, float)   # full 3D, so the projection is exact
    finally:
        # restore=True (default): teleport the arm back to its pre-orbit pose so a
        # coarse-vs-wrist A/B benchmark scores against an identical scene (state-neutral).
        # restore=False (DEPLOYMENT): a real arm cannot teleport, and snapping qpos back is
        # the jerky "return to origin" seen in the viewer. Leave the arm where the orbit
        # ended — that is a hover ABOVE the object, which is exactly where grasp descends
        # from, so the return trip is saved rather than paid.
        if restore:
            WR.restore_state(backend, snap)

    extras["n_rays"] = len(rays)
    if len(rays) < MIN_RAYS:
        return Locate3DResult(None, None, None, 0.0, "insufficient_views", extras)

    p, resid, n_in, cond = CM.triangulate_nrays(rays, max_residual_m=RANSAC_DROP_M,
                                                min_rays=MIN_RAYS)
    if p is None:
        return Locate3DResult(None, None, None, 0.0, "tri_singular", extras)

    width_mm, width_unc_mm = measure_extent(views, p)
    yaw_deg = estimate_face_yaw_deg(views, p)
    extras.update({"tri_xyz": [round(float(c), 4) for c in p],
                   "tri_resid_mm": round(resid * 1000, 2),
                   "tri_n_inliers": n_in,
                   "tri_cond": round(cond, 3),
                   "width_mm": None if width_mm is None else round(width_mm, 1),
                   "width_uncertainty_mm": None if width_unc_mm is None else round(width_unc_mm, 1),
                   "object_yaw_deg": None if yaw_deg is None else round(yaw_deg, 1),
                   "seed_shift_mm": round(math.hypot(float(p[0]) - x0,
                                                     float(p[1]) - y0) * 1000, 1)})

    if n_in < MIN_RAYS or cond < min_cond or resid > max_resid_m:
        # Honest decline path: return the point but flag it as untrustworthy rather than
        # dressing up a bad intersection as a grasp target.
        return Locate3DResult(float(p[0]), float(p[1]), float(p[2]), 0.3,
                              "triangulation_illcond", extras)

    conf = 0.6 + 0.4 * max(0.0, 1.0 - resid / max_resid_m)
    return Locate3DResult(float(p[0]), float(p[1]), float(p[2]),
                          round(conf, 3), _METHOD, extras)
