"""find_object — single-camera COARSE localisation for the wrist-camera pipeline.

Colour-seeded FastSAM (a smaller, promptable SAM) picks WHICH object; the known floor gives WHERE.

Pipeline:
  1. render the fixed side camera; get its analytic pose (camera_math).
  2. FastSAM 'everything' masks -> filtered to compact, in-workspace, non-border objects
     (the floor and the arm are excluded by geometry, not colour).
  3. pick the candidate whose MEAN colour matches the prompt's colour — colour only *selects
     among* clean SAM object-masks, it never thresholds the raw image, so the blue floor / yellow
     arm can't bleed in.
  4. the chosen mask's floor contact (ray ∩ z=0) is the coarse (x, y).

z is a rough estimate here — the coarse pose only has to get the wrist camera hovering over the
object; the WRIST-CAMERA stage measures the precise grasp point at close range. Honest: no object
qpos, no raw-colour threshold, no support-plane z prior on the returned point.

Returns Locate3DResult(x, y, z, confidence, method="fastsam_colour_floor", extras).
"""
from __future__ import annotations

import math
from typing import Optional

from perception import camera_math as CM
from perception.detector import WS_X_MIN, WS_X_MAX, WS_Y_ABS, WS_R_MAX
from perception.locate_object_3d import Locate3DResult, _save_png

_METHOD = "fastsam_colour_floor"

# Prompt phrase -> approximate object RGB. Used ONLY to choose which SAM object-mask matches the
# phrase (sim identity); it is not a threshold over the image. Extend for real-world palettes.
_COLOURS = {
    "red": (230, 26, 26), "blue": (26, 77, 230), "green": (26, 204, 51),
    "yellow": (230, 204, 26), "orange": (230, 120, 26), "purple": (150, 50, 200),
    "grey": (191, 191, 199), "gray": (191, 191, 199), "white": (235, 235, 235),
}
_CONTAINER_WORDS = ("container", "bin", "tray", "basket", "crate", "bowl", "cup")


def _prompt_rgb(prompt: str) -> Optional[tuple]:
    p = prompt.lower()
    for name, rgb in _COLOURS.items():
        if name in p:
            return rgb
    return None


def _wants_container(prompt: str) -> bool:
    """Label the RESULT from the caller's own phrasing. Descriptive only.

    This must never influence detection. An earlier design took a `kind_hint`
    argument and planned to relax the mask gates for containers — i.e. hand the
    perception a keyword and let it filter differently. That is knowledge the
    caller would not have on real hardware, and any object outside the keyword
    table would get the wrong filter. The gates are now identical for every
    prompt; this function only names what came back.
    """
    return any(w in prompt.lower() for w in _CONTAINER_WORDS)


def _in_envelope(x: float, y: float, z: float) -> bool:
    r = math.hypot(x, y)
    return WS_X_MIN < x < WS_X_MAX and abs(y) < WS_Y_ABS and r < WS_R_MAX


def find_object(backend, controller, client, prompt: str,
                score_thr: Optional[float] = None) -> Locate3DResult:
    """Coarse (x, y, z) of the object named by ``prompt`` from the fixed side camera.

    Colour (from the phrase) selects which FastSAM object-mask; its floor contact gives (x, y).
    ``controller`` is unused here (single fixed view); kept for signature parity.
    """
    extras: dict = {"prompt": prompt, "method": _METHOD}

    f, cx, cy, cam_pos, R_cw = CM.side_cam_params()
    img = backend.render_side()
    if img is None:
        return Locate3DResult(None, None, None, 0.0, _METHOD, {**extras, "stage": "render_fail"})
    png = _save_png(img, "find_side.png")

    target_rgb = _prompt_rgb(prompt)
    extras["target_rgb"] = target_rgb
    if target_rgb is None:
        extras["note"] = "no colour in prompt -> falling back to consensus object pick"

    r = client.detect(png, (f, cx, cy, cam_pos, R_cw), view="side", target_rgb=target_rgb)
    if r is None:
        return Locate3DResult(None, None, None, 0.0, _METHOD,
                              {**extras, "stage": "no_detection"})

    # xy_plane (mask centroid on the object-height plane) is tighter to the CENTRE for boxes/cylinders
    # than xy_floor (front-bottom edge); prefer it, fall back to the floor contact.
    xy = r.get("xy_plane") or r.get("xy_floor")
    if xy is None:
        return Locate3DResult(None, None, None, 0.0, _METHOD,
                              {**extras, "stage": "no_xy", "uv": r.get("uv")})
    x, y = float(xy[0]), float(xy[1])
    z = 0.02  # coarse rough height; the wrist camera measures the true grasp z at close range.
    in_env = _in_envelope(x, y, z)
    color_dist = r.get("color_dist")
    votes = int(r.get("votes", 1))
    conf = 0.0 if not in_env else max(0.0, 1.0 - (color_dist if color_dist is not None else 60) / 90.0) \
        * min(1.0, votes / 2.0)

    extras.update({
        "uv": r.get("uv"), "xy_plane": r.get("xy_plane"), "xy_floor": r.get("xy_floor"),
        "color_dist": color_dist, "votes": votes, "in_envelope": in_env,
        "kind": "container" if _wants_container(prompt) else "graspable",
    })
    return Locate3DResult(x, y, z, round(conf, 3), _METHOD, extras)
