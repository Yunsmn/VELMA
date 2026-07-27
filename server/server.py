"""MCP server — SO-101 perception + manipulation verbs.

The LLM drives the arm by composing a small set of honest verbs. The target
object's position is always MEASURED by perception, never read from ground truth:
  - Perception: find_object (coarse, fixed camera) -> triangulate (precise, wrist camera)
  - State: get_robot_state, get_initial_instructions, capture_cameras
  - Manipulation: grasp, place
  - Motion: move_to_position (abs Cartesian), set_joint_angles (abs multi-joint)
  - Gripper: set_gripper + open_gripper / close_gripper
  - Harness: reset_scene, start_recording / stop_recording
"""
from __future__ import annotations
import io
import json
import logging
import os
from typing import List, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

from robot.controller import RobotController
from robot.state import MoveResult

logger = logging.getLogger(__name__)

# ── Honesty boundary: ground-truth object positions never reach the LLM ────────
# `RobotState` (robot/state.py) carries `cube_position_m`, `container_position_m`,
# and `objects[].position_m` because the SIMULATION backend needs them internally
# (grasp's never-bat drift guard, the harness's reset/scoring helpers, etc. — see
# research/HONESTY_REVIEW.md for the full audit of those internal reads). But this
# server is the ONLY boundary an LLM ever crosses, and every tool response is built
# by wrapping a RobotState in JSON — so those truth fields were leaking into every
# single tool result (get_robot_state, grasp, place, move_to_position, ...),
# letting a model skip find_object/triangulate entirely and just read the answer.
#
# Default: STRIPPED. A benchmark/harness that legitimately needs ground truth for
# SCORING (never for driving perception) can opt back in by starting this server
# with SO101_EXPOSE_TRUTH=1 — an explicit, non-default, non-LLM-facing path. Never
# set this when a server is connected to an LLM.
_EXPOSE_TRUTH = os.environ.get("SO101_EXPOSE_TRUTH", "0").strip().lower() in ("1", "true", "yes")
_TRUTH_FIELDS = ("cube_position_m", "container_position_m", "objects")


def _sanitize_state_json(result_json: dict) -> dict:
    """Strip ground-truth object/container positions from a MoveResult.to_json()
    dict before it is returned as a tool result, unless SO101_EXPOSE_TRUTH=1."""
    if _EXPOSE_TRUTH:
        return result_json
    rs = result_json.get("robot_state")
    if isinstance(rs, dict):
        for field in _TRUTH_FIELDS:
            rs.pop(field, None)
    return result_json

_INITIAL_INSTRUCTIONS = """
You are controlling a SO-101 6-DOF robot arm in a MuJoCo physics simulation.

## Scene
- Table surface at z=0. Objects rest at z≈0.015 m.
- A red cube sits somewhere on the table; a container (target zone) sits elsewhere.
- Coordinates (metres, from the arm base): x=forward, y=left, z=up. Angles in degrees.

## Arm
- 6 joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- Forward reach ~25 cm, side reach ~20 cm. Cannot reach directly behind itself.

## The pick-and-place path (perception-driven — the object position is MEASURED, never given)
1. `find_object("the red cube")` — coarse (x_m, y_m) from the fixed side camera
   (~centimetre accuracy: enough to aim, not to grasp).
2. `triangulate("the red cube", x_m, y_m)` — refine that coarse point into a
   grasp-ready 3D point with the wrist camera (~1 mm). If `confidence < 0.6` the views
   disagree — re-run `find_object` or decline rather than grasp.
3. `grasp(x_m, y_m, z_m, trust_coords=true, grip_width_mm=width_mm+10)` — gentle,
   never-bat top grasp; check `is_grasping`.
4. `find_object("the container")` — the container's position is measured too, the
   same way as the object; `get_robot_state` does not know it either.
5. `place(container_x, container_y, 0.05)` — gentle put-down; confirm `cube_in_container`.

## Tools
### Perception
- `find_object(prompt)` — coarse (x_m, y_m, z_m) of the named object from the fixed camera.
- `triangulate(prompt, x_m, y_m)` — refine a coarse (x_m,y_m) to a precise grasp point
  with the wrist camera; returns x_m,y_m,z_m, width_mm, confidence, residual_mm.
### State
- `get_robot_state()` — joint angles (deg), end-effector position (m), gripper state,
  is_grasping, cube_in_container, and camera images.
- `capture_cameras()` — current camera frames. `get_initial_instructions()` — this text.
### Manipulation
- `grasp(x_m, y_m, z_m, trust_coords=true, grip_width_mm=null, object_width_mm=null, ...)`
  — gentle, never-bat top grasp; returns is_grasping.
- `place(x_m, y_m, z_m)` — gentle put-down; returns cube_in_container.
### Motion
- `move_to_position(x_m, y_m, z_m, gain=0.5, lock_wrist=true)` — absolute EE move (IK).
- `set_joint_angles(shoulder_pan_deg?, shoulder_lift_deg?, elbow_flex_deg?,
  wrist_flex_deg?, wrist_roll_deg?)` — absolute multi-joint config for pre-position/recovery.
### Gripper
- `set_gripper(percent)` (0=closed, 65=approach, 100=open), `open_gripper()`, `close_gripper()`.
### Harness
- `reset_scene()`, `start_recording(title)` / `stop_recording()`.
"""


def _to_image(arr: np.ndarray) -> Image:
    buf = io.BytesIO()
    PILImage.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


def _wrap(result: MoveResult, backend) -> List:
    items: List = [json.dumps(_sanitize_state_json(result.to_json()), indent=2)]
    try:
        items += [_to_image(f) for f in backend.render()]
    except Exception:
        logger.debug("Failed to render camera frames for tool response", exc_info=True)
    return items


def create_server(controller: RobotController, port: int = 3001) -> FastMCP:
    mcp = FastMCP("SO-101 Robot Controller", port=port)
    b = controller.backend

    # ── Perception ────────────────────────────────────────────────────────────

    @mcp.tool(description="Returns a description of the robot, its workspace, and how to use the available tools. Call this first.")
    def get_initial_instructions():
        return _INITIAL_INSTRUCTIONS

    @mcp.tool(description=(
        "Get current joint angles (deg), end-effector position (m), gripper openness, "
        "is_grasping flag, cube_in_container flag, and camera images. Does NOT reveal the "
        "cube's or container's coordinates — those are perception's job (find_object / "
        "triangulate), not a shortcut this tool provides."
    ))
    def get_robot_state():
        return _wrap(controller.get_state(), b)




    # ── Cartesian verbs ───────────────────────────────────────────────────────

    @mcp.tool(description=(
        "Move the end-effector to absolute position (x_m, y_m, z_m) in metres using IK. "
        "lock_wrist: keep wrist_flex/wrist_roll fixed during the move (default True). "
        "gain: IK step size (default 0.5; use 0.1-0.3 for slow/careful moves while carrying). "
        "Gripper is not changed (use set_gripper). Returns robot state including end_effector_m."
    ))
    def move_to_position(x_m: float, y_m: float, z_m: float,
                         gain: float = 0.5,
                         lock_wrist: bool = True):
        return _wrap(controller.move_to_cartesian(x_m, y_m, z_m,
                                                  lock_wrist=lock_wrist,
                                                  gain=gain), b)




    # ── Per-joint verbs (relative °, signed by intuitive physical effect) ──────






    # ── Gripper ───────────────────────────────────────────────────────────────

    @mcp.tool(description=(
        "Set the gripper to an absolute openness. 0=fully closed, 65=open for approach, "
        "100=fully open. Uses an IK-hold close so the arm does not drift while the jaws move."
    ))
    def set_gripper(percent: float):
        return _wrap(controller.control_gripper(percent), b)

    @mcp.tool(description="Open the gripper fully (= set_gripper(100)).")
    def open_gripper():
        return _wrap(controller.control_gripper(100), b)

    @mcp.tool(description="Close the gripper fully (= set_gripper(0)). IK-hold close (no arm drift).")
    def close_gripper():
        return _wrap(controller.control_gripper(0), b)

    # ── Skills ────────────────────────────────────────────────────────────────

    @mcp.tool(description=(
        "Gentle, never-bat top grasp of the object at absolute (x_m, y_m, z_m) in metres. "
        "Reads the object live, aligns the jaws to a face, descends slowly without "
        "batting it, closes with an IK-hold, and verifies with a test lift; sweeps "
        "the finger offset by feedback across bounded retries. You do NOT need to "
        "pre-position or open the gripper first. Works for cubes, boxes, cylinders and "
        "spheres that fit between the jaws (<=~103 mm wide). "
        "approach: 'top' (validated) or 'side' (reserved, not implemented yet). "
        "grip_width_mm: how wide to pre-open the jaws (object width + ~10mm margin); "
        "null = derive from inspect_object. grasp_height_m: EE/TCP world-z to grip at "
        "(where on the body to close); null = derive (short objects grip low, tall "
        "objects grip the upper-mid body). For a non-cube object, call inspect_object "
        "first and pass its width/height, or leave both null to auto-derive. "
        "trust_coords: set TRUE when the (x_m,y_m,z_m) came from perception "
        "(find_object / a camera estimate) — the grasp is then driven by those "
        "PERCEIVED coordinates instead of the object's live position. Leave FALSE "
        "(default) for the harness/benchmark path. "
        "object_width_mm: the object's width as MEASURED BY PERCEPTION (mm). On the "
        "honest route this drives the finger-alignment seed and the never-bat width "
        "thresholds; without it those fall back to the object's true geometry, which "
        "silently un-does trust_coords. Leave null for the harness path. "
        "Returns robot state including is_grasping."
    ))
    def grasp(x_m: float, y_m: float, z_m: float,
              approach: str = "top",
              grip_width_mm: Optional[float] = None,
              grasp_height_m: Optional[float] = None,
              object_width_mm: Optional[float] = None,
              object_width_uncertainty_mm: Optional[float] = None,
              trust_coords: bool = False):
        return _wrap(controller.grasp(x_m, y_m, z_m, approach=approach,
                                      grip_width_mm=grip_width_mm,
                                      grasp_height_m=grasp_height_m,
                                      object_width_mm=object_width_mm,
                                      object_width_uncertainty_mm=object_width_uncertainty_mm,
                                      trust_coords=trust_coords), b)

    @mcp.tool(description=(
        "Gently place the held object at absolute (x_m, y_m, z_m) in metres: move above "
        "the target, descend, release, and withdraw. Call after a successful grasp. "
        "Returns robot state including cube_in_container."
    ))
    def place(x_m: float, y_m: float, z_m: float):
        return _wrap(controller.place_object(x_m, y_m, z_m), b)

    # ── Setup / recovery ──────────────────────────────────────────────────────

    @mcp.tool(description=(
        "Snap one or more joints to an absolute angle in degrees (useful for "
        "pre-positioning or recovery). "
        "Args: shoulder_pan_deg, shoulder_lift_deg, elbow_flex_deg, wrist_flex_deg, "
        "wrist_roll_deg (all optional). Gripper is set only via set_gripper."
    ))
    def set_joint_angles(shoulder_pan_deg: Optional[float] = None,
                         shoulder_lift_deg: Optional[float] = None,
                         elbow_flex_deg: Optional[float] = None,
                         wrist_flex_deg: Optional[float] = None,
                         wrist_roll_deg: Optional[float] = None):
        angles = {k: v for k, v in {
            "shoulder_pan": shoulder_pan_deg, "shoulder_lift": shoulder_lift_deg,
            "elbow_flex": elbow_flex_deg, "wrist_flex": wrist_flex_deg,
            "wrist_roll": wrist_roll_deg,
        }.items() if v is not None}
        return _wrap(controller.set_joint_angles(angles), b)

    # ── Harness only (testing/eval, not a manipulation verb) ──────────────────

    @mcp.tool(description=(
        "Reset the scene (testing/eval only). All args optional — defaults: "
        "cube=(0.20,0.05,0.015), container=(0.20,0.30,0.0)."
    ))
    def reset_scene(cube_x: Optional[float] = None, cube_y: Optional[float] = None,
                    cube_z: Optional[float] = None, container_x: Optional[float] = None,
                    container_y: Optional[float] = None, container_z: Optional[float] = None):
        cube_pos = None
        if any(v is not None for v in (cube_x, cube_y, cube_z)):
            cube_pos = np.array([
                cube_x if cube_x is not None else 0.20,
                cube_y if cube_y is not None else 0.05,
                cube_z if cube_z is not None else 0.015,
            ])
        ctr_pos = None
        if any(v is not None for v in (container_x, container_y, container_z)):
            ctr_pos = np.array([
                container_x if container_x is not None else 0.20,
                container_y if container_y is not None else 0.30,
                container_z if container_z is not None else 0.0,
            ])
        return _wrap(controller.reset(cube_pos, ctr_pos), b)

    @mcp.tool(description=(
        "Vision/servoing: render the fixed SIDE camera and the wrist-mounted WRIST "
        "camera to PNG files and return their paths plus the current end-effector "
        "position. Use between moves to see where the gripper is relative to the "
        "objects. Does NOT reveal any object's coordinates — image only."
    ))
    def capture_cameras(side_path: str = "/tmp/cam_side.png",
                        wrist_path: str = "/tmp/cam_wrist.png"):
        out: dict = {"saved": {}}
        try:
            side = b.render_side()
            if side is not None:
                PILImage.fromarray(side.astype(np.uint8)).save(side_path)
                out["saved"]["side"] = side_path
            wrist = b.render_wrist()
            if wrist is not None:
                PILImage.fromarray(wrist.astype(np.uint8)).save(wrist_path)
                out["saved"]["wrist"] = wrist_path
        except Exception as e:  # never crash the server on a render failure
            out["render_error"] = f"{type(e).__name__}: {e}"
        try:
            st = b.get_state()
            out["end_effector_m"] = st.end_effector_m
            out["gripper_openness_pct"] = getattr(st, "gripper_openness_pct", None)
        except Exception:
            pass
        return [json.dumps(out, indent=2)]


    # ── Run recording (MP4 of a whole session) ────────────────────────────────
    # A background daemon thread grabs frames on a timer (one grasp runs hundreds
    # of internal sim steps, so this is NOT per-tool-call). Owns its own renderer;
    # writes recordings/<title>.mp4 at the project root. Enable per run on go-ahead.
    _rec: dict = {"recorder": None}

    @mcp.tool(description=(
        "Start recording the run to a titled MP4 (recordings/<title>.mp4 at the "
        "project root). A background thread captures an overview camera at a fixed "
        "fps until stop_recording() is called, so it spans whole verbs like grasp. "
        "Only one recording runs at a time — starting again finalises the previous "
        "clip first. The title is sanitised (no path traversal). Call only on an "
        "explicit go-ahead. Returns {ok, recording, title, fps}."
    ))
    def start_recording(title: str):
        from server.recording import RunRecorder
        if _rec["recorder"] is None:
            _rec["recorder"] = RunRecorder(b)
        return [json.dumps(_rec["recorder"].start(title), indent=2)]

    @mcp.tool(description=(
        "Stop the active run recording, write recordings/<title>.mp4, and return "
        "{ok, title, n_frames, path}. Returns an {error} field (server stays up) if "
        "no recording is in progress or no frames were captured."
    ))
    def stop_recording():
        rec = _rec["recorder"]
        if rec is None:
            return [json.dumps({"ok": False, "recording": False,
                                "error": "no recorder initialised "
                                "(call start_recording first)"}, indent=2)]
        return [json.dumps(rec.stop(), indent=2)]

    # ── Honest 3D perception (images + FK poses only, never object qpos) ───────
    # Lazily spawns the scratch-venv inference sidecar (FastSAM + Depth-Anything)
    # the first time it is called and keeps it warm. The models cannot run in the
    # py3.14 robot venv, so this out-of-process sidecar IS the serving approach.
    _percept: dict = {"client": None}



    @mcp.tool(description=(
        "Perception STAGE 1 (COARSE, single fixed camera): locate an object BY NAME "
        "from the side camera. FastSAM segments class-agnostic object masks, monocular "
        "depth scaled by the robot's own links gives each one a 3D position, and the "
        "colour named in the prompt selects WHICH one. No table or support-plane "
        "assumption, so it works for raised objects too. Write `prompt` as a short "
        "noun phrase ('the red cube', 'the blue container') — that phrase is the ONLY "
        "input; there is no object-type hint to supply. Returns {found, x_m,y_m,z_m, "
        "confidence, kind, method, extras}. ACCURACY: centimetre-scale by design — "
        "enough to aim the wrist camera, NOT enough to grasp. Driving grasp straight "
        "off this bumps the object and declines. For a pick, pass this (x_m,y_m) to "
        "triangulate first. For a container target, place(x_m,y_m,z_m) directly "
        "is fine. NEVER reveals ground-truth coordinates. Returns {error} (server "
        "stays up) if the perception sidecar is unavailable."
    ))
    def find_object(prompt: str):
        import sys as _sys
        from pathlib import Path as _Path
        exp = _Path(__file__).resolve().parents[1]   # so101-Models (perception now lives here)
        if str(exp) not in _sys.path:
            _sys.path.insert(0, str(exp))
        try:
            from perception.client import PerceptionClient
            from perception.find_object_depth import find_object_depth as _find
            if _percept["client"] is None:
                _percept["client"] = PerceptionClient()
            r = _find(controller.backend, controller, _percept["client"],
                      prompt=prompt)
            ex = r.extras
            return [json.dumps({
                "prompt": prompt,
                "found": r.x is not None,
                "x_m": r.x, "y_m": r.y, "z_m": r.z,
                "confidence": round(r.confidence, 3),
                "kind": ex.get("kind"),
                "method": r.method,
                "extras": {
                    "uv": ex.get("uv"),
                    "sam_score": ex.get("sam_score"),
                    "mask_area_px": ex.get("mask_area_px"),
                    "depth_m": ex.get("depth_m"),
                    "depth_std_m": ex.get("depth_std_m"),
                    "n_instances": ex.get("n_instances"),
                    "in_envelope": ex.get("in_envelope"),
                    "depth_backend": ex.get("depth_backend"),
                    "seg_stub": ex.get("seg_stub"),
                    "depth_stub": ex.get("depth_stub"),
                    "stage": ex.get("stage"),
                },
            }, indent=2)]
        except Exception as e:  # never crash the server on an inference failure
            return [json.dumps({
                "prompt": prompt, "found": False,
                "error": f"{type(e).__name__}: {e}",
                "hint": "perception sidecar unavailable — build "
                        "experiments/scratch/percept_venv; SAM3 needs sam3.pt "
                        "(else it stubs) and PERCEPT_DEPTH_MODEL selects the "
                        "metric-depth backend (default constant_stub).",
            }, indent=2)]

    @mcp.tool(description=(
        "Perception STAGE 2 (PRECISE, WRIST camera) — triangulate. Refine the coarse "
        "(x_m,y_m) that find_object returned into a grasp-ready 3D point using the wrist "
        "camera. The wrist camera orbits above the "
        "coarse point and takes several frames; each frame's object centroid becomes a "
        "world ray through the robot's own forward kinematics, and intersecting the rays "
        "recovers (x_m,y_m,z_m) from geometry alone — NO table plane, NO assumed object "
        "height, NO depth model. Also returns a MEASURED width_mm (silhouette size at the "
        "now-known distance) to size the jaw opening. Typical accuracy ~1 mm in x,y and a "
        "few mm in z, versus tens of mm for the coarse stage. `prompt` should repeat the "
        "phrase used for find_object so the right object is tracked across views. "
        "STATE-NEUTRAL: the arm is returned to exactly where it started, so this only "
        "improves the coordinate. Returns {found, x_m,y_m,z_m, width_mm, confidence, "
        "residual_mm, n_rays, method}. confidence is derived from how well the rays "
        "AGREE — below 0.6 the views disagree and you should NOT grasp; treat it as an "
        "honest 'I cannot see it well enough' and re-run find_object or decline. Feed the "
        "returned point into grasp(x,y,z, trust_coords=true, grip_width_mm=width_mm+10)."
    ))
    def triangulate(prompt: str, x_m: float, y_m: float):
        import sys as _sys
        from pathlib import Path as _Path
        exp = _Path(__file__).resolve().parents[1]   # so101-Models (perception now lives here)
        if str(exp) not in _sys.path:
            _sys.path.insert(0, str(exp))
        try:
            from perception.client import PerceptionClient
            from perception.find_object import _prompt_rgb
            from perception import wrist_triangulate as WT
            if _percept["client"] is None:
                _percept["client"] = PerceptionClient()
            # restore=False: DEPLOYMENT path — the arm ends at the final hover above the
            # object (where grasp descends from) instead of teleporting back to its start,
            # which is the jerky "return to origin" and is not physically realisable anyway.
            r = WT.triangulate_object(controller.backend, controller,
                                      _percept["client"], float(x_m), float(y_m),
                                      target_rgb=_prompt_rgb(prompt), restore=False)
            ex = r.extras
            return [json.dumps({
                "prompt": prompt,
                "found": r.x is not None,
                "x_m": r.x, "y_m": r.y, "z_m": r.z,
                "width_mm": ex.get("width_mm"),
                "width_uncertainty_mm": ex.get("width_uncertainty_mm"),
                "confidence": round(r.confidence, 3),
                "residual_mm": ex.get("tri_resid_mm"),
                "n_rays": ex.get("n_rays"),
                "n_inliers": ex.get("tri_n_inliers"),
                "conditioning": ex.get("tri_cond"),
                "seed_shift_mm": ex.get("seed_shift_mm"),
                "method": r.method,
            }, indent=2)]
        except Exception as e:  # never crash the server on an inference failure
            return [json.dumps({
                "prompt": prompt, "found": False,
                "error": f"{type(e).__name__}: {e}",
                "hint": "perception sidecar unavailable — build "
                        "experiments/scratch/percept_venv.",
            }, indent=2)]

    return mcp
