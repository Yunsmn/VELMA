"""MuJoCo physics backend."""
from __future__ import annotations
import math
import sys
import threading
from typing import Optional

import mujoco
import numpy as np

from robot.interface import RobotBackend
from robot.state import RobotState

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# NOTE: reset() writes these into qpos, so they OVERRIDE whatever pose the scene
# XML declares for the cube and container bodies. Two sources of truth — keep the
# XML in sync with these, or the rendered scene will not match the file you read.
DEFAULT_CUBE_POS = np.array([0.20, 0.05, 0.015])
# Container moved off the reach boundary: the old (0.20, 0.30) sat at r = 0.3606,
# which is both the arm's practical limit (IK leaves a 12 mm residual above it)
# and a hair outside the detector's r < 0.36 gate. (0.10, -0.27) is r = 0.288,
# 10 mm residual, unoccluded in the side camera, and 0.335 m clear of the cube.
DEFAULT_CONTAINER_POS = np.array([0.10, -0.27, 0.0])

# Finger pad total thickness on the grasp axis (each pad is a 1.25mm half-size
# box -> 2.5mm thick). Subtracted from the pad-center separation so jaw_gap_mm is
# the FREE object width the jaws bracket, not the pad-center distance.
_PAD_THICK_MM = 2.5


def _color_name(rgba) -> str:
    """Nearest basic color name for an rgba (so the agent can map 'the red one')."""
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    palette = {
        "red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.1, 0.3, 0.9),
        "yellow": (0.9, 0.8, 0.1), "purple": (0.6, 0.2, 0.8), "orange": (0.95, 0.5, 0.1),
        "grey": (0.5, 0.5, 0.5), "white": (0.95, 0.95, 0.95), "black": (0.05, 0.05, 0.05),
    }
    best, bestd = "unknown", 1e9
    for n, (cr, cg, cb) in palette.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < bestd:
            best, bestd = n, d
    return best


class SimulationBackend(RobotBackend):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, viewer: bool = False):
        self.model = model
        self.data = data
        self._lock = threading.Lock()
        self._step_lock = threading.Lock()

        self._setup_ids()

        # Offscreen renderer is disabled: cameras aren't needed for the verb library,
        # and the offscreen EGL/GL context crashes the server under repeated headless
        # restarts (the multi-shape test harness restarts many times). render() already
        # returns [] when this is None. Re-enable here only if camera frames are needed.
        self._renderer = None
        # Separate offscreen renderer for visual servoing (render_side/render_wrist).
        # Kept apart from self._renderer so enabling capture never bloats every tool
        # response via _wrap. Built lazily on first capture; needs an EGL context
        # (launch with MUJOCO_GL=egl). Single long-lived server = no restart crash.
        self._offscreen = None

    def _setup_ids(self) -> None:
        def _id(t, name):
            i = mujoco.mj_name2id(self.model, t, name)
            if i == -1:
                raise ValueError(f"'{name}' not found in model")
            return i

        J, G, S = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE

        cube_jid = _id(J, "cube_joint")
        self.cube_qpos_addr = self.model.jnt_qposadr[cube_jid]
        container_jid = _id(J, "container_joint")
        self.container_qpos_addr = self.model.jnt_qposadr[container_jid]
        self.container_dof_addr = self.model.jnt_dofadr[container_jid]

        self.cube_geom_id = _id(G, "cube_geom")
        self.container_site_id = _id(S, "container_center")
        self.ee_site_id = _id(S, "gripperframe")
        self.static_pad_id = _id(G, "static_finger_pad")
        self.moving_pad_id = _id(G, "moving_finger_pad")

        # ── Multi-object registry: every free body except the container, so grasp /
        #    inspect / is_grasping / check_grip can retarget to whatever object we are
        #    acting on. cube_geom_id / cube_qpos_addr are the ACTIVE pointers (mutated
        #    by set_active_object); primary_* anchor the reset to the 'cube' body. ──
        self.objects: dict[str, dict] = {}
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            jname = mujoco.mj_id2name(self.model, J, jid)
            if jname in ("container_joint", "wall_joint"):
                continue
            bid = int(self.model.jnt_bodyid[jid])
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            gid = next((g for g in range(self.model.ngeom)
                        if int(self.model.geom_bodyid[g]) == bid), -1)
            if gid != -1:
                self.objects[bname] = {"body_id": bid, "geom_id": gid,
                                       "qpos_addr": int(self.model.jnt_qposadr[jid])}

        # Obstacle gate: the WALL is a heavy free body, so a "collision" is judged by
        # DISPLACEMENT from its rest pose — a sub-mm graze leaves it put; a real push
        # moves it (fail). Track its body + freejoint qpos to measure how far it moved.
        self.floor_geom_id = mujoco.mj_name2id(self.model, G, "floor")
        self.obstacles: dict[str, int] = {}
        self.wall_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wall")
        if self.wall_body_id != -1:
            wjid = mujoco.mj_name2id(self.model, J, "wall_joint")
            self.wall_qpos_addr = int(self.model.jnt_qposadr[wjid])
            wgid = next((g for g in range(self.model.ngeom)
                         if int(self.model.geom_bodyid[g]) == self.wall_body_id), -1)
            if wgid != -1:
                self.obstacles["wall_geom"] = wgid
        else:
            self.wall_qpos_addr = -1
        self._wall_init_xyz = None          # rest pose, captured on reset
        # Fail only on a real PUSH: a deliberate plow shoves the wall ~28mm and it stays;
        # an incidental grasp/carry nudge peaks ~12mm and springs back. 18mm sits between.
        self._obstacle_move_tol = 0.018     # m
        self._obstacle_hit = False

        # The 'cube' body is the default active target and the reset anchor.
        self.primary_geom_id = self.cube_geom_id
        self.primary_qpos_addr = self.cube_qpos_addr
        self.active_object = "cube"

    def _mj_step(self) -> None:
        with self._step_lock:
            mujoco.mj_step(self.model, self.data)
            # Obstacle gate: latch if the wall has been DISPLACED from its rest pose by
            # more than the tolerance (a real push), ignoring incidental sub-mm grazes.
            if (self.wall_qpos_addr != -1 and self._wall_init_xyz is not None
                    and not self._obstacle_hit):
                wpos = self.data.qpos[self.wall_qpos_addr:self.wall_qpos_addr + 3]
                if float(np.linalg.norm(wpos - self._wall_init_xyz)) > self._obstacle_move_tol:
                    self._obstacle_hit = True

    def _mj_forward(self) -> None:
        with self._step_lock:
            mujoco.mj_forward(self.model, self.data)

    # ── RobotBackend interface ────────────────────────────────────────────────

    def _object_rest_z(self) -> float:
        """Z of the object's CENTER when it rests on the table (z=0), from its geom.
        box/sphere: half-extent up; cylinder/capsule: half-height (+r for capsule).
        Lets non-cube shapes start resting on the table instead of penetrating it."""
        T = mujoco.mjtGeom
        gtype = int(self.model.geom_type[self.cube_geom_id])
        gs = self.model.geom_size[self.cube_geom_id]
        if gtype == T.mjGEOM_BOX:
            return float(gs[2])
        if gtype == T.mjGEOM_SPHERE:
            return float(gs[0])
        if gtype == T.mjGEOM_CYLINDER:
            return float(gs[1])
        if gtype == T.mjGEOM_CAPSULE:
            return float(gs[1] + gs[0])
        return float(DEFAULT_CUBE_POS[2])

    def reset(self, cube_pos: Optional[np.ndarray] = None,
              container_pos: Optional[np.ndarray] = None) -> None:
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            # default the active target back to the primary cube before resting it
            self.cube_geom_id = self.primary_geom_id
            self.cube_qpos_addr = self.primary_qpos_addr
            self.active_object = "cube"
            self._obstacle_hit = False
            self._wall_init_xyz = None      # re-captured after the settle loop below
            if cube_pos is not None:
                cp = cube_pos
            else:
                # rest the object on the table at the default x,y for its actual shape
                cp = np.array([DEFAULT_CUBE_POS[0], DEFAULT_CUBE_POS[1],
                               self._object_rest_z()])
            ctp = container_pos if container_pos is not None else DEFAULT_CONTAINER_POS

            self.data.qpos[self.cube_qpos_addr:self.cube_qpos_addr + 3] = cp
            self.data.qpos[self.cube_qpos_addr + 3:self.cube_qpos_addr + 7] = [1, 0, 0, 0]
            self.data.qpos[self.container_qpos_addr:self.container_qpos_addr + 3] = ctp
            self.data.qpos[self.container_qpos_addr + 3:self.container_qpos_addr + 7] = [1, 0, 0, 0]
            self.data.qpos[3] = np.pi / 2
            self.data.qpos[4] = np.pi / 2
            self.data.ctrl[3] = np.pi / 2
            self.data.ctrl[4] = np.pi / 2
            self._mj_forward()
            for _ in range(50):
                self._mj_step()
            # Capture the wall's SETTLED rest pose as the reference for the move-gate.
            if self.wall_qpos_addr != -1:
                self._wall_init_xyz = self.data.qpos[
                    self.wall_qpos_addr:self.wall_qpos_addr + 3].copy()

    def apply_control(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl

    def step(self) -> None:
        self._mj_step()

    def move_gripper(self, openness_pct: float, steps: int = 120) -> None:
        r = self.model.actuator_ctrlrange[5]
        action = float(np.clip((openness_pct / 100.0) * 2.0 - 1.0, -1.0, 1.0))
        target = float((action + 1) / 2 * (r[1] - r[0]) + r[0])
        start = float(self.data.ctrl[5])
        arm_q = self.data.qpos[:5].copy()
        for s in range(steps):
            t = (s + 1) / steps
            self.data.ctrl[:5] = arm_q
            self.data.ctrl[5] = start + (target - start) * t
            self._mj_step()

    def get_state(self) -> RobotState:
        self._mj_forward()
        joint_deg = {name: round(float(np.degrees(self.data.qpos[i])), 2)
                     for i, name in enumerate(JOINT_NAMES)}
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        cube = self.data.qpos[self.cube_qpos_addr:self.cube_qpos_addr + 3].copy()
        container = self.data.qpos[self.container_qpos_addr:self.container_qpos_addr + 3].copy()

        r = self.model.actuator_ctrlrange[5]
        gripper_pct = round(float((self.data.ctrl[5] - r[0]) / (r[1] - r[0]) * 100), 1)

        dist_xy = float(np.linalg.norm(cube[:2] - container[:2]))
        # Honest containment check (2026-07-27 fix). The old bounds (dist_xy < 0.12,
        # z in [+0.01, +0.07]) FALSE-POSITIVE for a cube resting on the open TABLE just
        # outside the container: a cube on the table sits at container_z + ~0.015 (its
        # own half-height above z=0), which already falls inside the old z-band, and
        # dist_xy for an adjacent table position (e.g. 90-110 mm off-center) can still
        # be < 0.12. Verified empirically: cube at dist_xy=0.111, z=container_z+0.015
        # (sitting on the table beside the container, NOT inside it) scored True under
        # the old bounds. The container's inner half-width is 0.07 m (0.08 outer wall
        # half minus 0.01 wall thickness), so a cube resting on the RAISED FLOOR sits at
        # container_z + ~0.035 (floor top + cube half-height) -- distinctly higher than
        # a table-resting cube. Tightened bounds: dist_xy must be within the physical
        # inner footprint (with a small margin for a cube resting against a wall), and
        # z must be high enough that only floor-resting (not table-resting) qualifies.
        in_container = (
            dist_xy < 0.075
            and float(container[2]) + 0.022 < float(cube[2]) < float(container[2]) + 0.07
        )

        # Compact survey of every object on the table (full geometry via list_objects).
        objects_summary = []
        for name, o in self.objects.items():
            c = self.data.geom_xpos[o["geom_id"]]
            objects_summary.append({
                "name": name,
                "color": self._geom_color(o["geom_id"]),
                "position_m": {ax: round(float(v), 4) for ax, v in zip("xyz", c)},
            })

        return RobotState(
            joint_angles_deg=joint_deg,
            end_effector_m={ax: round(float(v), 4) for ax, v in zip("xyz", ee_pos)},
            cube_position_m={ax: round(float(v), 4) for ax, v in zip("xyz", cube)},
            container_position_m={ax: round(float(v), 4) for ax, v in zip("xyz", container)},
            is_grasping=self.is_grasping(),
            cube_in_container=in_container,
            gripper_openness_pct=gripper_pct,
            objects=objects_summary,
            obstacle_hit=self._obstacle_hit,
        )

    def is_grasping(self) -> bool:
        contacts: set[int] = set()
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom1, self.data.contact[i].geom2
            if g1 == self.cube_geom_id:
                contacts.add(g2)
            elif g2 == self.cube_geom_id:
                contacts.add(g1)
        return self.static_pad_id in contacts and self.moving_pad_id in contacts

    def pad_contacts(self) -> tuple[bool, bool]:
        """Return (static_pad_touches_object, moving_pad_touches_object).

        Decomposes is_grasping into the two individual pad contacts so check_grip
        can tell a real two-pad bracket from a single-pad graze."""
        contacts: set[int] = set()
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom1, self.data.contact[i].geom2
            if g1 == self.cube_geom_id:
                contacts.add(g2)
            elif g2 == self.cube_geom_id:
                contacts.add(g1)
        return self.static_pad_id in contacts, self.moving_pad_id in contacts

    def jaw_gap_mm(self) -> float:
        """Live gap (mm) between the two finger pads along the GRASP AXIS.

        The pads close toward each other along the gripper's local y; the moving
        jaw also arcs up (+z) as it opens, so the full 3D pad distance overstates
        the usable gap. We project the pad-to-pad vector onto the grasp axis (the
        line from the static pad to the moving pad in the x/y plane, which is the
        direction the jaws actually pinch) and add back the pad thickness so the
        result is the FREE object width the jaws are currently holding open to:
          gap = ||p_moving - p_static|| projected on the in-plane closing axis.
        A jaw closed on a W-mm object reads ~W; a jaw that closed PAST the object
        (pads met empty) reads near 0. Geometry-direct (no empirical fit)."""
        self._mj_forward()
        ps = self.data.geom_xpos[self.static_pad_id].copy()
        pm = self.data.geom_xpos[self.moving_pad_id].copy()
        d = pm - ps
        # in-plane closing separation (ignore the +z arc of the moving jaw) + the
        # two pad half-thicknesses that bracket the object faces.
        inplane = float(math.hypot(d[0], d[1]))
        free_gap = inplane - _PAD_THICK_MM / 1000.0
        return max(0.0, free_gap * 1000.0)

    def grip_metrics(self, settle_steps: int = 6) -> dict:
        """Confidence read for check_grip. Probe the grip's stability by stepping the
        sim a few frames (holding ctrl), then report the two pad contacts, the live jaw
        gap, the object width, and whether the grip held across the settle window.

        NON-DESTRUCTIVE: the full physics state (qpos/qvel/act/ctrl/time) is SNAPSHOT
        before the settle steps and RESTORED after, so this lookahead stability probe
        never advances or perturbs the real sim — a good grip is read without being
        shaken (the settle steps were perturbing edge-cell grips enough to shear them
        on the subsequent carry). The returned contacts/gap reflect the CURRENT (pre-
        probe) state; only `stable` comes from the lookahead.

        Returns {is_grasping, both_pads, static_pad, moving_pad, jaw_gap_mm,
        object_width_mm, stable}."""
        info = self.inspect_object()
        # snapshot the full sim state
        q0 = self.data.qpos.copy()
        v0 = self.data.qvel.copy()
        a0 = self.data.act.copy() if self.data.act.size else None
        c0 = self.data.ctrl.copy()
        t0 = float(self.data.time)
        holds = 0
        n = max(1, settle_steps)
        for _ in range(n):
            self.data.ctrl[:] = c0
            self._mj_step()
            if self.is_grasping():
                holds += 1
        stable = holds >= n - 1  # tolerate a single flicker frame
        # restore the pre-probe state exactly (lookahead must not perturb the sim)
        self.data.qpos[:] = q0
        self.data.qvel[:] = v0
        if a0 is not None:
            self.data.act[:] = a0
        self.data.ctrl[:] = c0
        self.data.time = t0
        self._mj_forward()
        sp, mp = self.pad_contacts()
        return {
            "is_grasping": self.is_grasping(),
            "both_pads": bool(sp and mp),
            "static_pad": bool(sp),
            "moving_pad": bool(mp),
            "jaw_gap_mm": round(self.jaw_gap_mm(), 2),
            "object_width_mm": round(float(info.get("width_mm", 30.0)), 2),
            "stable": bool(stable),
            "holds": holds,
            "settle_steps": int(n),
        }

    def get_object_positions(self) -> tuple[np.ndarray, np.ndarray]:
        self._mj_forward()
        cube = self.data.qpos[self.cube_qpos_addr:self.cube_qpos_addr + 3].copy()
        container = self.data.site_xpos[self.container_site_id].copy()
        return cube, container

    def inspect_object(self) -> dict:
        """Geometry of the ACTIVE target object (the one the last grasp bound to)."""
        return self._inspect_geom(self.cube_geom_id)

    def _inspect_geom(self, geom_id: int) -> dict:
        """Read a geom's geometry from the MuJoCo model.

        Derives width/height/footprint from geom_type + geom_size:
          - box:      size=(hx,hy,hz);  width = 2*max(hx,hy), height = 2*hz,
                      footprint_radius = hypot(hx,hy) (corner radius of the base).
          - cylinder: size=(r,halfh);   width = 2*r, height = 2*halfh, footprint = r.
          - sphere:   size=(r,);        width = height = 2*r, footprint = r.
          - capsule:  size=(r,halfh);   width = 2*r, height = 2*(halfh+r), footprint = r.
        Width is the jaw-opening dimension ("fits between the grippers"); height is
        how tall the object stands (drives where to grip). center_m is the live world
        position; upright is whether the object's body z-axis still points up (a
        cylinder/box can be tipped over by a bad approach).
        """
        self._mj_forward()
        gtype = int(self.model.geom_type[geom_id])
        gsize = self.model.geom_size[geom_id].copy()
        center = self.data.geom_xpos[geom_id].copy()
        # body z-axis in world frame (3rd column of the geom rotation matrix)
        xmat = self.data.geom_xmat[geom_id].reshape(3, 3)
        up_axis = xmat[:, 2]
        upright = bool(up_axis[2] > 0.94)  # within ~20deg of vertical

        T = mujoco.mjtGeom
        if gtype == T.mjGEOM_BOX:
            hx, hy, hz = float(gsize[0]), float(gsize[1]), float(gsize[2])
            shape = "box"
            width_mm = 2.0 * max(hx, hy) * 1000.0
            height_mm = 2.0 * hz * 1000.0
            footprint_mm = float(np.hypot(hx, hy)) * 1000.0
        elif gtype == T.mjGEOM_CYLINDER:
            r, halfh = float(gsize[0]), float(gsize[1])
            shape = "cylinder"
            width_mm = 2.0 * r * 1000.0
            height_mm = 2.0 * halfh * 1000.0
            footprint_mm = r * 1000.0
        elif gtype == T.mjGEOM_SPHERE:
            r = float(gsize[0])
            shape = "sphere"
            width_mm = height_mm = 2.0 * r * 1000.0
            footprint_mm = r * 1000.0
        elif gtype == T.mjGEOM_CAPSULE:
            r, halfh = float(gsize[0]), float(gsize[1])
            shape = "capsule"
            width_mm = 2.0 * r * 1000.0
            height_mm = 2.0 * (halfh + r) * 1000.0
            footprint_mm = r * 1000.0
        else:
            # ellipsoid or other — fall back to bounding extents from geom_size
            shape = f"geomtype_{gtype}"
            width_mm = 2.0 * float(max(gsize[0], gsize[1])) * 1000.0
            height_mm = 2.0 * float(gsize[2] if gsize[2] > 0 else gsize[0]) * 1000.0
            footprint_mm = float(max(gsize[0], gsize[1])) * 1000.0

        return {
            "shape": shape,
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "footprint_radius_mm": round(footprint_mm, 2),
            "center_m": [round(float(v), 4) for v in center],
            "upright": upright,
        }

    # ── Multi-object: active-target binding + scene scan ──────────────────────

    def _geom_color(self, geom_id: int) -> str:
        matid = int(self.model.geom_matid[geom_id])
        if matid < 0:
            return "unknown"
        return _color_name(self.model.mat_rgba[matid])

    def set_active_object(self, name: str) -> bool:
        """Bind the active target (what inspect/is_grasping/check_grip refer to)."""
        o = self.objects.get(name)
        if not o:
            return False
        self.cube_geom_id = o["geom_id"]
        self.cube_qpos_addr = o["qpos_addr"]
        self.active_object = name
        return True

    def set_active_object_by_point(self, x: float, y: float) -> str:
        """Rebind the active target to the free object nearest (x, y) in the table
        plane. Returns the chosen object's name; grasp calls this so it acts on
        whatever object is at the requested point, not always the original cube."""
        self._mj_forward()
        best, bestd = None, 1e9
        for name, o in self.objects.items():
            c = self.data.geom_xpos[o["geom_id"]]
            d = float((c[0] - x) ** 2 + (c[1] - y) ** 2)
            if d < bestd:
                best, bestd = name, d
        if best is not None:
            self.set_active_object(best)
        return best or self.active_object

    def list_objects(self) -> dict:
        """Perception: every manipulable object on the table + static obstacles.

        Per object: name, color, shape, width/height/footprint, live center, upright.
        Gives the agent a survey of a multi-object scene so it can pick a target (then
        pass that object's center to grasp). It intentionally does NOT judge
        graspability — that is the agent's call from the object's size and shape."""
        self._mj_forward()
        objs = []
        for name, o in self.objects.items():
            info = self._inspect_geom(o["geom_id"])
            info["name"] = name
            info["color"] = self._geom_color(o["geom_id"])
            objs.append(info)
        obstacles = []
        for name, gid in self.obstacles.items():
            c = self.data.geom_xpos[gid].copy()
            half = self.model.geom_size[gid].copy()
            ob = {"name": name, "type": "obstacle",
                  "center_m": [round(float(v), 4) for v in c],
                  "half_extent_m": [round(float(v), 4) for v in half[:3]]}
            if self.wall_qpos_addr != -1 and self._wall_init_xyz is not None:
                wpos = self.data.qpos[self.wall_qpos_addr:self.wall_qpos_addr + 3]
                ob["moved_mm"] = round(float(np.linalg.norm(wpos - self._wall_init_xyz)) * 1000, 1)
            obstacles.append(ob)
        return {"objects": objs, "obstacles": obstacles, "active_object": self.active_object}

    def render(self) -> list[np.ndarray]:
        if self._renderer is None:
            return []
        frames = []
        for azimuth, elevation in [(225.0, -35.0), (180.0, -85.0)]:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.20, 0.17, 0.05]
            cam.distance = 0.75 if elevation < -50 else 0.60
            cam.azimuth = azimuth
            cam.elevation = elevation
            self._renderer.update_scene(self.data, camera=cam)
            frames.append(self._renderer.render().copy())
        return frames

    # ── Isolated cameras for visual servoing ────────────────────────────────────
    SERVO_W = 640
    SERVO_H = 480

    # Set once if building/using the offscreen renderer fails, so the failure is reported
    # clearly one time instead of raising on every perception call (which, from the MCP
    # server's worker thread, previously surfaced as an opaque tool error or took the
    # server down). Perception callers already treat a None frame as an honest miss.
    _offscreen_failed = False

    # Set by main.py when the interactive viewer owns the GL context. Perception cameras are
    # then refused OUTRIGHT rather than allowed to return corrupted frames. Measured on this
    # machine: with the viewer up the offscreen render came back fully black in one session
    # (mean pixel 0.6) and merely dim-with-viewer-overlay-bleed in another (mean 43.6 vs 90.6
    # headless) — the second is far more dangerous, because it looks like a plausible image
    # and surfaces as "no detection", sending you hunting for a perception bug that is really
    # a rendering-context conflict. Detecting the CAUSE beats trying to threshold the symptom.
    _viewer_active = False

    def _get_offscreen(self) -> Optional["mujoco.Renderer"]:
        if self._viewer_active:
            if not type(self)._offscreen_failed:
                type(self)._offscreen_failed = True
                print("[perception] camera tools are DISABLED while the interactive viewer "
                      "is running — the viewer holds the GL context and the offscreen render "
                      "comes back black or dimmed, which looks like 'no detection'.\n"
                      "[perception] For find_object / refine_grasp_point, restart headless:\n"
                      "[perception]   config.yaml viewer: false   (and MUJOCO_GL=egl)",
                      file=sys.stderr)
            return None
        if self._offscreen_failed:
            return None
        if self._offscreen is None:
            try:
                self._offscreen = mujoco.Renderer(
                    self.model, height=self.SERVO_H, width=self.SERVO_W)
            except Exception as exc:
                # Most likely cause: an interactive viewer already owns a GL context on
                # the main thread and this renderer is being built on the server thread.
                type(self)._offscreen_failed = True
                print(f"[perception] offscreen renderer unavailable: "
                      f"{type(exc).__name__}: {exc}\n"
                      f"[perception] camera tools (find_object / refine_grasp_point) will "
                      f"return 'no frame'. Run the server WITHOUT the viewer "
                      f"(config.yaml viewer: false, MUJOCO_GL=egl) to use them.",
                      file=sys.stderr)
                return None
        return self._offscreen

    # A frame this dark and this flat is not a scene — the tabletop renders around mean 90.
    # Measured: a genuine headless frame reads mean 90.5 / std 38.8, whereas a frame taken
    # while an interactive viewer owns the GL context reads mean 0.6 / std 6.1.
    _BLANK_MEAN = 8.0
    _BLANK_STD = 12.0

    def _check_frame(self, img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Reject an all-black frame instead of handing it to the detector.

        When an interactive viewer holds the GL context, the offscreen renderer returns a
        BLANK image rather than raising — so perception silently reports "nothing detected"
        and looks broken, when in fact it never received a picture. Catching it here turns
        that into one explicit, actionable message.
        """
        if img is None:
            return None
        if float(img.mean()) < self._BLANK_MEAN and float(img.std()) < self._BLANK_STD:
            if not type(self)._offscreen_failed:
                type(self)._offscreen_failed = True
                print("[perception] offscreen renderer returned a BLANK frame — the "
                      "interactive viewer is holding the GL context.\n"
                      "[perception] Camera tools cannot work with the viewer on this "
                      "driver. Run the server headless for perception:\n"
                      "[perception]   config.yaml viewer: false   (and MUJOCO_GL=egl)",
                      file=sys.stderr)
            return None
        return img

    def render_view(self, azimuth: float, elevation: float,
                    distance: float = 1.15,
                    lookat=(0.0, 0.0, 0.04)) -> Optional[np.ndarray]:
        """Render a fixed free camera at (azimuth, elevation, distance, lookat).

        The general fixed-external-camera render: several of these at different
        azimuths give a multi-camera rig for plane-free triangulation WITHOUT a
        wrist camera (poses come from camera_math.free_cam_params with the same
        constants). render_side is the (135, -35) instance of this.

        Returns None if no offscreen renderer is available (see _get_offscreen) so a
        GL problem degrades into an honest 'no frame' rather than an exception on the
        server's worker thread.
        """
        r = self._get_offscreen()
        if r is None:
            return None
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        try:
            self._mj_forward()
            r.update_scene(self.data, camera=cam)
            return self._check_frame(r.render().copy())
        except Exception as exc:
            type(self)._offscreen_failed = True
            print(f"[perception] offscreen render failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return None

    def render_side(self) -> Optional[np.ndarray]:
        """Fixed angled overhead 'side' camera — gross object position + height.
        None if no offscreen renderer is available."""
        return self.render_view(135.0, -35.0, 1.15, (0.0, 0.0, 0.04))

    def render_wrist(self) -> Optional[np.ndarray]:
        """Wrist-mounted camera ('wrist_cam'), looking down the gripper — fine
        left/right + forward/back alignment, jaws visible at the frame bottom.
        Returns None if the active scene defines no wrist_cam, or if no offscreen
        renderer is available."""
        r = self._get_offscreen()
        if r is None:
            return None
        try:
            self._mj_forward()
            r.update_scene(self.data, camera="wrist_cam")
            return self._check_frame(r.render().copy())
        except (KeyError, ValueError):
            return None       # scene has no wrist_cam — not a renderer fault
        except Exception as exc:
            type(self)._offscreen_failed = True
            print(f"[perception] wrist render failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return None

    def gripper_geometry(self) -> dict:
        """World positions of the grasp point and fingertips — the true contact
        geometry. graspframe is the point that lands between the pads; its offset from
        the IK end-effector (gripperframe) is a rotated 3D vector that changes with
        wrist orientation, so it can't be guessed from the EE position alone."""
        self._mj_forward()

        def site(name):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                return None
            p = self.data.site_xpos[sid]
            return [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]

        return {
            "graspframe": site("graspframe"),
            "gripperframe": site("gripperframe"),
            "static_fingertip": site("static_fingertip"),
            "moving_fingertip": site("moving_fingertip"),
        }

    def on_pick_and_place_start(self) -> None:
        self.model.dof_damping[self.container_dof_addr:self.container_dof_addr + 6] = 5.0
        self.model.dof_frictionloss[self.container_dof_addr:self.container_dof_addr + 6] = 1.0
        container_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "container")
        self.model.body_mass[container_body] = 10.0
