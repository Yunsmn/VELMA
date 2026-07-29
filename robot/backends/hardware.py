"""Real SO-101 hardware backend (Feetech STS3215 bus via lerobot).

Wraps lerobot's `SO101Follower`, which owns the motor bus and loads the
calibration produced by `lerobot-calibrate` from its own cache (keyed by robot
id, under ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json).

Setup:
  uv pip install -e ".[core_scripts]"      # from a lerobot checkout, Python 3.12
  lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
      --robot.id=my_follower_arm

Because lerobot pulls in torch, and torch has no Python 3.14 wheels, this backend
only runs under the 3.12 interpreter that has lerobot installed. The MuJoCo model
is used for KINEMATICS ONLY (IK, forward kinematics); all motion goes to servos.

FRAME ALIGNMENT — READ BEFORE TRUSTING ANY CARTESIAN MOTION: lerobot's calibrated
joint zero and sign do not necessarily match the MuJoCo model's joint convention.
JOINT_SIGN and JOINT_OFFSET_DEG map model angles <-> servo degrees and are still
IDENTITY, i.e. unmeasured. Until they are measured joint by joint, `get_state()`'s
end_effector_m and anything driven by IK are not to be believed.

Three behaviours here differ deliberately from the simulation backend, because the
consequences differ on a physical arm:
  * reset() does NOT move. In simulation it re-poses the scene; here it would swing
    the arm to HOME_DEG in a single goal write from wherever it happens to be.
    Homing is an explicit, ramped action (`move_home`).
  * Torque is HELD on disconnect. lerobot's default cuts it, and the arm then sags
    under gravity (measured: elbow_flex fell ~11 deg on the first probe). Use
    `release()` to deliberately go limp, ideally from a low pose.
  * Every deliberate move is interpolated, and the bus enforces a per-command
    relative clamp, so no single write can command a large jump.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import mujoco
import numpy as np

from robot.interface import RobotBackend
from robot.state import RobotState

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_AXES = JOINT_NAMES + ["gripper"]

# Model-angle <-> servo-degree mapping.
#
# SIGNS: measured 2026-07-29 on the real arm by jogging one joint at a time and
# watching the wrist camera, then comparing against what the model predicts a
# positive angle does (see docs). All four position-affecting joints agreed with
# the model's convention, so the signs are +1 — not by assumption, by measurement:
#   shoulder_pan  +13.3 deg -> scene moved 726->542 px left, i.e. arm swung right;
#                              model says +pan moves the EE right.        -> +1
#   shoulder_lift -10.7 deg -> adapter radius 115->105 px, gripper rose;
#                              model says +lift lowers the EE.            -> +1
#   elbow_flex     -5.5 deg -> radius 93->88 px, gripper rose;
#                              model says +elbow lowers the EE.           -> +1
#   wrist_flex    -12.3 deg -> radius 112->92 px, nose tilted up;
#                              model says +wrist_flex lowers the EE.      -> +1
# wrist_roll is NOT measured: it does not move the end-effector, and the scene
# rotation is confounded because the camera and the jaws sit on opposite sides of
# the roll joint (the jaws shift in frame when it turns). Left at +1, UNVERIFIED.
JOINT_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

# OFFSETS: still unmeasured, and this is the live problem. Matching each joint's
# calibrated half-span against the model's range suggests offsets near zero
# (elbow +-97.2 vs +-96.8 deg, lift +-104.5 vs +-100), but that cannot be the whole
# story: at the arm's actual pose the model puts the gripper 33 mm BELOW the base
# plane while the camera plainly shows it above the table. Something in the zero
# convention is still wrong, so IK-driven tools (move_to_position, grasp, place)
# must not be trusted yet. Fixing this needs an absolute spatial reference — e.g.
# touching a known point and solving for the offsets — not another sign check.
JOINT_OFFSET_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Home posture in MODEL degrees (matches the sim's neutral pose).
HOME_DEG = np.array([0.0, 0.0, 0.0, 90.0, 90.0])
HOME_GRIPPER_PCT = 100.0

# Largest position jump a single command may request, in degrees. lerobot clamps
# each action against the present position, so this bounds servo speed too.
MAX_RELATIVE_TARGET_DEG = 8.0

# Deliberate ramped moves (move_home, move_gripper, release).
RAMP_HZ = 30.0
DEFAULT_RAMP_S = 2.5

# Largest relative move a single move_joint_servo_delta request may ask for.
MAX_JOINT_DELTA_DEG = 30.0

# is_grasping: the gripper is commanded shut but something holds the jaws apart.
GRIP_CLOSED_PCT = 8.0          # below this the jaws are effectively touching
GRIP_OBJECT_MIN_PCT = 3.0      # jaws stalled at least this far open
GRIP_LOAD_MIN = 60             # raw Present_Load magnitude (0-1023), provisional
_LOAD_MAGNITUDE_MASK = 0x3FF   # bit 10 is direction, not magnitude

# Wrist camera. Index is discovered unless SO101_WRIST_CAM pins it.
WRIST_CAM_ENV = "SO101_WRIST_CAM"
CAM_WIDTH, CAM_HEIGHT = 1280, 720
CAM_WARMUP_FRAMES = 5          # USB cameras need a few frames to settle exposure


class HardwareBackend(RobotBackend):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 port: str = "/dev/ttyACM0", robot_id: str = "my_follower_arm"):
        self.model = model
        self.data = data
        self.port = port
        self.robot_id = robot_id

        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as e:
            raise RuntimeError(
                "lerobot is not installed in this interpreter. The hardware backend "
                "needs lerobot (and therefore torch), which has no Python 3.14 wheels "
                "— run the server under the Python 3.12 venv that has it: "
                "`uv pip install -e \".[core_scripts]\"` from a lerobot checkout."
            ) from e

        self.robot = SO101Follower(SO101FollowerConfig(
            port=port,
            id=robot_id,
            use_degrees=True,
            max_relative_target=MAX_RELATIVE_TARGET_DEG,
            disable_torque_on_disconnect=False,
        ))
        self.robot.connect(calibrate=False)

        self._camera = None
        self._camera_index: Optional[int] = None
        self._last_gripper_cmd_pct = self._read_gripper_pct()
        self._torque_enabled = True  # connect() energises the servos

        # Hardware has no ground truth for object poses; these hold the most recent
        # perception estimate and are only as good as whatever wrote them.
        self._cube_pos = np.array([0.20, 0.05, 0.015])
        self._container_pos = np.array([0.20, 0.30, 0.0])

        self._sync_mujoco_state()

    # ── servo <-> model conversions ───────────────────────────────────────────

    def _model_rad_to_servo_deg(self, angles_rad: np.ndarray) -> np.ndarray:
        return np.degrees(angles_rad) * JOINT_SIGN + JOINT_OFFSET_DEG

    def _servo_deg_to_model_rad(self, angles_deg: np.ndarray) -> np.ndarray:
        return np.radians((angles_deg - JOINT_OFFSET_DEG) / JOINT_SIGN)

    # ── raw bus access ────────────────────────────────────────────────────────

    def _read_all(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {axis: float(obs[f"{axis}.pos"]) for axis in ALL_AXES}

    def _read_joints_deg(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{name}.pos"] for name in JOINT_NAMES], dtype=float)

    def _read_gripper_pct(self) -> float:
        return float(self.robot.get_observation()["gripper.pos"])

    def _send(self, joints_deg: np.ndarray, gripper_pct: float) -> None:
        action = {f"{name}.pos": float(joints_deg[i]) for i, name in enumerate(JOINT_NAMES)}
        gripper_pct = float(np.clip(gripper_pct, 0.0, 100.0))
        action["gripper.pos"] = gripper_pct
        self._last_gripper_cmd_pct = gripper_pct
        self.robot.send_action(action)

    def _ramp_to(self, targets: dict[str, float], secs: float = DEFAULT_RAMP_S) -> None:
        """Interpolate from the present pose to `targets` so no servo sees a step."""
        start = self._read_all()
        goal = {**start, **targets}
        n_steps = max(1, int(secs * RAMP_HZ))
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            blended = {axis: start[axis] + alpha * (goal[axis] - start[axis])
                       for axis in ALL_AXES}
            self._send(np.array([blended[j] for j in JOINT_NAMES]), blended["gripper"])
            time.sleep(1.0 / RAMP_HZ)

    def _sync_mujoco_state(self) -> None:
        """Mirror real servo positions into MuJoCo so IK/FK track the actual arm."""
        self.data.qpos[:5] = self._servo_deg_to_model_rad(self._read_joints_deg())
        mujoco.mj_forward(self.model, self.data)

    # ── RobotBackend interface ────────────────────────────────────────────────

    def reset(self, cube_pos: Optional[np.ndarray] = None,
              container_pos: Optional[np.ndarray] = None) -> None:
        """Update tracked object estimates. Deliberately does NOT move the arm —
        see the module docstring. Call `move_home()` to home it."""
        if cube_pos is not None:
            self._cube_pos = cube_pos.copy()
        if container_pos is not None:
            self._container_pos = container_pos.copy()
        self._sync_mujoco_state()

    def move_joint_servo_delta(self, joint: str, delta_deg: float,
                               secs: float = 2.5, settle_tries: int = 6) -> dict:
        """Ramp ONE joint by a relative delta in SERVO degrees.

        This deliberately bypasses the MuJoCo model's joint limits. Those limits
        describe the model's convention, and while the model<->servo mapping is
        unmeasured they do not describe the real arm: the arm's own resting pose
        already reads outside them, so clipping to them can turn a request to
        raise the arm into a command that lowers it into the table.

        The servo's own calibrated range still applies, and MAX_JOINT_DELTA_DEG
        bounds any single request. Because lerobot clamps each goal to
        MAX_RELATIVE_TARGET_DEG of the present position, a fast ramp under-delivers
        (the goal stream outruns the servo), so the target is re-issued until the
        reading stops improving.
        """
        if joint not in ALL_AXES:
            raise ValueError(f"unknown joint {joint!r}; valid: {ALL_AXES}")
        if abs(delta_deg) > MAX_JOINT_DELTA_DEG:
            raise ValueError(
                f"delta {delta_deg:+.1f} deg exceeds the {MAX_JOINT_DELTA_DEG} deg "
                "per-request cap; issue several smaller moves instead"
            )

        start = self._read_all()
        target = start[joint] + delta_deg

        previous = start[joint]
        for _ in range(settle_tries):
            self._ramp_to({joint: target}, secs=secs)
            reached = self._read_all()[joint]
            if abs(reached - target) <= 1.0 or abs(reached - previous) < 0.3:
                break  # arrived, or stopped making progress (stall / hard stop)
            previous = reached
            secs = 1.0  # already close; top up quickly

        end = self._read_all()
        return {
            "joint": joint,
            "requested_delta_deg": round(delta_deg, 2),
            "start_deg": round(start[joint], 2),
            "target_deg": round(target, 2),
            "reached_deg": round(end[joint], 2),
            "achieved_delta_deg": round(end[joint] - start[joint], 2),
            "shortfall_deg": round(target - end[joint], 2),
            "pose_deg": {k: round(v, 2) for k, v in end.items()},
        }

    def sample_pose(self) -> dict[str, float]:
        """One raw servo reading, no MuJoCo involved. Used to watch the arm while
        it is being moved by hand."""
        return self._read_all()

    def move_home(self, secs: float = 4.0) -> None:
        """Explicit, ramped move to the neutral pose."""
        home_servo = self._model_rad_to_servo_deg(np.radians(HOME_DEG))
        targets = {name: float(home_servo[i]) for i, name in enumerate(JOINT_NAMES)}
        targets["gripper"] = HOME_GRIPPER_PCT
        self._ramp_to(targets, secs=secs)

    def apply_control(self, ctrl: np.ndarray) -> None:
        joints_deg = self._model_rad_to_servo_deg(np.asarray(ctrl[:5], dtype=float))
        low, high = self.model.actuator_ctrlrange[5]
        gripper_pct = (ctrl[5] - low) / (high - low) * 100.0
        self._send(joints_deg, gripper_pct)

    def step(self) -> None:
        self._sync_mujoco_state()

    def move_gripper(self, openness_pct: float, steps: int = 120) -> None:
        self._ramp_to({"gripper": float(openness_pct)}, secs=1.5)

    def get_state(self) -> RobotState:
        self._sync_mujoco_state()
        angles_deg = {name: round(float(np.degrees(self.data.qpos[i])), 2)
                      for i, name in enumerate(JOINT_NAMES)}
        ee = self.data.site_xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        ].copy()
        return RobotState(
            joint_angles_deg=angles_deg,
            end_effector_m={ax: round(float(v), 4) for ax, v in zip("xyz", ee)},
            cube_position_m={ax: round(float(v), 4) for ax, v in zip("xyz", self._cube_pos)},
            container_position_m={ax: round(float(v), 4) for ax, v in zip("xyz", self._container_pos)},
            is_grasping=self.is_grasping(),
            cube_in_container=False,  # requires vision
            gripper_openness_pct=round(self._read_gripper_pct(), 1),
        )

    def is_grasping(self) -> bool:
        """True when the gripper is commanded shut but the jaws are held apart.

        There is no contact sensor. The primary signal is geometric — commanded
        closed, yet stalled open — which needs no force calibration. Present_Load
        corroborates it; its threshold is provisional and wants tuning against a
        known object before anything depends on it.
        """
        if self._last_gripper_cmd_pct > GRIP_CLOSED_PCT:
            return False
        actual_pct = self._read_gripper_pct()
        if actual_pct < GRIP_OBJECT_MIN_PCT:
            return False  # jaws fully shut: nothing between them
        try:
            raw = int(self.robot.bus.read("Present_Load", "gripper"))
            return (raw & _LOAD_MAGNITUDE_MASK) >= GRIP_LOAD_MIN
        except Exception:
            # Load unreadable: fall back to the geometric signal alone rather than
            # reporting a grasp the bus could not corroborate.
            return True

    def get_object_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cube_pos.copy(), self._container_pos.copy()

    def inspect_object(self) -> dict:
        """Hardware has no model geometry to read. The simulation backend answers
        this from the MuJoCo scene, i.e. from ground truth; here there is none, so
        callers that rely on it (grasp's width/shape checks) are running on a
        placeholder until the vision pipeline fills it in."""
        return {
            "shape": "unknown",
            "width_mm": 30.0,
            "height_mm": 30.0,
            "footprint_radius_mm": 15.0,
            "center_m": [round(float(v), 4) for v in self._cube_pos],
            "upright": True,
            "source": "placeholder",
        }

    # ── wrist camera ──────────────────────────────────────────────────────────

    def _discover_camera_index(self) -> int:
        pinned = os.environ.get(WRIST_CAM_ENV)
        if pinned is not None:
            return int(pinned)

        import cv2

        # Skip the laptop's built-in camera and any virtual loopback device: the
        # wrist camera is the external USB one.
        for index in range(10):
            path = f"/dev/video{index}"
            if not os.path.exists(path):
                continue
            name = ""
            try:
                with open(f"/sys/class/video4linux/video{index}/name") as fh:
                    name = fh.read().strip()
            except OSError:
                continue
            if "Integrated" in name or "OBS" in name or "Virtual" in name:
                continue
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            opened = cap.isOpened()
            ok = opened and cap.read()[0]
            cap.release()
            if ok:
                return index
        raise RuntimeError(
            "No wrist camera found. Plug in the USB camera, or pin it with "
            f"{WRIST_CAM_ENV}=<index> (see `ls /dev/video*`)."
        )

    def _ensure_camera(self):
        if self._camera is not None:
            return self._camera
        import cv2

        index = self._discover_camera_index()
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"wrist camera /dev/video{index} would not open")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self._camera, self._camera_index = cap, index
        return cap

    def render_wrist(self) -> np.ndarray:
        """A frame from the physical wrist camera, RGB."""
        import cv2

        cap = self._ensure_camera()
        frame = None
        for _ in range(CAM_WARMUP_FRAMES):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(
                f"wrist camera /dev/video{self._camera_index} opened but returned no frames"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def render_side(self) -> None:
        """There is no side camera on this rig — only two USB ports, both spoken
        for. Returning None makes callers say so; rendering the MuJoCo side camera
        instead would hand back a picture of a simulated table as if it were real."""
        return None

    def render(self) -> list[np.ndarray]:
        """Returns [wrist_rgb]. One camera only, so this is a one-element list
        rather than the simulation's [side, wrist]."""
        return [self.render_wrist()]

    def on_pick_and_place_start(self) -> None:
        pass

    # ── torque (lead-through / hand guiding) ──────────────────────────────────

    def set_torque(self, enabled: bool) -> dict:
        """Energise or release the arm's servos.

        Releasing lets the arm be moved BY HAND, which is how the model<->servo
        convention gets measured without commanding a motion whose direction is
        not yet known. It also means the arm is no longer holding itself up: it
        WILL sag under gravity the instant torque drops, so it must be supported
        first. Re-enabling holds wherever it is then resting.
        """
        if enabled:
            self.robot.bus.enable_torque()
        else:
            self.robot.bus.disable_torque()
        self._torque_enabled = enabled
        pose = self._read_all()
        return {
            "torque_enabled": enabled,
            "joint_angles_deg": {k: round(v, 2) for k, v in pose.items()},
            "note": ("Servos released — SUPPORT THE ARM, it will sag."
                     if not enabled else "Servos holding at the current pose."),
        }

    # ── shutdown ──────────────────────────────────────────────────────────────

    def release(self, secs: float = 3.0) -> None:
        """Deliberately go limp. The arm will sag, so send it low first."""
        self.robot.config.disable_torque_on_disconnect = True
        self.robot.disconnect()

    def close(self) -> None:
        """Disconnect while HOLDING torque, so the arm keeps its pose."""
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None
        try:
            self.robot.disconnect()
        except Exception:
            pass
