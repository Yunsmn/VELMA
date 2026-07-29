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

FRAME ALIGNMENT: lerobot's calibrated joint zero and sign do not match the MuJoCo
model's convention, so JOINT_SIGN and JOINT_OFFSET_DEG map between them. BOTH ARE
NOW MEASURED (see their definitions below), and the result was checked against
data it was not fitted to: three poses recorded earlier with an object resting on
the table agree on gripper height to 1.6 mm. Base-frame coordinates therefore
mean the same thing here as in the simulation — the table is z=0 and an object at
rest is z=0.015. `get_state()`'s end_effector_m can be believed.

Behaviours that differ deliberately from the simulation backend, because the
consequences differ on a physical arm:
  * reset() does NOT move. In simulation it re-poses the scene; here it would swing
    the arm to HOME_DEG in a single goal write from wherever it happens to be.
    Homing is an explicit, ramped action (`move_home`).
  * Torque is HELD on disconnect. lerobot's default cuts it, and the arm then sags
    under gravity (measured: elbow_flex fell ~11 deg on the first probe). Use
    `release()` to deliberately go limp, ideally from a low pose.
  * Motion is by GOAL, not by software ramp: publish the target and let the servo's
    own position loop run. Interpolating in software rate-limited the arm badly.
  * Cartesian targets are solved in the model and commanded once (`move_to_xyz`).
    The simulation's approach of iterating small increments against the robot
    stalls here, because the increments fall below the servo stiction floor.
  * pad_contacts/grip_metrics answer with real signals and are COARSER than the
    simulator's: there is no per-pad contact sensing, so a one-pad graze cannot be
    distinguished from a proper bracket.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import Optional

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

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

# OFFSETS: MEASURED 2026-07-29 from a single reference posture.
#
# lerobot puts each joint's zero at the midpoint of its calibrated sweep, which
# is meaningless here because this arm's joint limits are COUPLED — across three
# calibrations shoulder_pan and elbow_flex reproduced to 2-4 ticks while
# wrist_flex and wrist_roll never reproduced at all. Sweeping to mechanical stops
# does not rescue it either: the elbow's stop is set by the wrist camera's bulk,
# so it is a collision limit that moves with the arm's configuration.
#
# Instead the arm was posed BY HAND into a posture the model defines exactly —
# upper arm vertical, forearm horizontal, gripper axis horizontal, arm pointing
# straight forward — and the servos read there. The wrist camera confirmed the
# gripper was horizontal (it saw the room, not the table); the first attempt had
# it pointing down and produced a nonsense 106 deg offset, which is how that
# error was caught.
#
#   joint          servo@ref   model@ref   offset = servo - model
#   shoulder_pan     -3.08       0.00        -3.08
#   shoulder_lift    +2.90     -13.97       +16.87
#   elbow_flex       +4.57     +16.18       -11.61
#   wrist_flex       +8.40      -2.21       +10.61
#   wrist_roll       -4.35       0.00        -4.35
#
# wrist_roll is the least trustworthy: its calibration hit the full encoder span
# in all three runs, so levelling the jaws by eye is the only thing pinning it.
JOINT_OFFSET_DEG = np.array([-3.08, 16.87, -11.61, 10.61, -4.35])

# Home posture in MODEL degrees (matches the sim's neutral pose).
HOME_DEG = np.array([0.0, 0.0, 0.0, 90.0, 90.0])
HOME_GRIPPER_PCT = 100.0

# Largest position jump a single command may request, in degrees. lerobot clamps
# each action against the present position, so this bounds servo speed too.
#
# It also bounds FORCE: the servo's position P-controller drives in proportion to
# the goal-minus-present error, so a tight clamp caps how hard it can push. At
# 8 deg this arm could not lift its own weight — elbow_flex answered a request
# for -8 deg (against gravity) with -1.3, while +8 deg (with gravity) gave +7.0.
# Widening the clamp restores authority without touching Max_Torque_Limit, which
# lerobot sets to 50% and which still protects the motor from burnout.
MAX_RELATIVE_TARGET_DEG = float(os.environ.get("SO101_MAX_REL_TARGET_DEG", "20.0"))

# Servo position-loop stiffness. lerobot writes P=16, which is soft: the arm
# yields to gravity between commands, so every upward move begins by clawing
# back sag that should never have happened, and lands short. Stiffer holding is
# the fix for both the lag and the shortfall — it is a HOLDING problem, not a
# trajectory problem. Raised here; D is left as lerobot sets it to damp the
# oscillation a higher P would otherwise invite.
POSITION_P = int(os.environ.get("SO101_POSITION_P", "32"))
POSITION_I = int(os.environ.get("SO101_POSITION_I", "0"))
POSITION_D = int(os.environ.get("SO101_POSITION_D", "32"))

# Deliberate ramped moves (move_home, move_gripper, release).
RAMP_HZ = 30.0
DEFAULT_RAMP_S = 2.5

# Largest relative move a single move_joint_servo_delta request may ask for.
MAX_JOINT_DELTA_DEG = 30.0

# Goal-and-poll motion. The servo runs its own position loop, so we publish the
# goal and watch, rather than interpolating the trajectory in software.
POLL_INTERVAL_S = 0.05        # 20 Hz goal refresh + readback
ARRIVE_TOL_DEG = 0.8
STALL_EPS_DEG = 0.15          # per-poll movement below this counts as no progress
STALL_POLLS = 8               # ~0.4 s of no progress means it is not going further
DEFAULT_MOVE_TIMEOUT_S = 8.0

# Smooth limit sweeps. The goal advances at a fixed rate so the servo follows a
# moving setpoint rather than lunging at a distant one.
SWEEP_SPEED_DEG_S = 12.0
SWEEP_TIMEOUT_S = 25.0
SWEEP_STALL_POLLS = 14      # ~0.7 s of no progress before calling it a limit

# Grasping.
#
# The first version of this asked whether the gripper was commanded below an
# ABSOLUTE 8%, which is only reachable when the jaws are nearly touching. On a
# 45 mm object that can never happen, so the only way to make it report a grip
# was to command a full close and leave the servo stalled against the object.
# Holding that stall burned out motor 6 — it dropped off the bus entirely and
# needed a power cycle. The test is now RELATIVE (is the goal meaningfully
# tighter than where the jaws actually are?), which works at any object size and
# never requires a sustained stall.
GRIP_OBJECT_MIN_PCT = 3.0       # jaws stalled at least this far open
GRIP_STALL_MARGIN_PCT = 1.5     # goal must be this much tighter than actual
GRIP_LOAD_MIN = 60              # raw Present_Load magnitude (0-1023), provisional
_LOAD_MAGNITUDE_MASK = 0x3FF    # bit 10 is direction, not magnitude

# Closing onto an object: step in until the jaws stop making progress (contact),
# then hold with a small bounded squeeze at REDUCED torque, rather than leaving
# the servo straining at full effort against something it can never reach.
GRIP_STEP_PCT = 8.0
GRIP_STALL_EPS_PCT = 0.8        # progress below this between steps means contact
# Squeeze and hold torque are both deliberately small. A 4% squeeze at torque
# limit 250 still tripped the STS3215's overload protection after a few minutes
# of holding a rigid object — the servo latched and dropped off the bus. Holding
# TIME matters as much as force: the jaws are already mechanically closed around
# the object, and the squeeze only adds stall current.
# Balanced against BOTH failure modes seen on this arm: 4% at torque 250 tripped
# overload after minutes of holding, while 1.5% at 130 was too weak and the
# adapter was dropped mid-carry. The decisive variable is hold TIME, not force,
# so this sits in between and the caller is expected to keep grips short.
GRIP_SQUEEZE_PCT = 3.5          # how far past contact to hold
GRIP_HOLD_TORQUE = 220          # Max_Torque_Limit while holding (lerobot uses 500)
GRIP_MOVE_TORQUE = 500
GRIP_MAX_STEPS = 14

# Do not sit clamped indefinitely. Overload trips on sustained current, so a
# grip that is going nowhere should be reported rather than quietly held.
GRIP_HOLD_WARN_S = 60.0

# Finger-pad thickness, matching the simulation's constant, so the free-gap
# figure means the same thing on both backends.
PAD_THICKNESS_M = 0.0025
# Gap between re-reads when answering "is this grip stable?". The simulator steps
# physics and rewinds; hardware just watches for a moment.
GRIP_SETTLE_INTERVAL_S = 0.05

# Wrist camera. Index is discovered unless SO101_WRIST_CAM pins it.
WRIST_CAM_ENV = "SO101_WRIST_CAM"
CAM_WIDTH, CAM_HEIGHT = 1280, 720
CAM_WARMUP_FRAMES = 5          # USB cameras need a few frames to settle exposure


def resolve_port(port: str) -> str:
    """Return a serial port that exists, preferring a stable by-id path.

    /dev/ttyACM<N> is assigned in plug order, so unplugging the arm or a USB
    glitch renames it — this rig moved from ttyACM0 to ttyACM1 mid-session and
    every call then failed with a misleading "Port is in use". The symlinks in
    /dev/serial/by-id are keyed to the adapter's serial number and survive that,
    so fall back to one when the configured path has gone missing.
    """
    if os.path.exists(port):
        return port

    by_id = "/dev/serial/by-id"
    candidates = sorted(glob.glob(os.path.join(by_id, "*"))) if os.path.isdir(by_id) else []
    if not candidates:
        raise RuntimeError(
            f"{port} does not exist and no serial adapter was found under {by_id}. "
            "Is the arm plugged in and powered?"
        )
    resolved = os.path.realpath(candidates[0])
    logger.warning("serial port %s is gone; using %s (%s)",
                   port, resolved, os.path.basename(candidates[0]))
    return resolved


class HardwareBackend(RobotBackend):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 port: str = "/dev/ttyACM0", robot_id: str = "my_follower_arm"):
        self.model = model
        self.data = data
        self.port = port = resolve_port(port)
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
            position_p_coefficient=POSITION_P,
            position_i_coefficient=POSITION_I,
            position_d_coefficient=POSITION_D,
        ))
        self.robot.connect(calibrate=False)

        self._camera = None
        self._camera_index: Optional[int] = None
        self._last_gripper_cmd_pct = self._read_gripper_pct()
        # The gripper's GOAL, which must survive arm motion. _send writes all six
        # axes on every call, so moving an arm joint used to overwrite the gripper
        # goal with wherever the jaws currently are — silently cancelling the
        # squeeze and dropping the grip the moment the arm moved.
        self._gripper_goal_pct = self._last_gripper_cmd_pct
        self._grip_started_at: Optional[float] = None
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
                               timeout_s: float = DEFAULT_MOVE_TIMEOUT_S) -> dict:
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
        hold = dict(start)
        if joint != "gripper":
            # Keep squeezing whatever is held, rather than re-commanding the jaws
            # to their present position and letting go.
            hold["gripper"] = self._gripper_goal_pct
        else:
            self._gripper_goal_pct = target

        # Drive by GOAL, not by a software ramp. Feeding the servo 75
        # interpolated setpoints at 30 Hz and then retrying that up to six times
        # meant a single jog could spend 15 s travelling 3 degrees — the motion
        # was rate-limited by us, not by the motor. Instead: publish the goal and
        # let the servo's own position controller run at its speed, refreshing
        # the goal (lerobot clamps it to MAX_RELATIVE_TARGET ahead of present) and
        # polling until it either arrives or stops making progress.
        deadline = time.time() + timeout_s
        previous = start[joint]
        stalled_polls = 0
        while time.time() < deadline:
            blended = {**hold, joint: target}
            self._send(np.array([blended[j] for j in JOINT_NAMES]), blended["gripper"])
            time.sleep(POLL_INTERVAL_S)

            reached = self._read_all()[joint]
            if abs(reached - target) <= ARRIVE_TOL_DEG:
                break
            if abs(reached - previous) < STALL_EPS_DEG:
                stalled_polls += 1
                if stalled_polls >= STALL_POLLS:
                    break  # hard stop, or gravity/torque limit — stop pushing
            else:
                stalled_polls = 0
            previous = reached

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

    def _set_gripper_torque_limit(self, value: int) -> None:
        try:
            self.robot.bus.write("Max_Torque_Limit", "gripper", int(value))
        except Exception:
            logger.warning("could not set gripper Max_Torque_Limit", exc_info=True)

    def close_on_object(self) -> dict:
        """Close the jaws until they meet resistance, then hold gently.

        Steps inward and watches for the jaws to stop moving. That stall is
        contact. Once found, the goal is set a small fixed amount past contact
        and the gripper's torque limit is dropped, so the hold is a light squeeze
        rather than a servo straining indefinitely at full effort — which is what
        destroyed motor 6 the first time round.
        """
        self._set_gripper_torque_limit(GRIP_MOVE_TORQUE)
        previous = self._read_gripper_pct()
        contact_pct: Optional[float] = None

        for _ in range(GRIP_MAX_STEPS):
            target = max(0.0, previous - GRIP_STEP_PCT)
            self.move_joint_servo_delta("gripper", target - previous)
            actual = self._read_gripper_pct()

            if abs(actual - previous) < GRIP_STALL_EPS_PCT:
                contact_pct = actual
                break
            previous = actual
            if actual <= 0.5:
                break  # fully closed without meeting anything

        actual = self._read_gripper_pct()
        if contact_pct is None:
            self._set_gripper_torque_limit(GRIP_HOLD_TORQUE)
            return {"grasped": False, "reason": "jaws closed without meeting an object",
                    "gripper_pct": round(actual, 1)}

        # Bounded squeeze, then ease off the torque for the hold.
        hold_goal = max(0.0, contact_pct - GRIP_SQUEEZE_PCT)
        self._grip_started_at = time.time()
        self._gripper_goal_pct = hold_goal
        self._send(self._read_joints_deg(), hold_goal)
        time.sleep(0.3)
        self._set_gripper_torque_limit(GRIP_HOLD_TORQUE)

        return {
            "grasped": self.is_grasping(),
            "contact_pct": round(contact_pct, 1),
            "hold_goal_pct": round(hold_goal, 1),
            "gripper_pct": round(self._read_gripper_pct(), 1),
            "load": self._gripper_load(),
            "hold_torque_limit": GRIP_HOLD_TORQUE,
        }

    def clear_gripper_overload(self) -> dict:
        """Clear the gripper's latched overload-protection state.

        Sustained stall current makes the STS3215 latch into overload: it keeps
        answering pings with an error status and refuses to operate, so the whole
        bus handshake fails and the server will not start. The latch releases
        once the motor stops straining, so dropping Torque_Enable clears it
        WITHOUT a power cycle — at the cost of letting go of anything held.
        """
        self.robot.bus.write("Torque_Enable", "gripper", 0, num_retry=3)
        time.sleep(1.5)
        self._grip_started_at = None
        try:
            self.robot.bus.write("Torque_Enable", "gripper", 1, num_retry=3)
            self._set_gripper_torque_limit(GRIP_MOVE_TORQUE)
            recovered = True
        except Exception:
            logger.warning("gripper did not re-enable after overload clear", exc_info=True)
            recovered = False
        return {"recovered": recovered,
                "note": "Anything the gripper was holding has been released."}

    def grip_hold_seconds(self) -> Optional[float]:
        if self._grip_started_at is None:
            return None
        return round(time.time() - self._grip_started_at, 1)

    def release_object(self, open_to_pct: float = 70.0) -> dict:
        """Open the jaws, in chunks if need be.

        Opening from a grip to wide open is a bigger move than the per-request
        cap allows, so asking for it in one go raised and left the object still
        clamped — the failure that ended the first teach run. Split it instead.
        """
        self._grip_started_at = None
        self._set_gripper_torque_limit(GRIP_MOVE_TORQUE)
        self._gripper_goal_pct = open_to_pct

        for _ in range(6):
            remaining = open_to_pct - self._read_gripper_pct()
            if abs(remaining) <= 1.0:
                break
            step = float(np.clip(remaining, -MAX_JOINT_DELTA_DEG, MAX_JOINT_DELTA_DEG))
            self.move_joint_servo_delta("gripper", step)
        return {"gripper_pct": round(self._read_gripper_pct(), 1)}

    def _gripper_load(self) -> Optional[int]:
        try:
            return int(self.robot.bus.read("Present_Load", "gripper")) & _LOAD_MAGNITUDE_MASK
        except Exception:
            return None

    def goto_servo_angles(self, targets: dict[str, float],
                          timeout_s: float = DEFAULT_MOVE_TIMEOUT_S) -> dict:
        """Drive several joints to ABSOLUTE servo angles at once.

        Moving joints together is both quicker and gentler than one at a time:
        the arm sweeps a direct path instead of a staircase. Same goal-and-poll
        approach as a single jog, and the gripper is left alone unless named, so
        a held object keeps being held.
        """
        unknown = set(targets) - set(ALL_AXES)
        if unknown:
            raise ValueError(f"unknown joints: {sorted(unknown)}; valid: {ALL_AXES}")

        start = self._read_all()
        goal = {**start, **targets}
        if "gripper" not in targets:
            goal["gripper"] = self._gripper_goal_pct
        else:
            self._gripper_goal_pct = goal["gripper"]

        moving = [j for j in targets if j != "gripper"]
        deadline = time.time() + timeout_s
        previous = {j: start[j] for j in moving}
        stalled_polls = 0

        while time.time() < deadline:
            self._send(np.array([goal[j] for j in JOINT_NAMES]), goal["gripper"])
            time.sleep(POLL_INTERVAL_S)

            now = self._read_all()
            if moving and max(abs(now[j] - goal[j]) for j in moving) <= ARRIVE_TOL_DEG:
                break
            if moving and max(abs(now[j] - previous[j]) for j in moving) < STALL_EPS_DEG:
                stalled_polls += 1
                if stalled_polls >= STALL_POLLS:
                    break
            else:
                stalled_polls = 0
            previous = {j: now[j] for j in moving}

        end = self._read_all()
        return {
            "target_deg": {k: round(v, 2) for k, v in targets.items()},
            "reached_deg": {k: round(end[k], 2) for k in ALL_AXES},
            "residual_deg": {k: round(end[k] - goal[k], 2) for k in moving},
            "stalled": stalled_polls >= STALL_POLLS,
        }

    def calibrated_limits(self) -> dict[str, tuple[float, float]]:
        """Each joint's calibrated travel, in servo degrees.

        lerobot puts zero at the midpoint of the swept range, so the limits are
        symmetric by construction: +-(span/2). Note these describe the sweep that
        calibration happened to capture, and on this arm the joint limits are
        COUPLED — what a joint can actually reach depends on where the others
        are — so treat these as nominal and trust the measured stop instead.
        """
        limits = {}
        for name, cal in self.robot.bus.calibration.items():
            half = (cal.range_max - cal.range_min) / 2 * 360 / 4095
            limits[name] = (-half, half)
        return limits

    def sweep_joint(self, joint: str, direction: int,
                    speed_deg_s: float = SWEEP_SPEED_DEG_S,
                    timeout_s: float = SWEEP_TIMEOUT_S) -> dict:
        """Move one joint smoothly toward its limit and report where it stops.

        The goal is advanced at a fixed rate rather than being thrown to the far
        end: the servo then tracks a moving setpoint a small distance ahead,
        which is what makes the motion continuous instead of a lurch followed by
        a stall. Stops at the calibrated limit, when the joint stops making
        progress (a real mechanical or torque limit), or on timeout.
        """
        if joint not in ALL_AXES:
            raise ValueError(f"unknown joint {joint!r}; valid: {ALL_AXES}")
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")

        low, high = self.calibrated_limits()[joint]
        nominal_limit = high if direction > 0 else low

        start = self._read_all()
        goal = start[joint]
        hold = dict(start)
        if joint != "gripper":
            hold["gripper"] = self._gripper_goal_pct

        deadline = time.time() + timeout_s
        previous = start[joint]
        stalled_polls = 0
        stopped_by = "timeout"

        while time.time() < deadline:
            goal += direction * speed_deg_s * POLL_INTERVAL_S
            goal = min(goal, nominal_limit) if direction > 0 else max(goal, nominal_limit)

            blended = {**hold, joint: goal}
            self._send(np.array([blended[j] for j in JOINT_NAMES]), blended["gripper"])
            time.sleep(POLL_INTERVAL_S)

            now = self._read_all()[joint]
            if abs(now - previous) < STALL_EPS_DEG:
                stalled_polls += 1
                if stalled_polls >= SWEEP_STALL_POLLS:
                    stopped_by = ("reached calibrated limit"
                                  if abs(now - nominal_limit) < 2.0 else "stalled early")
                    break
            else:
                stalled_polls = 0
            previous = now

            if abs(goal - nominal_limit) < 1e-6 and abs(now - nominal_limit) <= ARRIVE_TOL_DEG:
                stopped_by = "reached calibrated limit"
                break

        end = self._read_all()[joint]
        return {
            "joint": joint,
            "direction": "max" if direction > 0 else "min",
            "start_deg": round(start[joint], 2),
            "nominal_limit_deg": round(nominal_limit, 2),
            "reached_deg": round(end, 2),
            "shortfall_deg": round(abs(nominal_limit - end), 2),
            "stopped_by": stopped_by,
            "speed_deg_s": speed_deg_s,
        }

    def solve_ik(self, target_xyz, seed_deg=None, iters: int = 400,
                 tol_m: float = 0.0015) -> dict:
        """Solve IK for a base-frame target ENTIRELY IN THE MODEL. Moves nothing.

        The simulation's move_to_cartesian iterates against the robot: command a
        small joint increment, read back, repeat. That works in a simulator where
        every increment executes exactly. On this arm the increments land near or
        below the servo stiction floor (~1.5 deg), so the arm does not move, the
        error never shrinks, and the loop exhausts its step budget having gone
        nowhere — the "Max steps" failures.

        Solving in the model instead turns a Cartesian target into ONE finished
        joint configuration, which the goal-and-poll mover executes reliably.
        Damped least squares keeps it stable near singularities; joint limits are
        respected so the answer is one the arm can actually adopt.
        """
        target = np.asarray(target_xyz, dtype=float)
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        scratch = mujoco.MjData(self.model)
        if seed_deg is None:
            self._sync_mujoco_state()
            q = self.data.qpos[:5].copy()
        else:
            q = np.radians(np.asarray(seed_deg, dtype=float))

        lo, hi = self.model.jnt_range[:5, 0], self.model.jnt_range[:5, 1]
        damping = 1e-3
        best_q, best_err = q.copy(), np.inf

        for _ in range(iters):
            scratch.qpos[:5] = q
            mujoco.mj_forward(self.model, scratch)
            err = target - scratch.site_xpos[site]
            norm = float(np.linalg.norm(err))
            if norm < best_err:
                best_err, best_q = norm, q.copy()
            if norm < tol_m:
                break
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, jac, None, site)
            J = jac[:, :5]
            # damped least squares: J^T (J J^T + lambda I)^-1 err
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
            q = np.clip(q + dq, lo, hi)

        model_deg = np.degrees(best_q)
        servo_deg = self._model_rad_to_servo_deg(best_q)
        return {
            "reachable": bool(best_err < 0.01),
            "residual_mm": round(best_err * 1000, 2),
            "model_deg": {n: round(float(model_deg[i]), 2)
                          for i, n in enumerate(JOINT_NAMES)},
            "servo_deg": {n: round(float(servo_deg[i]), 2)
                          for i, n in enumerate(JOINT_NAMES)},
        }

    def move_to_xyz(self, target_xyz, timeout_s: float = 12.0) -> dict:
        """Solve IK in the model, then command the whole joint solution at once."""
        sol = self.solve_ik(target_xyz)
        if not sol["reachable"]:
            return {"moved": False, "reason": "target not reachable", **sol}

        move = self.goto_servo_angles(sol["servo_deg"], timeout_s=timeout_s)
        self._sync_mujoco_state()
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        reached = self.data.site_xpos[site].copy()
        target = np.asarray(target_xyz, dtype=float)
        return {
            "moved": True,
            "target_m": [round(float(v), 4) for v in target],
            "reached_m": [round(float(v), 4) for v in reached],
            "error_mm": round(float(np.linalg.norm(reached - target)) * 1000, 1),
            "ik_residual_mm": sol["residual_mm"],
            "joint_residual_deg": move["residual_deg"],
            "stalled": move["stalled"],
        }

    # ── grasp-primitive support ───────────────────────────────────────────────
    #
    # The grasp primitive was written against the simulator and asks it questions
    # only a simulator can answer: which finger pad is touching the object, and
    # whether the grip survives a physics lookahead. Neither exists on hardware.
    # What follows answers with the real signals available, and is explicit about
    # where the answer is coarser than the simulator's — inventing per-pad data
    # would make grasp's decisions look informed when they are not.

    def jaw_gap_mm(self) -> float:
        """Free gap between the finger pads, in mm, at the MEASURED jaw opening.

        Jaw geometry belongs to the ARM, not to the object, and the MuJoCo model
        describes the same printed gripper. So this poses a scratch copy of the
        model at the opening the gripper servo actually reports and measures pad
        to pad, exactly as the simulation does. No object truth is consulted.
        """
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:5] = self.data.qpos[:5]

        lo, hi = self.model.jnt_range[5]
        pct = float(np.clip(self._read_gripper_pct(), 0.0, 100.0))
        scratch.qpos[5] = lo + (hi - lo) * pct / 100.0
        mujoco.mj_forward(self.model, scratch)

        static_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      "static_finger_pad")
        moving_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      "moving_finger_pad")
        if static_id < 0 or moving_id < 0:
            return 0.0
        d = scratch.geom_xpos[moving_id] - scratch.geom_xpos[static_id]
        inplane = float(np.hypot(d[0], d[1]))          # ignore the moving jaw's arc
        return max(0.0, (inplane - PAD_THICKNESS_M) * 1000.0)

    def pad_contacts(self) -> tuple[bool, bool]:
        """(static_pad_touching, moving_pad_touching).

        There is no per-pad sensor on this arm. The only contact evidence is the
        gripper as a whole refusing to close, so BOTH entries carry the same
        aggregate answer. A one-pad graze — which the simulator can see and which
        grasp uses to decide whether to re-seat — is INVISIBLE here, so grasp will
        treat an edge catch as a proper bracket. That is a real loss of fidelity,
        not a shim, and it is why check_grip's quality reads are coarser on
        hardware than in simulation.
        """
        grasped = self.is_grasping()
        return grasped, grasped

    def grip_metrics(self, settle_steps: int = 6) -> dict:
        """Confidence read for check_grip, using measurements rather than physics.

        The simulator answers `stable` by stepping a lookahead and rewinding. A
        real arm cannot rewind, so stability is instead answered honestly: hold
        the current command and RE-READ the gripper over a short window. If the
        jaws are still held apart at the end, the grip survived; the object is not
        disturbed because nothing is commanded to move.
        """
        static_pad, moving_pad = self.pad_contacts()
        gap_mm = self.jaw_gap_mm()

        holds = 0
        n = max(1, int(settle_steps))
        for _ in range(n):
            time.sleep(GRIP_SETTLE_INTERVAL_S)
            if self.is_grasping():
                holds += 1
        stable = holds >= n - 1          # tolerate a single flickering read

        # Object width is INFERRED from where the jaws stalled, not looked up.
        # Without a contact position there is nothing better available, and
        # inspect_object's 30 mm default would be a fabricated number.
        width_mm = gap_mm if (static_pad or moving_pad) else 0.0

        return {
            "is_grasping": self.is_grasping(),
            "both_pads": static_pad and moving_pad,
            "static_pad": static_pad,
            "moving_pad": moving_pad,
            "jaw_gap_mm": round(gap_mm, 2),
            "object_width_mm": round(width_mm, 2),
            "stable": stable,
            "per_pad_sensing": False,     # tells callers the pads are an aggregate
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
        actual_pct = self._read_gripper_pct()
        if self._last_gripper_cmd_pct > actual_pct - GRIP_STALL_MARGIN_PCT:
            return False  # not pressing tighter than it already is
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

    def wrist_camera_pose(self) -> dict:
        """The wrist camera's pose in BASE coordinates, from forward kinematics.

        The camera is rigid to the gripper, so once the joint offsets are known
        FK gives its position and orientation for nothing — the hardware
        equivalent of the simulation's fixed `side_cam_params`. This is what lets
        a single detected pixel become a ray in the robot's own frame.

        The mounting transform still comes from the MuJoCo model's `wrist_cam`,
        which was set up for a different camera part, so treat the pose as good
        to a few millimetres and degrees rather than exact.
        """
        self._sync_mujoco_state()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        if cam_id < 0:
            raise RuntimeError("model has no camera named 'wrist_cam'")
        return {
            "position_m": [round(float(v), 5) for v in self.data.cam_xpos[cam_id]],
            "rotation": [[round(float(v), 6) for v in row]
                         for row in self.data.cam_xmat[cam_id].reshape(3, 3)],
        }

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

    def set_torque(self, enabled: bool, joints: Optional[list[str]] = None) -> dict:
        """Energise or release servos, all of them or a named subset.

        Releasing lets the arm be moved BY HAND, which is how a known-good pose
        gets captured without commanding a motion whose direction is not yet
        known. It also means the arm is no longer holding itself up: it WILL sag
        under gravity the instant torque drops, so it must be supported first.

        The subset matters for hand-guided grasping: release the five arm joints
        so the pose can be set by hand, but leave the GRIPPER energised so the
        jaws can still be commanded shut on the object once it is in place.
        """
        targets = joints if joints else ALL_AXES
        unknown = set(targets) - set(ALL_AXES)
        if unknown:
            raise ValueError(f"unknown joints: {sorted(unknown)}; valid: {ALL_AXES}")

        if enabled:
            self.robot.bus.enable_torque(list(targets))
        else:
            self.robot.bus.disable_torque(list(targets))
        if not joints:
            self._torque_enabled = enabled

        pose = self._read_all()
        return {
            "torque_enabled": enabled,
            "affected_joints": list(targets),
            "joint_angles_deg": {k: round(v, 2) for k, v in pose.items()},
            "note": ("Released — SUPPORT THE ARM, it will sag."
                     if not enabled else "Holding at the current pose."),
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
