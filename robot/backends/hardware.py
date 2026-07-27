"""Real SO-101 hardware backend (Feetech servo bus via lerobot).

Install lerobot first:  pip install lerobot
Run calibration first:  python -m lerobot.scripts.control_robot calibrate ...
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import mujoco

from robot.interface import RobotBackend
from robot.state import RobotState

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class HardwareBackend(RobotBackend):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 port: str = "/dev/ttyUSB0", calibration: str = "calibration.json"):
        self.model = model
        self.data = data
        self.port = port

        try:
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
            self.bus = FeetechMotorsBus(
                port=port,
                motors={name: (i, "sts3215") for i, name in enumerate(JOINT_NAMES + ["gripper"])},
            )
            self.bus.connect()
            self._load_calibration(calibration)
        except ImportError:
            raise RuntimeError("lerobot is not installed. Run: pip install lerobot")

        # Track last known object positions (updated via reset() or vision)
        self._cube_pos = np.array([0.20, 0.05, 0.015])
        self._container_pos = np.array([0.20, 0.30, 0.0])

    def _load_calibration(self, path: str) -> None:
        import json
        try:
            with open(path) as f:
                self.bus.set_calibration(json.load(f))
        except FileNotFoundError:
            raise RuntimeError(
                f"Calibration file '{path}' not found. "
                "Run calibration before using the hardware backend."
            )

    def _sync_mujoco_state(self) -> None:
        """Read real servo positions and sync MuJoCo data so IK stays accurate."""
        angles_deg = self.bus.read("Present_Position", JOINT_NAMES)
        self.data.qpos[:5] = np.radians(angles_deg)
        mujoco.mj_forward(self.model, self.data)

    # ── RobotBackend interface ────────────────────────────────────────────────

    def reset(self, cube_pos: Optional[np.ndarray] = None,
              container_pos: Optional[np.ndarray] = None) -> None:
        if cube_pos is not None:
            self._cube_pos = cube_pos.copy()
        if container_pos is not None:
            self._container_pos = container_pos.copy()
        home_deg = [0.0, 0.0, 0.0, 90.0, 90.0]
        self.bus.write("Goal_Position", home_deg, JOINT_NAMES)

    def apply_control(self, ctrl: np.ndarray) -> None:
        angles_deg = np.degrees(ctrl[:5]).tolist()
        self.bus.write("Goal_Position", angles_deg, JOINT_NAMES)
        r = self.model.actuator_ctrlrange[5]
        gripper_pct = (ctrl[5] - r[0]) / (r[1] - r[0]) * 100
        # Map 0–100% to gripper servo range (adjust limits to your hardware)
        gripper_deg = float(np.interp(gripper_pct, [0, 100], [0, 100]))
        self.bus.write("Goal_Position", [gripper_deg], ["gripper"])

    def step(self) -> None:
        self._sync_mujoco_state()

    def move_gripper(self, openness_pct: float, steps: int = 120) -> None:
        gripper_deg = float(np.interp(openness_pct, [0, 100], [0, 100]))
        self.bus.write("Goal_Position", [gripper_deg], ["gripper"])

    def get_state(self) -> RobotState:
        self._sync_mujoco_state()
        angles_deg = {name: round(float(np.degrees(self.data.qpos[i])), 2)
                      for i, name in enumerate(JOINT_NAMES)}
        ee = self.data.site_xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        ].copy()
        gripper_raw = self.bus.read("Present_Position", ["gripper"])[0]
        gripper_pct = float(np.interp(gripper_raw, [0, 100], [0, 100]))
        return RobotState(
            joint_angles_deg=angles_deg,
            end_effector_m={ax: round(float(v), 4) for ax, v in zip("xyz", ee)},
            cube_position_m={ax: round(float(v), 4) for ax, v in zip("xyz", self._cube_pos)},
            container_position_m={ax: round(float(v), 4) for ax, v in zip("xyz", self._container_pos)},
            is_grasping=self.is_grasping(),
            cube_in_container=False,  # requires vision; left for integrator to implement
            gripper_openness_pct=round(gripper_pct, 1),
        )

    def is_grasping(self) -> bool:
        # No contact sensors — infer from gripper current draw if available
        # For now returns False; integrate a force sensor or current threshold here
        return False

    def get_object_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cube_pos.copy(), self._container_pos.copy()

    def inspect_object(self) -> dict:
        # Hardware has no model-geometry perception; real object dims need a vision
        # pipeline. Return the cube default so grasp() falls back to its tuned values.
        return {
            "shape": "unknown",
            "width_mm": 30.0,
            "height_mm": 30.0,
            "footprint_radius_mm": 15.0,
            "center_m": [round(float(v), 4) for v in self._cube_pos],
            "upright": True,
        }

    def render(self) -> list[np.ndarray]:
        # Connect a USB camera here via OpenCV if desired:
        # import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); ...
        return []

    def on_pick_and_place_start(self) -> None:
        pass
