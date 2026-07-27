from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from robot.state import RobotState


class RobotBackend(ABC):
    """Contract every backend must satisfy.

    Simulation backend: uses MuJoCo physics.
    Hardware backend: uses real servos; MuJoCo model is used for kinematics only.
    """

    @abstractmethod
    def reset(self, cube_pos: Optional[np.ndarray] = None,
              container_pos: Optional[np.ndarray] = None) -> None: ...

    @abstractmethod
    def apply_control(self, ctrl: np.ndarray) -> None:
        """Apply a 6-element IK control vector: [joint_angles_rad x5, gripper_ctrl]."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance one physics step (simulation) or sync servo state (hardware)."""
        ...

    @abstractmethod
    def move_gripper(self, openness_pct: float, steps: int = 120) -> None:
        """Move gripper to target openness while holding arm joints fixed."""
        ...

    @abstractmethod
    def get_state(self) -> RobotState: ...

    @abstractmethod
    def is_grasping(self) -> bool: ...

    @abstractmethod
    def get_object_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cube_xyz, container_xyz) in world metres."""
        ...

    @abstractmethod
    def inspect_object(self) -> dict:
        """Returns the target object's geometry read from the model:
        {shape, width_mm, height_mm, footprint_radius_mm, center_m:[x,y,z], upright}.
        """
        ...

    # ── Multi-object perception (concrete defaults; the sim backend overrides) ──
    def list_objects(self) -> dict:
        """Every manipulable object on the table + static obstacles. The default is
        a single-object view; multi-object backends override it."""
        return {
            "objects": [{**self.inspect_object(), "name": "object", "color": "unknown"}],
            "obstacles": [],
            "active_object": "object",
        }

    def set_active_object_by_point(self, x: float, y: float) -> str:
        """Rebind the active target to the object nearest (x, y). Single-object
        backends have nothing to rebind; multi-object backends override."""
        return "object"

    @abstractmethod
    def render(self) -> list[np.ndarray]:
        """Returns RGB camera frames. Empty list if unavailable."""
        ...

    @abstractmethod
    def on_pick_and_place_start(self) -> None:
        """Hook called once at the start of run_pick_and_place for backend setup."""
        ...
