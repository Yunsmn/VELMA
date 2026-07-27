from __future__ import annotations
from dataclasses import dataclass, asdict, field


@dataclass
class RobotState:
    joint_angles_deg: dict
    end_effector_m: dict
    cube_position_m: dict
    container_position_m: dict
    is_grasping: bool
    cube_in_container: bool
    gripper_openness_pct: float
    # All manipulable objects on the table (name/color/position). Empty on
    # single-object/hardware backends; populated by the multi-object sim backend.
    objects: list = field(default_factory=list)
    # True if the wall obstacle was pushed off its rest pose (a real collision);
    # an incidental graze that leaves it in place does not set it.
    obstacle_hit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MoveResult:
    success: bool
    message: str
    state: RobotState

    def to_json(self) -> dict:
        return {
            "status": "success" if self.success else "warning",
            "message": self.message,
            "robot_state": self.state.to_dict(),
        }
