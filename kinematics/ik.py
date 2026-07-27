"""Damped least-squares IK controller using MuJoCo's Jacobian."""
from __future__ import annotations
import mujoco
import numpy as np


class IKController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 end_effector_site: str = "gripperframe",
                 damping: float = 0.1, max_dq: float = 0.5):
        self.model = model
        self.data = data
        self.damping = damping
        self.max_dq = max_dq
        self.n_arm = 5

        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{end_effector_site}' not found in model")

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def get_ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def step_toward_target(self, target_pos: np.ndarray, gripper_action: float = 0.0,
                           gain: float = 1.0,
                           locked_joints: list[int] | None = None) -> np.ndarray:
        """Compute one IK step toward target. Returns full control vector (6,)."""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.ee_site_id)

        active = [i for i in range(self.n_arm) if locked_joints is None or i not in locked_joints]
        J = self._jacp[:, active]
        error = target_pos - self.get_ee_position()

        JTJ = J.T @ J
        try:
            dq_active = np.linalg.solve(JTJ + self.damping ** 2 * np.eye(len(active)), J.T @ error)
        except np.linalg.LinAlgError:
            dq_active = np.linalg.pinv(J) @ error

        dq_active = np.clip(dq_active, -self.max_dq, self.max_dq) * gain

        dq = np.zeros(self.n_arm)
        for i, j in enumerate(active):
            dq[j] = dq_active[i]

        current_q = self.data.qpos[:self.n_arm].copy()
        target_q = current_q + dq
        for i in range(self.n_arm):
            lo, hi = self.model.jnt_range[i]
            if lo != hi:
                target_q[i] = np.clip(target_q[i], lo, hi)

        ctrl = np.zeros(self.model.nu)
        ctrl[:self.n_arm] = target_q
        r = self.model.actuator_ctrlrange[5]
        ctrl[5] = (gripper_action + 1) / 2 * (r[1] - r[0]) + r[0]
        return ctrl
