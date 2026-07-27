"""The arm's reach envelope — a fact about the ROBOT, used to gate perception candidates.

This exists to replace a gate that looked like a robot fact but was not. The old test was

    WS_X_MIN < x < WS_X_MAX and abs(y) < WS_Y_ABS and r < WS_R_MAX     # 0.15..0.34

an axis-aligned box in front of the base. Nothing about the SO-101 is box-shaped: it has one
azimuthal joint (`shoulder_pan`) and a radial extent, so its workspace is an ANNULAR SECTOR.
The box both over-rejects (a point at x=0.05, y=0.28 is azimuth 80 deg and comfortably reachable,
but fails x > 0.15) and encodes a scene the constants were tuned on rather than the machine.

MEASURED, not assumed. `experiments/scratch/measure_reach_clean.py` drives the real IK over a
grid of (azimuth, radius) at several heights, with the cube and container parked far away, and
records where the end effector settles within 15 mm:

    z = 0.03    azimuth -105..+105 deg     r 0.06 .. 0.36 m
    z = 0.10    azimuth -105..+105 deg     r 0.10 .. 0.38 m

The azimuth span matches `shoulder_pan`'s +-110 deg limit (the +-110 samples fail on IK
tolerance, +-105 is the last that holds), which is the cross-check that this is kinematics and
not an artifact. Measuring it with the objects still in the scene gave r_max 0.20-0.25 at
azimuths -50..-70 — that was the container physically blocking the arm, an obstacle, not a
joint limit. Hence the obstacle-free run.

The bounds below are the measured envelope with margin. Margin is deliberate: this gate runs on
a COARSE stage whose job is to aim the wrist camera, so wrongly discarding a real object costs
far more than admitting one the arm may not quite reach.
"""
from __future__ import annotations

import math
from typing import Optional

# Measured envelope, widened for margin. R_MIN excludes the arm's own base column; R_MAX sits
# just past the furthest the IK held (0.38).
R_MIN, R_MAX = 0.05, 0.42

# Vertical band. Generous on purpose so an object standing on top of another still passes —
# this is the bound that used to be a table-plane assumption.
Z_MIN, Z_MAX = -0.02, 0.30

# Fallback azimuth half-span, in degrees, when the joint limit cannot be read from a model.
# `shoulder_pan` is +-1.91986 rad = +-110 deg.
AZ_MAX_DEG_DEFAULT = 110.0

_PAN_JOINT = "shoulder_pan"


def pan_limit_deg(model=None) -> float:
    """Azimuth half-span the base can turn through, read from the robot when possible.

    Reading it from `model` keeps the gate honest on hardware with a different pan range;
    the constant is only a fallback for callers that have no model handy.
    """
    if model is None:
        return AZ_MAX_DEG_DEFAULT
    try:
        import mujoco
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _PAN_JOINT)
        if jid < 0:
            return AZ_MAX_DEG_DEFAULT
        lo, hi = model.jnt_range[jid]
        return float(math.degrees(min(abs(lo), abs(hi))))
    except Exception:
        return AZ_MAX_DEG_DEFAULT


def reachable(P, model=None, az_max_deg: Optional[float] = None) -> bool:
    """Is 3D point `P` somewhere this arm could act — annular sector, not a box."""
    x, y, z = float(P[0]), float(P[1]), float(P[2])
    if not (Z_MIN < z < Z_MAX):
        return False
    r = math.hypot(x, y)
    if not (R_MIN < r < R_MAX):
        return False
    az = abs(math.degrees(math.atan2(y, x)))
    return az <= (az_max_deg if az_max_deg is not None else pan_limit_deg(model))
