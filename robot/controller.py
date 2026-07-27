"""RobotController — backend-agnostic motion logic."""
from __future__ import annotations
import math
import threading
from typing import Optional

import numpy as np

from robot.interface import RobotBackend
from robot.state import MoveResult, RobotState
from kinematics.ik import IKController

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_LOCKED_WRIST = [3, 4]
_LOCKED_PAN_AND_WRIST = [0, 3, 4]  # for arm-frame forward/up IK (only shoulder_lift + elbow_flex active)

_GRIPPER_OPEN   = 0.5
_GRIPPER_CLOSED = -1.0
_FINGER_Y_OFFSET = -0.015
_APPROACH_Z      = 0.04
_GRASP_Z_OFFSET  = 0.005
_TIGHTEN         = 0.4
_LIFT_Z          = 0.15
_ABOVE_CTR_Z     = 0.17
_PLACE_Z         = 0.03
# ── place carry (Task #21) ────────────────────────────────────────────────────
# The place traverse to above-the-container is the long diagonal that a marginal
# +y/shallow grip must survive. A single fast _move (gain 0.5, tol 0.025) SHEARS a
# firmly-held +y cube and FLINGS it outward (measured: (0.24,+0.04) ended at
# x~0.48, far past the container). Carrying in fine interpolated sub-steps at a low
# gain — exactly the gentle staged lift that holds the grasp at carry height — keeps
# the cube tracking the EE the whole traverse (verified: it then places cleanly).
# The near/-y cells that already place fine are unaffected: the staged carry reaches
# the same above-container point, just without the fling.
_PLACE_CARRY_SUBSTEPS = 6      # interpolated rungs across the horizontal traverse
_PLACE_CARRY_GAIN     = 0.2    # gentle IK gain per rung (no fling-inducing shove)
_PLACE_CARRY_SETTLE   = 250    # extra hold steps at the top rung so the move converges
_PLACE_DROP_HEIGHT    = 0.105  # release this far ABOVE the place point (rim clearance) — drop, don't lower inside
_PLACE_DROP_MIN_Z     = 0.15   # never open the jaws below this world Z (keeps fingers above the bin walls)

# ── grasp() constants — ported verbatim from feedback_pick.py (validated 97/94) ──
# DO NOT tune these; they are the calibrated gentle-grasp parameters.
_G_NEG_WF, _G_POS_WF = 49.0, 30.0          # basin pre-position wrist_flex (-y / +y)
# +y INNER wrist_flex (RADIUS-GATED, used only for short-radius +y cells, see grasp()):
# the shallow wf=30 descent let the STATIC finger arc inward-and-down and clip the cube's
# outer (+x) top edge at SHORT radius — the cube's +x face sits right in the finger's
# descent arc when r<~0.21 — nudging the cube past the 3.5mm descend drift-abort BEFORE
# the close could bracket it, so the inner +y column (e.g. (0.196,+0.01),(0.197,+0.04))
# drift-aborted on contact. A steeper wf=42 brings the fingers down more vertically so
# they clear the cube during descent and bracket it cleanly on the close. This is gated
# to SHORT radius (_G_POS_INNER_R) so the validated longer-radius +y column keeps its
# wf=30 basin untouched (steeper wf shifts the longer-radius close basin and would
# regress those cells). Measured: short-radius +y benchmark cells fixed, grid unchanged.
_G_POS_WF_INNER = 42.0       # steeper +y wrist_flex for the inner (short-radius) +y clip
_G_POS_INNER_R  = 0.21       # +y cube radius below which the inner (steeper) wf is used
_G_POS_STEEP_PAN = 20.0      # +y |pan_deg| above which the steeper wf is also used (the
                             # off-axis +y close basin only seats both pads under wf=42)
# MID-radius, MID-pan +y pocket: cell 66 (0.227,+0.046) r0.232 pan-11.5 sits in a gap —
# not short-radius (r>0.21), not high-pan (pan<20) — so it took wf=30, at which the seed
# lat=-0.010 leaves the jaw 9deg off-parallel and the SEAT rung one-pad-shoves the cube
# ~10mm -> a seat drift-abort that (by the never-bat rule) ENDS the grasp before the retry
# sweep can find a seating seed. wf=42 seats both pads (jaw_off 9->4.8deg, seat shove 10->
# 1.7mm, firm first-try). This pocket (r in [MID_RMIN, STEEP_RMAX), |pan|>MID_PAN) captures
# cell 66 and its mid-pan neighbours (57/58/52/67, all already firm at wf=42) while the
# r<MID_RMIN low-pan column (e.g. cell 25 r0.224 pan-10, which wf=42 would regress) keeps
# wf=30.
_G_POS_MID_PAN  = 10.5       # +y |pan_deg| above which (with mid radius) the steeper wf is used
_G_POS_MID_RMIN = 0.228      # +y radius at/above which the mid-pan steeper-wf pocket applies
_G_POS_STEEP_RMAX = 0.26     # ...but ONLY out to here: past ~0.26 the arm is near its +y
                             # reach limit and the steeper wf makes a far high-pan cell
                             # ENGAGE-and-bat (measured: r0.269 pan-38 flung the cube
                             # 31.5mm) instead of declining cleanly. Those far cells stay
                             # at wf=30, where they decline gently (0mm) — a clean fail is
                             # always better than a bat. Mid-radius high-pan (55,56 ~0.255)
                             # stays inside and grasps cleanly.
_G_PRE_SL, _G_PRE_EF = -6.0, 20.0          # basin pre-position shoulder_lift / elbow_flex
_G_SAFE_DZ = 0.045          # above-cube safe height for the initial approach (open)
_G_FINGER_LAT_NEG = -0.014  # best single seed for the -y close basin (30mm cube)
_G_FINGER_LAT_POS = -0.010  # +y close basin: -0.010 grasps the steep +0.04 edge row
                            # FIRMLY first-try (the old -0.014 missed it with a 15mm
                            # close-shove, then walked the cube past budget over the
                            # sweep). -0.010 grasps the rest of the +y column too.
_G_FINGER_LAT_POS_INNER = -0.013  # near-axis short-radius +y seed (r<_G_POS_INNER_R,
                            # |pan|<_G_POS_LOWPAN, e.g. cells 17 & 20): the base -0.010
                            # target sits in the descending finger's arc for the y~+0.02
                            # row -> the OPEN jaw brushes the cube and the descend
                            # drift-aborts before contact. -0.013 clears that arc (approach
                            # 0.0mm, clean both-pad close ~4mm; measured). Drift-free (a
                            # lateral-target shift, no added motion).
_G_POS_LOWPAN = 9.0         # |pan_deg| below which (with r<_G_POS_INNER_R) the near-axis
                            # inner +y seed above is used. Captures cells 17 (pan-6.7) and
                            # 20 (pan-2.9); excludes r<0.21 |pan|>9 cells and ALL grid cells.
# Width-adaptive seed: the close basin centers on the object's near face, so a
# NARROWER object needs the static pad to reach ~half-the-width-difference further
# in (more-negative finger_lat). Measured: 30mm cube grasps at -0.014; 20mm box
# grasps at -0.023 (Δlat ≈ -0.0009 per mm narrower than 30mm). Objects >=30mm keep
# the validated -0.014 seed (their basin is wide and the seed already sits inside).
_G_LAT_WIDTH_REF_MM = 30.0   # reference width (the validated cube) at the base seed
_G_LAT_PER_MM = 0.0009       # extra -lat per mm narrower than the reference
# ── NEVER-BAT wide-box early decline (P0) ─────────────────────────────────────
# A SHORT, WIDE box gripped at its LOW center anchor cannot be center-gripped by the
# flat pads without batting: the off-center pad lands mid-face on the wide top and the
# seating rung LEVERS the box sideways. This is strongly POSITION-DEPENDENT — verified
# across the -y (0.24,-0.13) AND +y (0.22,0.10) cells with peak (not just final) drift:
#   * a short box at the CUBE width (~30mm) is the only safe one: the 30mm cube peaks
#     12.4mm (-y) / 14.6mm (+y) — under the 15mm ceiling, but only just.
#   * a 40mm short box (e.g. the 40x20 rect) peaks 15.1-15.6mm — OVER the ceiling, a bat.
#   * a 48-60mm short box is flung 24-83mm in one seat rung.
# So a SHORT box wider than the cube band is DECLINED UP FRONT (0mm displacement). The
# threshold sits just above the 30mm cube so the validated cube grasp is byte-identical
# and unaffected, while every wider short box (which the flat pads cannot hold without a
# bat at some cell) is declined gently. This fires ONLY when grasp_z is None (the
# low-center grip of a SHORT object): a TALL wide box (e.g. the 40mm cube, gripped high
# on its upper body) is unaffected and still grasps cleanly.
_G_WIDE_DECLINE_MM = 32.0    # SHORT (low-center-grip) box wider than this -> decline.
# ── NEVER-BAT narrow-box early decline (P0) ───────────────────────────────────
# A box NARROWER than the validated cube cannot be grasped without batting: the
# flat moving pad must travel from the (wide) pre-open gap onto a small object, and
# that close-shove is fundamentally 12-19mm for a sub-30mm box — at or OVER the 15mm
# ceiling. Measured (square short box, real derived params, BOTH cells): the close
# either drift-aborts (a gentle ~11mm decline) OR firm-grasps at 12-19mm drift (a
# bat). There is NO seed that latches gently: every firm seed shoves past budget,
# every gentle seed misses. The 30mm cube itself peaks at 14.6mm at the +y cell —
# right at the edge — so anything narrower is over. So a box narrower than the cube
# is DECLINED UP FRONT WITHOUT touching it (0mm). The threshold sits just below the
# 30mm cube so the validated cube grasp is byte-identical and unaffected.
_G_NARROW_DECLINE_MM = 29.0  # box narrower than this -> decline (keeps the 30mm cube).
_G_GRASP_DZ = 0.000         # ee reaches z~0.026 here (fingers bracket the cube top)
_G_SEAT_DZ = -0.004         # final rung; ee~0.022, fingers around the cube (gentle)
_G_SEAT_GAIN = 0.35         # gentle final-rung IK gain (less shove if lat is off)
_G_DRIFT_ABORT = 0.010      # cube drift that means we're bumping it -> abort & lift
_G_COMMIT_ON_GRIP = True    # once the jaws close ON the cube (any pad contact), CARRY it
                            # out — never re-seat/open/decline a held grip; only a genuine
                            # fall ends it. Pre-grip bumps + empty closes sweep the next
                            # seed WITHOUT re-homing (stay near the object).
# ── NEVER-BAT cumulative budget (the #1 constraint) ───────────────────────────
# The object must NEVER be displaced more than this from where it started — across
# the WHOLE grasp, including every failed/aborted attempt. The validated cube + box
# family grasp first-try with ZERO drift, so this never fires for them. A rolling
# round object (cylinder/sphere) moves on first contact; the instant cumulative
# drift crosses this budget the retry loop STOPS GENTLY (open, lift clear, decline)
# instead of chasing it — sweeping a roller just walks it off the table. Set well
# under the 15mm hard ceiling so we abort with margin to spare.
_G_BAT_BUDGET_MM = 22.0     # cumulative object migration from origin -> stop the sweep.
                            # Set well under the 15mm hard ceiling. Each one-pad close +
                            # seat-lift on a roller (e.g. cyl_short) drags the object ~5mm,
                            # and the between-attempt check fires only AFTER a full attempt.
                            # Measured cyl_short: attempt 0 -> 10.3mm, attempt 1's seat-lift
                            # peaks at 15.0mm (right at the ceiling). Lowering to 9mm trips
                            # the check after attempt 0 (10.3 > 9) so the sweep stops BEFORE
                            # the second drag, keeping the roller well under 15mm. The cube +
                            # box family grasp first-try with ZERO drift, so this never fires
                            # for them; only multi-attempt rollers reach it.
_G_CLOSE_DRIFT_ABORT = 0.016 # object push DURING the close (while still un-gripped) that
                            # aborts the close early — the NEVER-BAT cap, kept under the
                            # 15mm ceiling. Measured close-slide-to-latch: cube 4.4mm,
                            # 6cm box 4.9mm, short cyl 5.8mm, 4cm rect 10.6mm (all latch
                            # below this -> grasp). A 20-25mm CUBE slides 12-20mm to latch
                            # and then EJECTS sideways during carry (its tiny pad-contact
                            # area can't hold a lift) — capping the close here declines it
                            # at ~11mm (a gentle, on-table decline) instead of letting it
                            # latch-then-fling. A wide-short box / round object slides
                            # 20-27mm without ever latching — also stopped here.
# A single gentle attempt whose descent/seat DRIFT-ABORTED means the object moved on
# contact (it rolls/slips) — retrying just shoves it again. So a drift-abort ends the
# whole grasp gently rather than continuing the lateral sweep.
_G_LIFT_CLEAR_MM = 35.0
_G_CARRY_Z = 0.17
# Staged carry lift: lift to carry height in these EE-z rungs (a single fast carry
# shears a marginal grip on a small/short object, dropping it from height; the gentle
# staged lift holds it — verified for the 30mm cube, the 20mm cube, and 20mm-tall
# wide/rect boxes). Gain matches the validated single-carry gain (0.12) — going
# slower (0.10) actually lost the cube grip at the upper rung.
_G_CARRY_RUNGS = (0.09, 0.11, 0.13, 0.15, _G_CARRY_Z)  # fine rungs: a grip that fails is
                            # caught within ~2cm of lift, so a slipped object falls only a
                            # short way (small scatter) instead of dropping from carry.
_G_CARRY_GAIN = 0.12
_G_SETDOWN_Z = 0.05         # EE height to lower a slipped object to before releasing
_G_TALL_CARRY_H_MM = 28.0   # objects this tall+ carry best in one move; shorter -> staged
_G_GATE_Z = 0.075           # low stability-gate height: prove the grip here before lifting
                            # high, so a marginal grip fails LOW (small scatter) not at carry
# HEIGHT-TRACKING carry guard (Task #21): a cube genuinely held by the jaws tracks the
# EE the whole lift — it sits ~6-9mm below the grasp frame and rises WITH it, EVEN WHILE
# is_grasping flickers False (the pad contact toggles during the dynamic lift but the
# cube never leaves the jaws). Measured: d(ee_z - cube_z) ≈ 0.007m held, on every rung,
# firm and "marginal" cells alike. A real drop shows the cube FALLING far behind the
# rising EE (d balloons past this). So the carry's drop test is: the cube is lost ONLY
# if it has fallen this far below the EE — is_grasping flicker alone is NOT a drop. This
# is what was losing the firm-but-flickering shallow -y cells (0.20/0.24,-0.08) mid-carry.
_G_CARRY_DROP_GAP = 0.045   # ee_z - cube_z (m) beyond which the cube has fallen out
_G_TEST_LIFT_Z = 0.060      # quick verification lift after close
_G_PREOPEN_PCT = 65.0       # jaw openness for approach (cube default)
# ── jaw geometry (measured empirically from the pad y-separation, see jaw_results.txt)
# The jaws open along the grasp axis; pad center y-gap is linear in openness pct:
#   pad_y_gap_mm = _JAW_SLOPE * pct + _JAW_INTERCEPT
# Pads are 2.5mm thick on the gap axis, so the FREE object width = pad_y_gap - 2.5.
# Inverting gives the pct needed to pre-open the jaws to (object_width + margin).
_JAW_SLOPE = 1.0694          # mm of pad y-gap per % openness
_JAW_INTERCEPT = 13.882      # pad y-gap (mm) at pct=0
_JAW_PAD_THICK = 2.50        # total pad thickness on the gap axis (mm)
_JAW_MAX_WIDTH = 103.0       # max free graspable width at 100% (mm); wider can't fit
# Clean parallel-grasp band: the moving jaw arcs UP as it opens, so cap the pre-open
# openness used for approach. Past ~70% the moving pad is >35mm higher than the static
# pad and the grip is no longer a clean parallel bracket.
_JAW_PREOPEN_MAX_PCT = 75.0  # floor for pre-open is the cube default _G_PREOPEN_PCT
# Sweep FINGER_LAT around the seed: the close basin is ~2mm wide and varies with
# position; try the most-likely values first, then bracket. +y needs a wider sweep.
# -y: base seed -0.014 first (validated), then bracket BOTH ways. The far short-radius
# -y cells (cx=0.26 / -0.08, 0.24/-0.12) seat both pads only at a LESS-negative seed
# (-0.010 / -0.008), so the sweep also reaches those (steps +0.004, +0.006). The
# original -0.003/-0.006 (more-negative) side is kept first after the base so the
# validated near cells are unchanged. (Firm-both-pads seeds vary non-monotonically.)
_G_RETRY_LAT_STEP_NEG = [0.0, -0.003, +0.004, -0.006, +0.006]
_G_RETRY_LAT_STEP_POS = [0.0, -0.003, -0.006, +0.003, -0.009]

# ── check_grip / confidence-driven retry (Task #21) ───────────────────────────
# The CONFIDENCE READ classifies the current grip without moving the arm so the
# loop can decide: firm -> carry straight away (no re-verify motion that shakes a
# good grip loose); marginal -> a TINY local re-seat (open a little, shift a few
# mm laterally at the CURRENT height, re-close) instead of the far re-home; empty
# -> the jaws closed on nothing (a clean miss) so re-seat too.
_G_GRIP_SETTLE_STEPS = 6     # sim frames to confirm the contact isn't a 1-frame flicker
_G_FIRM_GAP_TIGHT_MM = 6.0   # |jaw_gap - object_width| within this AND both pads ->
                             # FIRM regardless of the post-close settle flicker. A gap
                             # that brackets the object to within a few mm of its width
                             # is a face-to-face hold; the flicker while it is still
                             # table-supported is not a drop (OWNER complaint #1).
                             # Measured: good cube grips read gap_err 0 to -9mm; the
                             # near-perfect ones (err 0..-3) latch firm immediately.
_G_FIRM_GAP_TOL_MM = 12.0    # the LOOSER bracket band: |gap_err| within this AND both
                             # pads AND stable -> FIRM. Accepts the soft-pad compression
                             # cases (gap_err down to -9mm) once the settle confirms the
                             # hold, while staying clear of the -12mm EMPTY band.
_G_EMPTY_GAP_MM = 12.0       # jaw closed this many mm PAST the object width AND not
                             # grasping -> EMPTY (pads met near-shut, object missed).
# TINY-NUDGE re-seat: instead of _grasp_lift_clear + _grasp_rehome + _grasp_prepos
# (which travels far and shakes the object), a marginal/empty close re-seats LOCALLY:
# open slightly, shift laterally by a few mm at the current height, re-close. The
# whole nudge stays well under the never-bat budget.
_G_NUDGE_OPEN_PCT = 55.0     # partial open for the re-seat (not the full 65% approach)
_G_NUDGE_LAT_MM = 3.0        # lateral shift per re-seat step (mm); <= 4mm budget-safe
_G_NUDGE_LIFT_MM = 6.0       # tiny vertical lift before the lateral shift so the open
                             # jaw clears the object top instead of dragging across it
_G_NUDGE_GAIN = 0.3          # gentle gain for the re-seat moves (no shove)
_G_NUDGE_MAX = 3             # bounded local re-seats before declining gently


class RobotController:
    def __init__(self, backend: RobotBackend, ik: IKController):
        self.backend = backend
        self.ik = ik
        self._lock = threading.Lock()
        self._grasp_gripper: float = _GRIPPER_CLOSED
        self._carry_log = lambda _m: None  # set per-grasp to the active log sink

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ga(self, pct: float) -> float:
        return float(np.clip((pct / 100.0) * 2.0 - 1.0, -1.0, 1.0))

    def _step(self, target: np.ndarray, ga: float, gain: float = 0.5,
              locked: list[int] | None = _LOCKED_WRIST) -> None:
        ctrl = self.ik.step_toward_target(target, gripper_action=ga, gain=gain, locked_joints=locked)
        self.backend.apply_control(ctrl)
        self.backend.step()

    def _move(self, target: np.ndarray, ga: float, max_steps: int = 300,
              tol: float = 0.012, gain: float = 0.5,
              locked: list[int] | None = _LOCKED_WRIST) -> None:
        for _ in range(max_steps):
            self._step(target, ga, gain=gain, locked=locked)
            if np.linalg.norm(target - self.ik.get_ee_position()) < tol:
                break

    # ── Public API (called by MCP server) ─────────────────────────────────────

    def get_state(self) -> MoveResult:
        return MoveResult(True, "Current state", self.backend.get_state())

    def inspect_object(self) -> dict:
        """Perception: the target object's geometry, read from the model.

        Returns {shape, width_mm, height_mm, footprint_radius_mm, center_m, upright}.
        The LLM uses this to choose grasp(grip_width_mm, grasp_height_m) for any shape.
        """
        return self.backend.inspect_object()

    def list_objects(self) -> dict:
        """Perception: every manipulable object on the table + static obstacles, so
        the agent can survey a multi-object scene and choose what to act on."""
        return self.backend.list_objects()

    def check_grip(self, settle_steps: int = _G_GRIP_SETTLE_STEPS) -> dict:
        """Confidence read of the CURRENT grip WITHOUT moving the arm.

        Combines three signals (the arm ctrl is held fixed, so this never disturbs
        a good grip — it only settles a few sim steps and reads):
          1. both finger pads in contact with the object (the is_grasping signal,
             decomposed into static/moving so a one-pad graze is distinguishable);
          2. the jaw closure vs the object width — the live jaw gap (pad-to-pad on
             the grasp axis) compared to the object's width. A grip that closed to
             ~the object width is solid; closed far PAST it (pads met near-empty)
             means the jaws missed; barely closed (gap >> width) means a marginal
             edge catch;
          3. a brief STABILITY check — step the sim a few frames holding ctrl and
             confirm the contact holds, so a single is_grasping flicker is NOT
             read as a drop (this is what lets the grasp loop keep a good grip).

        Returns {quality: "firm"|"marginal"|"empty", is_grasping, both_pads,
        jaw_gap_mm, object_width_mm, stable}. quality:
          - "firm":     both pads + stable + jaw gap brackets the object width
                        (closed to within tolerance of the object's width).
          - "marginal": grasping but unstable, OR a one-pad/edge catch, OR the jaw
                        closed somewhat past/short of the object width — a hold that
                        a tiny re-seat could firm up.
          - "empty":    not grasping AND the jaws closed well past the object width
                        (the pads met with nothing between them — a clean miss).
        """
        m = self.backend.grip_metrics(settle_steps=settle_steps)
        gap = float(m["jaw_gap_mm"])
        width = float(m["object_width_mm"])
        both = bool(m["both_pads"])
        stable = bool(m["stable"])
        grasping = bool(m["is_grasping"])

        # how far the closed jaw sits from the object width (signed, mm):
        #   gap << width  -> jaws closed PAST the object (missed / pads near-empty)
        #   gap ~~ width  -> jaws bracket the object face-to-face (solid)
        #   gap >> width  -> jaws only caught an edge / barely closed (marginal)
        gap_err = gap - width

        # FIRMNESS: a jaw gap that brackets the object to within a few mm of its width
        # (|gap_err| small) with BOTH pads in contact is the strongest grip signal —
        # the pads are face-to-face on the object. That alone is firm, EVEN IF the
        # immediate post-close settle flickers: right after a close the object is still
        # table-supported and the contact toggles as the soft pads micro-seat, which is
        # exactly the flicker that must NOT be read as a drop (OWNER complaint #1). A
        # looser bracket needs the stability check to confirm it is a real hold.
        if both and abs(gap_err) <= _G_FIRM_GAP_TIGHT_MM:
            quality = "firm"
        elif both and stable and abs(gap_err) <= _G_FIRM_GAP_TOL_MM:
            quality = "firm"
        elif (not grasping) and gap_err <= -_G_EMPTY_GAP_MM:
            # not holding AND jaws closed well past the object: a clean empty miss
            quality = "empty"
        elif both or grasping:
            # holding something but not confidently firm -> a tiny re-seat may help
            quality = "marginal"
        else:
            # not holding and not a clear empty-close -> treat as marginal (the
            # descent/close didn't seat; a lateral re-seat is the right next move)
            quality = "marginal"

        return {
            "quality": quality,
            "is_grasping": grasping,
            "both_pads": both,
            "static_pad": bool(m["static_pad"]),
            "moving_pad": bool(m["moving_pad"]),
            "jaw_gap_mm": round(gap, 2),
            "object_width_mm": round(width, 2),
            "gap_err_mm": round(gap_err, 2),
            "stable": stable,
        }

    def reset(self, cube_pos: Optional[np.ndarray] = None,
              container_pos: Optional[np.ndarray] = None) -> MoveResult:
        self.backend.reset(cube_pos, container_pos)
        return MoveResult(True, "Scene reset", self.backend.get_state())

    def move_to_cartesian(self, x: float, y: float, z: float,
                          gripper_pct: Optional[float] = None,
                          lock_wrist: bool = True,
                          gain: float = 0.5) -> MoveResult:
        target = np.array([x, y, z])
        if gripper_pct is None:
            gripper_pct = self.backend.get_state().gripper_openness_pct
        ga = self._ga(gripper_pct)
        locked = _LOCKED_WRIST if lock_wrist else None
        with self._lock:
            # If the requested gripper openness differs materially from the current
            # actuator ctrl, the fingers need simulation time to physically settle,
            # so the early position-based return is suppressed for the first
            # `min_steps` steps. ctrl[5] still jumps straight to the target value
            # (set by step_toward_target) exactly as before: A/B testing showed the
            # +y close recipe (65->0 at an unreachable target) depends on the
            # immediate full-close command — a 300-step linear ramp weakened the
            # settled grasp and caused drops during carry.
            r = self.ik.model.actuator_ctrlrange[5]
            g_start = float(self.ik.data.ctrl[5])
            g_end = float((ga + 1) / 2 * (r[1] - r[0]) + r[0])
            min_steps = 300 if abs(g_end - g_start) > 0.02 * abs(r[1] - r[0]) else 0
            for step in range(400):
                ctrl = self.ik.step_toward_target(target, gripper_action=ga,
                                                   gain=gain, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()
                err = float(np.linalg.norm(target - self.ik.get_ee_position()))
                if err < 0.012 and step >= min_steps:
                    return MoveResult(True, f"Reached in {step+1} steps (err={err:.4f}m)",
                                      self.backend.get_state())
            final_err = float(np.linalg.norm(target - self.ik.get_ee_position()))
            return MoveResult(final_err < 0.05, f"Max steps — err={final_err:.4f}m",
                              self.backend.get_state())

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                      gripper_pct: Optional[float] = None,
                      lock_wrist: bool = True, gain: float = 0.5) -> MoveResult:
        ee = self.ik.get_ee_position()
        if gripper_pct is None:
            state = self.backend.get_state()
            gripper_pct = state.gripper_openness_pct
        return self.move_to_cartesian(float(ee[0]) + dx, float(ee[1]) + dy, float(ee[2]) + dz,
                                      gripper_pct, lock_wrist, gain)

    def servo_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                       lock_wrist: bool = True, gain: float = 0.25,
                       tol: float = 0.003, max_steps: int = 600) -> MoveResult:
        """Precise small relative Cartesian move for visual servoing.

        Unlike move_to_cartesian (12 mm convergence tolerance, tuned for the grasp/
        place primitives) this targets an EXACT 3D point ee+(dx,dy,dz) and converges
        to `tol` (default 3 mm), so sub-cm corrections actually take effect and a pure
        dz descent keeps x,y fixed. Gripper openness is held at its current value, so
        this never triggers the finger-settle ramp. Deltas are in metres."""
        ee = self.ik.get_ee_position()
        target = np.array([float(ee[0]) + dx, float(ee[1]) + dy, float(ee[2]) + dz])
        gripper_pct = self.backend.get_state().gripper_openness_pct
        ga = self._ga(gripper_pct)
        locked = _LOCKED_WRIST if lock_wrist else None
        with self._lock:
            for step in range(max_steps):
                ctrl = self.ik.step_toward_target(target, gripper_action=ga,
                                                  gain=gain, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()
                err = float(np.linalg.norm(target - self.ik.get_ee_position()))
                if err < tol:
                    return MoveResult(True, f"servo reached in {step+1} steps "
                                      f"(err={err*1000:.1f}mm)", self.backend.get_state())
            final = float(np.linalg.norm(target - self.ik.get_ee_position()))
            return MoveResult(final < 0.008, f"servo max steps (err={final*1000:.1f}mm)",
                              self.backend.get_state())

    def control_gripper(self, openness_pct: float, ik_hold: bool = True) -> MoveResult:
        """Move gripper to target openness. With ik_hold=True (default), runs IK during
        actuation to prevent arm from drifting away from its current position."""
        with self._lock:
            if ik_hold:
                target_ee = self.ik.get_ee_position()
                ga_start  = self._ga(self.backend.get_state().gripper_openness_pct)
                ga_end    = self._ga(openness_pct)
                r = self.ik.model.actuator_ctrlrange[5]
                g_start = float(self.ik.data.ctrl[5])
                g_end   = float((ga_end + 1) / 2 * (r[1] - r[0]) + r[0])
                steps = 300
                for s in range(steps):
                    t = (s + 1) / steps
                    ga_now = ga_start + (ga_end - ga_start) * t
                    ctrl = self.ik.step_toward_target(target_ee, gripper_action=ga_now,
                                                       gain=0.5, locked_joints=_LOCKED_WRIST)
                    ctrl[5] = g_start + (g_end - g_start) * t
                    self.backend.apply_control(ctrl)
                    self.backend.step()
            else:
                self.backend.move_gripper(openness_pct)
        return MoveResult(True, f"Gripper set to {openness_pct:.0f}%", self.backend.get_state())

    def set_joint_angles(self, angles_deg: dict, gripper_pct: Optional[float] = None) -> MoveResult:
        """Snap to an absolute multi-joint config (pre-position / recovery).

        gripper_pct is an internal-only parameter (the MCP `set_joint_angles` tool
        no longer exposes it — gripper is driven only via `set_gripper`). It is
        retained because the validated `grasp` port re-homes / pre-positions with a
        known jaw openness in a single locked move; None holds the current openness.
        """
        with self._lock:
            start_q = self.ik.data.qpos[:5].copy()
            target_q = start_q.copy()
            for name, deg in angles_deg.items():
                if name in JOINT_NAMES:
                    idx = JOINT_NAMES.index(name)
                    lo, hi = float(self.ik.model.jnt_range[idx, 0]), float(self.ik.model.jnt_range[idx, 1])
                    target_q[idx] = np.clip(np.radians(float(deg)), lo, hi) if lo != hi else np.radians(float(deg))

            ga = self._ga(gripper_pct) if gripper_pct is not None else None
            r = self.ik.model.actuator_ctrlrange[5]
            start_g = float(self.ik.data.ctrl[5])
            target_g = float((ga + 1) / 2 * (r[1] - r[0]) + r[0]) if ga is not None else start_g

            for s in range(150):
                t = (s + 1) / 150
                ctrl = np.zeros(self.ik.model.nu)
                ctrl[:5] = start_q + (target_q - start_q) * t
                ctrl[5] = start_g + (target_g - start_g) * t
                self.backend.apply_control(ctrl)
                self.backend.step()

        return MoveResult(True, "Joints set", self.backend.get_state())

    def move_joint_delta(self, joint: str, delta_deg: float, steps: int = 80) -> MoveResult:
        """Apply a relative angle delta to a single arm joint, holding the others.

        Used by the per-joint MCP verbs (move_shoulder, move_elbow). The MCP layer
        is responsible for the intuitive-effect sign convention (e.g. inverting the
        trebuchet so move_shoulder(+) raises the EE); this helper applies the raw
        signed delta to the named joint in radians, ramped over `steps`.
        """
        if joint not in JOINT_NAMES:
            return MoveResult(False, f"Unknown joint '{joint}'", self.backend.get_state())
        idx = JOINT_NAMES.index(joint)
        with self._lock:
            n = self.ik.n_arm
            lo, hi = float(self.ik.model.jnt_range[idx, 0]), float(self.ik.model.jnt_range[idx, 1])
            start = float(self.ik.data.qpos[idx])
            end = np.clip(start + np.radians(delta_deg), lo, hi) if lo != hi else start + np.radians(delta_deg)
            g_ctrl = float(self.ik.data.ctrl[5])
            for s in range(steps):
                t = (s + 1) / steps
                ctrl = self.ik.data.qpos[:n].copy()
                ctrl[idx] = start + (end - start) * t
                self.backend.apply_control(np.append(ctrl, g_ctrl))
                self.backend.step()
        return MoveResult(True, f"{joint} moved {delta_deg:+.1f}°", self.backend.get_state())

    def pick_object(self, x: float, y: float, z: float,
                    y_offset: float = _FINGER_Y_OFFSET) -> MoveResult:
        """Pick up an object at (x, y, z) and lift it to carrying height.
        y_offset adjusts the lateral finger alignment (-0.015 default for -y cubes,
        -0.010 works better for +y cubes)."""
        obj = np.array([x, y, z])
        locked = _LOCKED_WRIST

        with self._lock:
            # Open gripper
            for _ in range(20):
                self._step(self.ik.get_ee_position(), _GRIPPER_OPEN)

            # Approach above
            above = obj.copy()
            above[2] += _GRASP_Z_OFFSET + _APPROACH_Z
            above[1] += y_offset
            self._move(above, _GRIPPER_OPEN)

            # Lower to grasp
            grasp_pos = obj.copy()
            grasp_pos[2] += _GRASP_Z_OFFSET
            grasp_pos[1] += y_offset
            self._move(grasp_pos, _GRIPPER_OPEN, tol=0.008)

            # Close with contact detection and tighten — wrist snapped to π/2
            contact_step = contact_gripper = None
            grasp_gripper = _GRIPPER_CLOSED
            for step in range(300):
                if contact_step is None:
                    gripper = _GRIPPER_OPEN - 2.0 * min(step / 250, 1.0)
                else:
                    tgt = max(contact_gripper - _TIGHTEN, -1.0)
                    gripper = contact_gripper + (tgt - contact_gripper) * min((step - contact_step) / 100, 1.0)
                ctrl = self.ik.step_toward_target(grasp_pos, gripper_action=gripper,
                                                   gain=0.5, locked_joints=locked)
                ctrl[3] = np.pi / 2
                ctrl[4] = np.pi / 2
                self.backend.apply_control(ctrl)
                self.backend.step()
                if self.backend.is_grasping() and contact_step is None:
                    contact_step, contact_gripper = step, gripper
                if contact_step is not None:
                    if gripper <= max(contact_gripper - _TIGHTEN, -1.0) + 0.01:
                        grasp_gripper = gripper
                        break
            else:
                if contact_step is not None:
                    grasp_gripper = gripper

            # Lift to carrying height
            lift_pos = obj.copy()
            lift_pos[2] = _LIFT_Z
            for s in range(300):
                t = min(s / 250, 1.0)
                tgt = grasp_pos + (lift_pos - grasp_pos) * t
                ctrl = self.ik.step_toward_target(tgt, gripper_action=grasp_gripper,
                                                   gain=0.3, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()
            for _ in range(100):
                ctrl = self.ik.step_toward_target(lift_pos, gripper_action=grasp_gripper,
                                                   gain=0.3, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

        state = self.backend.get_state()
        if state.is_grasping:
            self._grasp_gripper = grasp_gripper
        return MoveResult(state.is_grasping,
                          "Object picked and lifted" if state.is_grasping else "Grasp failed — try repositioning",
                          state)

    def place_object(self, x: float, y: float, z: float) -> MoveResult:
        """Lower the held object to (x, y, z), release, and withdraw."""
        target = np.array([x, y, z])
        locked = _LOCKED_WRIST
        # Derive the current gripper action from the live actuator state, so the
        # carry grip is preserved regardless of how the pick happened (pick_object
        # or the move_to_cartesian +y recipe). Inverse of the forward mapping
        # g = (ga + 1) / 2 * (r1 - r0) + r0 used by IKController.step_toward_target.
        r = self.ik.model.actuator_ctrlrange[5]
        g_now = float(self.ik.data.ctrl[5])
        grasp_gripper = float(np.clip((g_now - r[0]) / (r[1] - r[0]) * 2.0 - 1.0, -1.0, 1.0))

        with self._lock:
            # GENTLE STAGED traverse to above the target (Task #21). A single fast
            # _move shears a marginal +y/shallow grip and flings the cube outward; a
            # fine interpolated carry at low gain keeps the cube tracking the EE the
            # whole diagonal. Interpolate from the CURRENT EE so the path is the real
            # carry, sub-stepped — not a teleport target the IK races toward.
            above = target.copy()
            above[2] = max(target[2] + 0.14, _ABOVE_CTR_Z)
            start_ee = self.ik.get_ee_position().copy()
            for i in range(1, _PLACE_CARRY_SUBSTEPS + 1):
                t = i / _PLACE_CARRY_SUBSTEPS
                rung = start_ee + (above - start_ee) * t
                self._move(rung, grasp_gripper, max_steps=200, tol=0.012,
                           gain=_PLACE_CARRY_GAIN)
            # settle at the top so the IK fully converges over the container before the
            # descent (a partly-converged top rung would start the descent off-center)
            for _ in range(_PLACE_CARRY_SETTLE):
                ctrl = self.ik.step_toward_target(above, gripper_action=grasp_gripper,
                                                   gain=_PLACE_CARRY_GAIN, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # DROP (not a gentle lower-inside): the open gripper barely fits the bin,
            # and the object only needs to END UP in the container — so we descend just
            # to the rim, open there, and let it fall the last few cm. No fingers-inside.
            drop_pt = target.copy()
            drop_pt[2] = max(target[2] + _PLACE_DROP_HEIGHT, _PLACE_DROP_MIN_Z)
            for s in range(160):
                t = min(s / 130, 1.0)
                tgt = above + (drop_pt - above) * t
                ctrl = self.ik.step_toward_target(tgt, gripper_action=grasp_gripper,
                                                   gain=0.4, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # Open the jaws over the bin and hold position so the object drops straight in
            for s in range(60):
                g = grasp_gripper + (_GRIPPER_OPEN - grasp_gripper) * min(s / 25, 1.0)
                ctrl = self.ik.step_toward_target(drop_pt, gripper_action=g,
                                                   gain=0.5, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # Let the object finish falling / settling in the bin before withdrawing
            for _ in range(60):
                ctrl = self.ik.step_toward_target(drop_pt, gripper_action=_GRIPPER_OPEN,
                                                   gain=0.3, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # Withdraw back up, jaws open
            for s in range(120):
                t = s / 120
                tgt = drop_pt + (above - drop_pt) * t
                ctrl = self.ik.step_toward_target(tgt, gripper_action=_GRIPPER_OPEN,
                                                   gain=0.3, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

        state = self.backend.get_state()
        return MoveResult(state.cube_in_container,
                          "Object placed" if state.cube_in_container
                          else "Placed but cube not in container",
                          state)

    # ── grasp() — server-side port of feedback_pick.pick_feedback ─────────────
    # Validated 97% pick / 94% place gentle, never-bat grasp. Ported faithfully
    # from feedback_pick.py: re-home -> basin pre-position with wrist-face
    # alignment -> gentle drift-aborted descent ladder -> IK-hold close -> test-lift
    # verify -> FINGER_LAT feedback sweep; never re-closes in place.
    #
    # LOCK DESIGN (critical): grasp() and its helpers DO NOT hold self._lock. They
    # orchestrate by calling the public locked methods (set_joint_angles,
    # move_to_cartesian, control_gripper, execute_intuitive_move), each of which
    # acquires/releases self._lock for its own duration. threading.Lock is
    # non-reentrant, so holding it across those calls would deadlock. State is read
    # between calls via self.backend.get_state() (which does not take self._lock)
    # or from the MoveResult each locked call returns.

    @staticmethod
    def _grasp_pan_deg(x: float, y: float) -> float:
        return math.degrees(math.atan2(-y, x))

    @staticmethod
    def _grasp_wrist_roll(pan: float, is_pos: bool, yaw_offset_deg: float = 0.0) -> float:
        """Wrist_roll (deg) keeping the jaws PARALLEL to a cube face (OWNER INSIGHT
        #1). Home wrist_roll=90 leaves jaws axis-aligned only at pan=0; as the base
        pans, the jaw line rotates with it and catches a corner. Counter-rotating
        wrist_roll by ~pan restores a face-parallel jaw line. Set at rest in the
        pre-position (wrist_roll spins about its own axis -> no EE translation, so
        it is drift-free); the descent then holds it via lock_wrist.

        -y slope 1.24 (was 1.04): the residual off-parallel jaw GROWS with |pan|, so
        at the inner high-pan -y cells (pan +15..+25°, e.g. (0.19,-0.07)) the 1.04
        formula left the jaw ~8-9° off-parallel -> only the MOVING pad caught the cube
        (a one-pad bracket), the seat-lift then DRAGGED that off-centre hold 12-19mm
        (past the 9mm never-bat budget) and the grasp declined having already batted
        it. The steeper 1.24 slope keeps the jaw face-parallel out to high pan so BOTH
        pads bracket -> firm first-try, clean lift, no drag. The correction scales with
        pan, so low-pan -y cells (the validated near column) are essentially unchanged
        (Δwr < 1.7° at pan +8). Measured: -y benchmark 46/51 -> 51/51, zero declines
        that bat.

        yaw_offset_deg (2026-07-25, opt-in, default 0.0 = old behaviour byte-identical):
        both formulas above implicitly assume the object's own top face is WORLD-AXIS-
        ALIGNED (yaw 0) -- true only because the validated benchmark scenes never
        randomize object yaw. `wrist_triangulate.estimate_face_yaw_deg` measures the
        object's ACTUAL in-plane rotation from the wrist silhouette instead of
        assuming it; a caller that has that measurement passes it here (wrapped to
        [-45, 45) already, so it always points at the NEAREST face) so a rotated
        object still gets a face-parallel approach. See grasp(object_yaw_deg=...)."""
        base = (90.0 + pan) if is_pos else (1.24 * pan + 92.0)
        return base + yaw_offset_deg

    def _grasp_read_cube(self) -> tuple[float, float, float]:
        cu = self.backend.get_state().cube_position_m
        return cu["x"], cu["y"], cu["z"]

    def _grasp_rehome(self) -> None:
        self.set_joint_angles({
            "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
            "wrist_flex": 90.0, "wrist_roll": 90.0}, gripper_pct=_G_PREOPEN_PCT)

    def _grasp_prepos(self, cx: float, cy: float, wf: float, is_pos: bool,
                      preopen_pct: float, yaw_offset_deg: float = 0.0) -> None:
        """Pre-position facing the cube AND align the jaws parallel to a cube face.
        Setting wrist_roll here (at rest, before any descent) is OWNER INSIGHT #1.
        yaw_offset_deg: measured object rotation (see _grasp_wrist_roll); 0.0 (the
        default at every call site unless the caller supplied object_yaw_deg to
        grasp()) reproduces the exact validated angle."""
        pan = self._grasp_pan_deg(cx, cy)
        self.set_joint_angles({
            "shoulder_pan": pan, "shoulder_lift": _G_PRE_SL,
            "elbow_flex": _G_PRE_EF, "wrist_flex": wf,
            "wrist_roll": self._grasp_wrist_roll(pan, is_pos, yaw_offset_deg)},
            gripper_pct=preopen_pct)

    def _grasp_descend(self, cx: float, ty: float, gz: float, wf: float,
                       cu0x: float, cu0y: float, log) -> tuple[bool, float]:
        """Slow ladder descent to grasp height along the fixed vertical line (cx,ty).
        Monitors cube drift each rung; returns (aborted, drift_mm). The fixed target
        self-centers laterally as the IK converges -> no lateral sweep into the cube.

        gz is the EE grasp-z ANCHOR: the ladder descends to gz + each rung dz. For the
        cube this is the live object center (validated behavior); grasp_height_m lets a
        taller object be gripped on its upper-mid body instead."""
        drift = 0.0
        for dz, gain in [(0.030, 0.5), (0.020, 0.5), (0.010, 0.4), (_G_GRASP_DZ, 0.3)]:
            self.set_joint_angles({"wrist_flex": wf})
            r = self.move_to_cartesian(cx, ty, gz + dz, lock_wrist=True, gain=gain)
            cu = r.state.cube_position_m
            drift = math.hypot(cu["x"] - cu0x, cu["y"] - cu0y)
            if drift > _G_DRIFT_ABORT:
                log(f"      descend ABORT: cube drift {drift*1000:.1f}mm at dz={dz:+.3f}")
                # A wide/round object that the descending pad grazes off-center is now
                # SLIDING. Break contact FAST by lifting the EE straight up before it
                # coasts further (a slow disengage let a 50mm box coast 7->25mm). This is
                # a pure vertical retreat (no lateral motion), so it never shoves the
                # object — it only un-touches it sooner.
                self.move_to_cartesian(cx, ty, gz + _G_SAFE_DZ, lock_wrist=True, gain=0.5)
                return True, drift * 1000
        return False, drift * 1000

    def _grasp_seat_and_close(self, cx: float, ty: float, gz: float, wf: float,
                              cu0x: float, cu0y: float, log) -> tuple[bool, bool, float]:
        """From grasp height, take the final seating rung then close with IK-hold
        (control_gripper(0), no wrist snap). Monitors drift. Returns
        (grasped, aborted, drift_mm). gz is the EE grasp-z anchor (see _grasp_descend)."""
        self.set_joint_angles({"wrist_flex": wf})
        r = self.move_to_cartesian(cx, ty, gz + _G_SEAT_DZ, lock_wrist=True, gain=_G_SEAT_GAIN)
        cu = r.state.cube_position_m
        drift = math.hypot(cu["x"] - cu0x, cu["y"] - cu0y)
        if drift > _G_DRIFT_ABORT:
            log(f"      seat ABORT: cube drift {drift*1000:.1f}mm")
            # fast vertical disengage (see _grasp_descend) — un-touch a rolling object
            # before it coasts further; pure up-move, never shoves it.
            self.move_to_cartesian(cx, ty, gz + _G_SAFE_DZ, lock_wrist=True, gain=0.5)
            return False, True, drift * 1000
        # Drift-guarded close: a clean grasp brackets the object and the pads close on
        # it with ZERO push (the cube). A MISALIGNED close (one pad hits a face off-
        # center) SHOVES the object — for a small/narrow box a single full close can
        # push it 25mm, a bat. So we close while watching drift and STOP the instant it
        # exceeds the abort threshold, capping the shove to a few mm (gentle decline).
        aborted, drift = self._close_with_drift_guard(cx, ty, cu0x, cu0y)
        if aborted:
            log(f"      close ABORT: object pushed {drift:.1f}mm during close")
            return False, True, drift
        return self.backend.is_grasping(), False, drift

    def _close_with_drift_guard(self, cx: float, ty: float,
                                cu0x: float, cu0y: float) -> tuple[bool, float]:
        """IK-hold close (mirrors control_gripper(0)) that ABORTS the moment the object
        is pushed past the drift threshold. Returns (aborted, drift_mm). A clean grasp
        closes with ~0 push and runs to completion; a misaligned close that shoves the
        object stops early so it is never batted."""
        with self._lock:
            target_ee = self.ik.get_ee_position()
            ga_start = self._ga(self.backend.get_state().gripper_openness_pct)
            ga_end = self._ga(0.0)
            rr = self.ik.model.actuator_ctrlrange[5]
            g_start = float(self.ik.data.ctrl[5])
            g_end = float((ga_end + 1) / 2 * (rr[1] - rr[0]) + rr[0])
            steps = 300
            drift = 0.0
            for s in range(steps):
                t = (s + 1) / steps
                ga_now = ga_start + (ga_end - ga_start) * t
                ctrl = self.ik.step_toward_target(target_ee, gripper_action=ga_now,
                                                  gain=0.5, locked_joints=_LOCKED_WRIST)
                ctrl[5] = g_start + (g_end - g_start) * t
                self.backend.apply_control(ctrl)
                self.backend.step()
                cu = self.backend.get_state().cube_position_m
                # only a FREE (un-gripped) object being shoved is a bat; once the pads
                # have it (is_grasping) the small co-motion as it seats is not a push.
                if not self.backend.is_grasping():
                    drift = math.hypot(cu["x"] - cu0x, cu["y"] - cu0y)
                    if drift > _G_CLOSE_DRIFT_ABORT:
                        return True, drift * 1000.0
        return False, drift * 1000.0

    def _grasp_lift_clear(self, wf: float) -> None:
        """Open and lift the EE clear of the cube without batting."""
        self.control_gripper(_G_PREOPEN_PCT)
        self.set_joint_angles({"wrist_flex": wf})
        self.execute_intuitive_move(move_gripper_up_mm=_G_LIFT_CLEAR_MM, gain=0.4, lock_pan=True)

    def _grasp_carry_lift(self, cx: float, ty: float, obj_h_mm: float = 30.0,
                          is_pos: bool = False) -> bool:
        """Lift the just-grasped object to carry height, re-checking the grip on the way.
        NEVER-BAT: a marginal grip can pass the seat-lift then slip during the carry,
        dropping the object from height and scattering it. So the carry climbs in FINE
        rungs (every ~2cm) and re-checks the grip at each one; the instant the grip is
        genuinely lost (confirmed by the non-destructive settle, NOT a single flicker)
        it STOPS and sets the object down from the LOW rung it just reached instead of
        carrying it higher to drop it. With fine rungs the object only ever falls from
        one rung, capping any scatter to a few mm (well under the never-bat budget).

        Carry style by workspace SIDE (the validated -y behavior, plus the +y fix):
          - -y TALL grips (the 30mm cube) hold best under ONE continuous carry — staging
            them shears the grip between rungs.
          - +y grips are marginal/steep -> the gentle fine STAGED lift HOLDS them; a
            single fast carry shears them and flung the cube 128mm at (0.20,+0.08).
        On loss, set the object down from the rung it reached (fine rungs cap the fall).
        Returns True iff the object is still grasped at carry height."""
        # LOW-HEIGHT STABILITY GATE (never-bat): dwell at a low height and re-confirm the
        # grip before lifting high, so a marginal grip fails LOW (small scatter) not at
        # carry height. The drop check is STABILITY-GATED (Task #21): a single
        # is_grasping=False frame is often a contact flicker, not a drop — reacting to it
        # dropped good grips — so we set down only if the grip also FAILS the
        # non-destructive settle.
        for _ in range(3):
            r = self.move_to_cartesian(cx, ty, _G_GATE_Z, lock_wrist=True, gain=_G_CARRY_GAIN)
            if not _G_COMMIT_ON_GRIP and self._cube_dropped(_G_GATE_Z) and not self._grip_holds():
                # commit mode SKIPS this: never open a held grip mid-carry — carry to the bin
                self.move_to_cartesian(cx, ty, _G_TEST_LIFT_Z, lock_wrist=True, gain=_G_CARRY_GAIN)
                self.control_gripper(_G_PREOPEN_PCT)
                return False

        # Carry in FINE rungs always (Task #21): the staged climb plus the height-
        # tracking drop guard holds the shallow -y cells (0.20/0.24,-0.08) that a single
        # 0.17 carry lost to contact flicker. The -y tall single-carry was shearing a
        # firm-but-flickering grip; fine rungs + cube-height guard keep it.
        rungs = _G_CARRY_RUNGS
        for zt in rungs:
            r = self.move_to_cartesian(cx, ty, zt, lock_wrist=True, gain=_G_CARRY_GAIN)
            dropped = self._cube_dropped(zt)
            self._carry_log(f"      carry rung z={zt:.3f} is_grasping={r.state.is_grasping} "
                            f"cube_dropped={dropped}")
            # DROP TEST: the cube is lost ONLY if it has FALLEN out (cube_z far below the
            # EE) — an is_grasping flicker while the cube still tracks the EE is NOT a drop.
            if not _G_COMMIT_ON_GRIP and dropped and not self._grip_holds():
                # commit mode SKIPS this: never open a held grip mid-carry
                setdown = min(_G_SETDOWN_Z, zt)
                self.move_to_cartesian(cx, ty, setdown, lock_wrist=True, gain=_G_CARRY_GAIN)
                self.control_gripper(_G_PREOPEN_PCT)
                return False
        # Success iff the cube is still aloft with the gripper (tracks the EE), not the
        # flickery is_grasping flag. SETTLE briefly at the top so the pad contact re-seats
        # and is_grasping reads True for the caller/harness (the dynamic lift leaves it
        # flickering; a short dwell holding ctrl lets it re-latch — measured to recover).
        if self._cube_dropped(_G_CARRY_Z):
            return False
        # The cube is aloft. The dynamic lift can leave the pad contact flickering so
        # is_grasping reads False even though the cube is firmly between the jaws (it
        # tracks the EE). A held cube at carry height is SAFE to firm: an IK-hold full
        # close (no arm move, cube already up) re-establishes solid pad contact so
        # is_grasping reads True for the caller/place — gentle, never-bat (the cube can't
        # be batted while it is aloft in the jaws). Only do this if it is genuinely held.
        # (Removed the control_gripper(0) full re-close that used to run here to make
        # is_grasping read True for the caller: at steep wrist angles that full close
        # EJECTED the cube it was already holding — the cube tracked the EE all the way
        # up, then the re-close dropped it to the table, a FALSE decline of a good pick.
        # The cube is held (cube_dropped is False = it tracks the EE); report success on
        # that robust height signal and leave the seated grip untouched.)
        return not self._cube_dropped(_G_CARRY_Z)

    def _cube_dropped(self, ee_z_target: float) -> bool:
        """True iff the cube has FALLEN out of the jaws: its z sits more than
        _G_CARRY_DROP_GAP below the current EE height. A cube genuinely held tracks the
        EE (~7mm below it) the whole carry even while is_grasping flickers, so this is
        the robust drop signal (height tracking), not the toggling contact flag."""
        ee = self.ik.get_ee_position()
        cu = self.backend.get_state().cube_position_m
        return (float(ee[2]) - float(cu["z"])) > _G_CARRY_DROP_GAP

    def _grip_holds(self) -> bool:
        """True if the grip survives the settle check (not a single-frame flicker).
        Holds ctrl fixed and steps a few frames; the grip must stay grasping. Used to
        gate the carry's drop decision so a momentary is_grasping flicker — the thing
        that dropped good grips — does not trigger a set-down."""
        return bool(self.backend.grip_metrics(settle_steps=_G_GRIP_SETTLE_STEPS)["stable"])

    def _grasp_reseat(self, cx: float, ty: float, gz: float, wf: float,
                      lat_delta: float, cu0x: float, cu0y: float,
                      log) -> tuple[bool, bool, float, dict]:
        """TINY LOCAL re-seat (Task #21) — the GENTLE alternative to the far
        re-home/lift-clear/re-approach retry. From the current grip position it:
          1. opens the jaws only PARTWAY (_G_NUDGE_OPEN_PCT, not the full approach
             open) so the object is released without a fling;
          2. lifts the EE a few mm (_G_NUDGE_LIFT_MM) so the open jaw clears the
             object top instead of dragging across it;
          3. shifts laterally by a few mm (lat_delta, |.|<=_G_NUDGE_LAT_MM) to the
             new finger line at the CURRENT height — no re-home, no big travel;
          4. re-descends to the seat rung and re-closes with the drift-guarded
             IK-hold close.
        Returns (grasped_firm, aborted, drift_mm, info). The whole nudge stays well
        under the never-bat budget (a few mm of lateral motion, drift-guarded). The
        object is read between steps so a shove still aborts cleanly."""
        new_ty = ty + lat_delta
        # 0. lower the EE back toward the table FIRST (the prior seat-lift left it at
        #    ~0.06 holding the object marginally). Releasing at 0.06 would drop the
        #    object 6cm; lowering to seat height first means the partial-open releases
        #    it at table level — no drop, no scatter.
        self.set_joint_angles({"wrist_flex": wf})
        self.move_to_cartesian(cx, ty, gz + _G_SEAT_DZ, lock_wrist=True, gain=_G_NUDGE_GAIN)
        # 1. partial open (release at table level, don't fling)
        self.control_gripper(_G_NUDGE_OPEN_PCT)
        # 2. lift the EE CLEAR of the object top BEFORE shifting laterally. Shifting at
        #    object level rams the cube (a bat); lifting to a safe clearance first means
        #    the lateral shift happens with the open jaw ABOVE the object, so it never
        #    touches it. This is the gentle "mini re-approach" — local (no full re-home
        #    travel) but still over-the-top, not a table-level sweep.
        self.set_joint_angles({"wrist_flex": wf})
        self.move_to_cartesian(cx, ty, gz + _G_SAFE_DZ, lock_wrist=True, gain=0.5)
        # 3. shift laterally to the new finger line, still ABOVE the object (no touch)
        r = self.move_to_cartesian(cx, new_ty, gz + _G_SAFE_DZ,
                                   lock_wrist=True, gain=_G_NUDGE_GAIN)
        cu = r.state.cube_position_m
        drift = math.hypot(cu["x"] - cu0x, cu["y"] - cu0y) * 1000.0
        if drift / 1000.0 > _G_DRIFT_ABORT:
            log(f"      reseat ABORT: object moved {drift:.1f}mm during lateral shift")
            self.move_to_cartesian(cx, new_ty, gz + _G_SAFE_DZ, lock_wrist=True, gain=0.5)
            return False, True, drift, {"stage": "reseat_shift"}
        # 4. gentle re-descent ladder to the new finger line, then seat + drift-guarded
        #    close (same gentle rungs as the first attempt -> no shove on the way down).
        ab, drift = self._grasp_descend(cx, new_ty, gz, wf, cu0x, cu0y, log)
        if ab:
            return False, True, drift, {"stage": "reseat_descend"}
        self.set_joint_angles({"wrist_flex": wf})
        r = self.move_to_cartesian(cx, new_ty, gz + _G_SEAT_DZ,
                                   lock_wrist=True, gain=_G_SEAT_GAIN)
        cu = r.state.cube_position_m
        drift = math.hypot(cu["x"] - cu0x, cu["y"] - cu0y) * 1000.0
        if drift / 1000.0 > _G_DRIFT_ABORT:
            log(f"      reseat ABORT: object moved {drift:.1f}mm during re-seat rung")
            self.move_to_cartesian(cx, new_ty, gz + _G_SAFE_DZ, lock_wrist=True, gain=0.5)
            return False, True, drift, {"stage": "reseat_seat"}
        aborted, drift = self._close_with_drift_guard(cx, new_ty, cu0x, cu0y)
        if aborted:
            log(f"      reseat close ABORT: object pushed {drift:.1f}mm during close")
            return False, True, drift, {"stage": "reseat_close"}
        # gentle seat-lift so the object settles between both pads, then read grip
        self.move_to_cartesian(cx, new_ty, _G_TEST_LIFT_Z, lock_wrist=True, gain=0.15)
        cg = self.check_grip()
        log(f"      reseat lat_delta={lat_delta:+.4f} -> grip {cg['quality']} "
            f"(gap={cg['jaw_gap_mm']}mm err={cg['gap_err_mm']}mm both={cg['both_pads']} "
            f"stable={cg['stable']})")
        return (cg["quality"] == "firm"), False, drift, {
            "ty": new_ty, "grip": cg, "quality": cg["quality"]}

    def _grasp_attempt_gentle(self, cx: float, cy: float, cz: float, wf: float,
                              finger_lat: float, preopen_pct: float, log,
                              grasp_z: Optional[float] = None) -> tuple[bool, bool, dict]:
        """TIER 1 grasp: gentle near-vertical descent to a fixed (cx, cy+finger_lat)
        line, then an IK-hold close (control_gripper(0), no wrist snap). Near-zero
        cube drift. Returns (grasped, aborted, info). Never re-closes in place.

        grasp_z: EE grasp-z anchor for the descent ladder. None -> the live object
        center cz (validated cube behavior). Drift is always measured against the live
        object center (cx, cy), independent of where we choose to grip vertically."""
        gz = cz if grasp_z is None else grasp_z
        # 1. above the object, OPEN, finger-compensated lateral target. The safe
        #    approach clears whichever is higher: the grasp anchor or the cube center.
        self.set_joint_angles({"wrist_flex": wf})
        self.move_to_cartesian(cx, cy + finger_lat, max(gz, cz) + _G_SAFE_DZ,
                               gripper_pct=preopen_pct, lock_wrist=True, gain=0.5)
        # 2. slow descent to grasp height (fixed vertical target self-centers; no
        #    lateral nudge at depth -> no bump).
        ty = cy + finger_lat
        aborted, drift = self._grasp_descend(cx, ty, gz, wf, cx, cy, log)
        if aborted:
            return False, True, {"stage": "descend", "drift": drift}
        # 3. seat + close (drift-guarded IK-hold close)
        grasped, aborted, drift = self._grasp_seat_and_close(cx, ty, gz, wf, cx, cy, log)
        if aborted:
            return False, True, {"stage": "seat", "drift": drift}
        # 4. GENTLE SEAT-LIFT, then the CONFIDENCE READ. The seat-lift to ~0.06 is
        #    FUNCTIONAL (the validated behavior): on the table the jaws often bracket
        #    the object with only ONE pad touching, and lifting it a few cm settles it
        #    fully between BOTH pads — so we lift even when the raw close shows one pad.
        #    This is the small lift the object needs anyway to leave the table, NOT the
        #    big 35mm-clear + re-home motion that shook good grips loose. check_grip
        #    then settles a few sim steps holding ctrl (so a single is_grasping flicker
        #    is NOT a drop) and classifies the grip post-lift.
        #    BUT (Task #21): if the raw close caught NEITHER pad (a clean miss / empty),
        #    a seat-lift just DRAGS the off-center cube laterally (measured +5mm at the far
        #    short-radius -y cells), eating the never-bat budget before the sweep can reach
        #    the seed that seats both pads. So skip the lift on a no-pad close and read the
        #    grip at table level — the sweep then moves to the next seed having barely
        #    moved the cube.
        sp0, mp0 = self.backend.pad_contacts()
        if not (sp0 or mp0):
            cg = self.check_grip()
            info = {"ty": ty, "grip": cg, "quality": cg["quality"], "lifted_grasping": False}
            log(f"      no-pad close (skip seat-lift) -> grip {cg['quality']} "
                f"(gap={cg['jaw_gap_mm']}mm err={cg['gap_err_mm']}mm)")
            return False, False, info
        rlift = self.move_to_cartesian(cx, ty, _G_TEST_LIFT_Z, lock_wrist=True, gain=0.15)
        cg = self.check_grip()
        # The grip is usable if it is FIRM, or at least a stable two-pad hold that
        # survived leaving the table (is_grasping after the lift) — the original
        # success signal. A pure one-pad bracket that never became is_grasping is a
        # miss; the loop sweeps the next seed for it.
        usable = (cg["quality"] == "firm") or (cg["both_pads"] and rlift.state.is_grasping)
        info = {"ty": ty, "grip": cg, "quality": cg["quality"],
                "lifted_grasping": bool(rlift.state.is_grasping)}
        log(f"      seat-lift -> grip {cg['quality']} (gap={cg['jaw_gap_mm']}mm "
            f"err={cg['gap_err_mm']}mm both={cg['both_pads']} "
            f"sp={cg['static_pad']} mp={cg['moving_pad']} stable={cg['stable']} "
            f"lifted_grasping={rlift.state.is_grasping})")
        return usable, False, info

    @staticmethod
    def _jaw_pct_for_width(width_mm: float) -> float:
        """Map a desired FREE jaw gap (object width + margin, mm) to the approach
        openness percent that yields AT LEAST that gap, using the empirical pad-gap
        fit. The result is never below the validated cube default (65%) — the wide
        pre-open is what keeps the descent contact-free and gentle — and never above
        the clean parallel-grasp band (the moving jaw arcs up past ~75%). So narrow
        objects keep the proven 65% pre-open; only genuinely wide objects open wider."""
        center_gap = float(width_mm) + _JAW_PAD_THICK
        pct = (center_gap - _JAW_INTERCEPT) / _JAW_SLOPE
        # floor at the cube default so we never pre-open tighter than the validated grasp
        return float(np.clip(pct, _G_PREOPEN_PCT, _JAW_PREOPEN_MAX_PCT))

    def _derive_grasp_params(self, grip_width_mm: Optional[float],
                             grasp_height_m: Optional[float],
                             object_width_mm: Optional[float] = None,
                             object_width_uncertainty_mm: Optional[float] = None
                             ) -> tuple[float, Optional[float], float, dict]:
        """Resolve (preopen_pct, grasp_z, lat_offset, info) for a grasp.

        grip_width_mm None -> derive from inspect_object: object width + ~10mm margin,
        clamped to the gripper max. grasp_height_m None -> derive the EE grasp-z
        anchor: short objects grip low (cube behavior, anchor = center); tall objects
        grip the upper-mid body (top - 15mm) so the jaws bracket the body, not the
        table. lat_offset is the width-adaptive shift added to the finger_lat seed so
        narrow objects (<30mm) grasp first-try. Returns the resolved values plus an
        info dict for logging/telemetry.

        object_width_uncertainty_mm (2026-07-23): softens the never-bat width
        thresholds from a hard cliff into a band, so a NOISY measured width
        degrades gracefully. None (default, every existing call site) keeps the
        exact current cliff — this is opt-in and does not change the validated
        path. See the _G_WIDTH_UNCERT_MM comment below for what this does and
        does NOT fix (it is not a substitute for correcting a known bias)."""
        info = self.backend.inspect_object()
        # TRUTH LEAK, now closable: `width` below drives the finger_lat seed shift and the
        # never-bat decline thresholds, and it was ALWAYS read from inspect_object() — the
        # simulator's true geometry — even when the caller had supplied a perception-measured
        # grip_width_mm. So a caller on the "honest" route still got truth-driven finger
        # placement. `object_width_mm` lets the caller pass the width its CAMERA measured
        # (wrist_triangulate returns one, from sqrt(mask area) x depth / f). None keeps the
        # old behaviour so the validated benchmark path is unchanged.
        # NOTE what is still truth: info["shape"], used by the sphere/box never-bat declines
        # below. The pick remains honest about WHERE and now about HOW WIDE, but not yet
        # about WHAT — say so rather than claiming a fully perceptual grasp.
        width = float(object_width_mm) if object_width_mm is not None else info.get("width_mm", 30.0)
        height = info.get("height_mm", 30.0)
        center = info.get("center_m", [0, 0, 0.015])
        cz = float(center[2])

        # --- width -> jaw pre-open ---
        if grip_width_mm is None:
            grip_width_mm = min(width + 10.0, _JAW_MAX_WIDTH)
        else:
            grip_width_mm = min(float(grip_width_mm), _JAW_MAX_WIDTH)
        preopen_pct = self._jaw_pct_for_width(grip_width_mm)

        # --- height -> EE grasp-z anchor ---
        # A "short" object (height <= ~2x the cube, i.e. fingers reach the table side
        # when bracketing the center) is gripped at its center like the cube -> anchor
        # None preserves the validated descent exactly. A tall object is gripped on
        # the upper-mid body: top - 15mm, but never below the center (so we always
        # have body above and below the pads).
        grasp_z: Optional[float] = None
        if grasp_height_m is not None:
            grasp_z = float(grasp_height_m)
        elif height > 36.0:  # taller than ~the 3cm cube + a little -> grip up high
            top = cz + (height / 2000.0)
            grasp_z = max(top - 0.015, cz)  # upper-mid body, at least the center

        # --- width -> finger_lat seed shift ---
        # Narrower-than-reference objects need a more-negative finger_lat so the static
        # pad reaches their (closer) near face. Wider objects keep the validated seed.
        lat_offset = -_G_LAT_PER_MM * max(0.0, _G_LAT_WIDTH_REF_MM - float(width))

        # ── width-uncertainty margin (2026-07-23) ─────────────────────────────
        # Measured (eval_grasp_measured_width.py, real wrist-triangulation pipeline):
        # a 30mm cube reads 38.6mm nadir-only (+29%) and _G_WIDE_DECLINE_MM=32.0, so
        # that single reading crosses the decline line by 6.6mm and turns a 12/12
        # honest grasp into 0/12 — every attempt declined untouched, 0mm displacement,
        # correct behaviour given a WRONG number. An independent colour-threshold
        # measurement (eval_width_bias.py, 5 shapes x 5 positions, experiments/
        # results/width_bias_results.txt) confirms the direction and rough stability
        # of this bias FOR BOXES ONLY: nadir bias +4.8..+7.4% mean per box shape
        # (cube_3cm/box_25mm/box_40mm), std 0.7-3.1 percentage points across pose —
        # i.e. mostly a stable multiplicative BIAS, not noise. The SAME script found
        # the sign FLIPS for round objects (cylinder -7.1%, sphere -11.6%, matching
        # the sqrt(pi/4)=0.886 circle-vs-square packing factor almost exactly) — a
        # single correction factor would be wrong for round shapes. That does not
        # matter for these two flags: both are gated `shape == "box"` already, so a
        # box-only bias number is the correct one to reason about here.
        #
        # A per-pose ~1-3mm bias-corrected SCATTER (measurement uncertainty proper)
        # is a legitimate reason to soften a hard cliff — object_width_uncertainty_mm
        # lets a caller declare it, and only the CONFIDENT side of the estimate is
        # compared to the threshold (decline only if even the more-generous bound
        # still crosses the line), so noise near the boundary no longer flips the
        # decision outright. But it is NOT a fix for the 0/12 case above: that gap
        # (6.6mm) is a SYSTEMATIC bias, not scatter, and covering it would require an
        # uncertainty band wide enough to also un-decline a genuinely too-wide short
        # box — measured: a 40mm short box peaks 15.1-15.6mm drift (a bat) if
        # attempted, and 40mm is only 8mm over this same threshold. Widening the
        # margin to swallow a 29% bias would swallow that bat case too. The bias
        # itself has to be corrected upstream (by whatever derived object_width_mm —
        # this layer doesn't know the estimator, so it doesn't guess at one); this
        # margin only absorbs genuine leftover noise around an already-reasonable
        # estimate. Default None keeps every existing call site's exact behaviour.
        unc = 0.0 if object_width_uncertainty_mm is None else float(object_width_uncertainty_mm)
        width_lo = float(width) - unc   # the pessimistic-for-"is it too wide" bound
        width_hi = float(width) + unc   # the pessimistic-for-"is it too narrow" bound

        # NEVER-BAT wide-box flag: a SHORT box (grasp_z None -> gripped at its low
        # center) that is WIDER than the lever threshold cannot be center-gripped by the
        # flat pads — the deeper seat rung levers it sideways. Flagged here so grasp()
        # declines it up front WITHOUT touching it. A TALL wide box (grasp_z set, gripped
        # high) is NOT flagged (it grasps on its upper body). Round objects are excluded
        # (they have their own roll-decline path); this is the box lever case only.
        # Uses width_lo (not width) so an uncertain reading must clear the line even at
        # its most charitable-to-attempting reading before we decline it.
        low_grip_wide = (grasp_z is None
                         and info.get("shape") == "box"
                         and width_lo > _G_WIDE_DECLINE_MM)

        # NEVER-BAT narrow-box flag: a box NARROWER than the validated cube cannot be
        # closed on without the flat moving pad shoving it 12-19mm (a bat) — verified at
        # both the -y and +y cells. Applies regardless of grip height: a tall narrow box
        # (box_tall_2x5) topples 38mm if the close is attempted. So any sub-cube box is
        # declined up front. Keyed strictly below 30mm so the 30mm cube is unaffected.
        # Uses width_hi (mirrors low_grip_wide's width_lo) for the same reason.
        narrow_box = (info.get("shape") == "box"
                      and width_hi < _G_NARROW_DECLINE_MM)

        # NEVER-BAT short-cylinder flag: a SHORT cylinder (grasp_z None -> gripped at its
        # low center) is a rolling round BAND for the flat pads — the same class as the
        # sphere. Each one-pad close + seat-lift DRAGS it a few mm and it never seats both
        # pads; measured cyl_short walks to a 15.0mm transient peak (right at the ceiling)
        # before the budget guard can stop the sweep. A TALL cylinder (grasp_z set) is NOT
        # flagged — gripped high on its body it presents a graspable band and declines
        # gently on contact (well under budget) if it can't be held. So only the SHORT
        # roller is declined up front WITHOUT touching it (0mm) — the honest never-bat
        # outcome that keeps the transient peak at zero instead of 15mm.
        low_grip_round = (grasp_z is None and info.get("shape") == "cylinder")

        return preopen_pct, grasp_z, lat_offset, {
            "shape": info.get("shape"), "obj_w_mm": width, "obj_h_mm": height,
            "obj_w_uncertainty_mm": unc,
            "grip_w_mm": round(float(grip_width_mm), 1),
            "preopen_pct": round(preopen_pct, 1),
            "grasp_z": None if grasp_z is None else round(grasp_z, 4),
            "lat_seed": round(_G_FINGER_LAT_NEG + lat_offset, 4),
            "low_grip_wide": low_grip_wide,
            "low_grip_round": low_grip_round,
            "narrow_box": narrow_box,
        }

    def grasp(self, x: float, y: float, z: float, approach: str = "top",
              grip_width_mm: Optional[float] = None,
              grasp_height_m: Optional[float] = None,
              width_mm: Optional[float] = None,
              object_width_mm: Optional[float] = None,
              object_width_uncertainty_mm: Optional[float] = None,
              object_yaw_deg: Optional[float] = None,
              trust_coords: bool = False,
              log_sink=None) -> MoveResult:
        """Skill verb: gentle, never-bat top grasp of the object at (x, y, z).

        Server-side port of feedback_pick.pick_feedback (validated 97% pick / 94%
        place across the workspace grid). Reads the object live and drives the grasp
        by feedback: re-home -> basin pre-position with wrist-face alignment ->
        slow drift-aborted descent ladder -> IK-hold close -> test-lift verify ->
        FINGER_LAT feedback sweep across bounded retries; OPEN+LIFT+re-home+re-read
        between attempts; never re-closes blindly in place. GENTLE-ONLY: every
        motion is a gain<=0.5 rung that drift-aborts — no wrist-snap, no fling.
        There is deliberately NO pick_object fallback (its wrist-snap close flings
        the object 35-64mm when it misses, violating the never-bat constraint).

        approach: "top" (implemented + validated) or "side" (reserved, D4 — returns
        a clear not-implemented warning until built). Any other value errors.

        grip_width_mm: how wide to pre-open the jaws (object width + margin, mm),
        clamped to the gripper max (~103mm). None -> derived from inspect_object
        (object width + 10mm margin). Sets only the approach jaw gap; the validated
        descent/close geometry is unchanged.

        grasp_height_m: the EE/TCP world-z to descend to for the close (where on the
        body to grip). None -> derived: short objects grip low like the cube; tall
        objects grip the upper-mid body (top - 15mm). Threaded into the descent ladder
        in place of the hardcoded cube-center anchor.

        width_mm: deprecated alias for grip_width_mm (kept for the old tool signature).

        object_width_mm: the object's width as MEASURED BY PERCEPTION (mm). Feeds the
        finger_lat seed shift and the never-bat width thresholds. Without it those fall
        back to the simulator's true geometry, which silently un-does the honest route
        even when trust_coords=True and a measured grip_width_mm was passed. NOTE this
        is read as-is: a systematic estimator bias (measured for wrist-triangulation's
        sqrt(area) width: a stable +5-8% over-read for BOX shapes, opposite-signed for
        round ones — see eval_width_bias.py) is the caller's to correct BEFORE passing
        it in; this layer has no way to know which estimator produced the number.

        object_width_uncertainty_mm: softens the never-bat width thresholds from a
        hard cliff into a band — declines only if the object is too wide/narrow even
        at the more-charitable-to-attempting edge of [object_width_mm - unc,
        object_width_mm + unc]. For genuine per-pose measurement scatter (measured
        ~1-3mm for box shapes, eval_width_bias.py) this avoids a noisy reading
        flipping the decision right at the boundary. It is NOT a substitute for
        correcting a systematic bias: covering a 29%, 6.6mm-over-threshold bias this
        way would also un-decline a genuinely-too-wide short box that DOES bat (a
        measured 40mm short box is only 8mm over the same line). None (default, every
        existing call site) keeps the exact current cliff.

        object_yaw_deg (2026-07-25): the object's own in-plane rotation about world z
        (degrees, any wrap — wrapped internally to the nearest face via mod-90), as
        MEASURED by perception (wrist_triangulate.estimate_face_yaw_deg reads it from
        the wrist silhouette's edge direction). Added to the wrist_roll pre-position
        (see _grasp_wrist_roll) so the jaws meet a rotated object's face squarely
        instead of a corner. ADOPTED IDEA from QuickGrasp (arXiv 2504.19716, parallel-
        jaw antipodal grasp planning): choose the approach from the object's own
        geometry rather than assuming it is world-axis-aligned. None (default, every
        existing call site) adds a zero offset — byte-identical to the old `90+pan` /
        `1.24*pan+92` formulas, which is what the validated axis-aligned benchmark
        (shape_scene.py never randomizes yaw) continues to exercise.

        trust_coords: HONEST-PERCEPTION route. When True, the grasp is driven by the
        passed (x, y, z) — a coordinate PERCEIVED by find_object / the camera pipeline
        — instead of reading the object's true simulator qpos. Default False preserves
        the validated benchmark path (grasp reads the live object). Passing a nominal
        width/height via grip_width_mm/grasp_height_m stays allowed under either route;
        only reading the true CENTER to drive motion is what trust_coords replaces.

        Returns MoveResult; result.state.is_grasping is the success signal.
        """
        if approach == "side":
            return MoveResult(False,
                              "grasp(approach='side') is not implemented yet (reserved, D4). "
                              "Use approach='top'.",
                              self.backend.get_state())
        if approach != "top":
            return MoveResult(False,
                              f"grasp: unknown approach '{approach}' — must be 'top' or 'side'.",
                              self.backend.get_state())

        # Back-compat: width_mm is the old name for grip_width_mm.
        if grip_width_mm is None and width_mm is not None:
            grip_width_mm = width_mm

        # Multi-object: bind the active target to the object nearest the grasp point,
        # so inspect/is_grasping/check_grip all refer to what we're actually picking
        # up (single-object scenes have only the cube, so this is a no-op there).
        self.backend.set_active_object_by_point(x, y)

        # Derive jaw pre-open, EE grasp-z anchor, and width-adaptive lat seed shift.
        preopen_pct, grasp_z, lat_offset, dinfo = self._derive_grasp_params(
            grip_width_mm, grasp_height_m, object_width_mm, object_width_uncertainty_mm)
        # Measured object rotation -> wrist_roll offset (see _grasp_wrist_roll).
        # None (every existing call site) keeps the exact validated angle.
        yaw_offset_deg = 0.0 if object_yaw_deg is None else float(object_yaw_deg)

        # NEVER-BAT, shape-based early decline: a SPHERE is a single point of contact
        # for a flat parallel pad. It rolls on the FIRST graze of the descending pad
        # and — being frictionless-round — COASTS 15-25mm before any drift-abort can
        # lift the arm clear, so even one gentle touch bats it past the budget. This is
        # a fundamental flat-pad limitation, not a tuning gap (verified: every descent
        # gentle enough to reach it still sets it rolling). So we DECLINE a sphere up
        # front WITHOUT touching it (0mm displacement) — the honest, never-bat outcome.
        # (Cylinders are still attempted: stood upright they present a graspable band
        # and the gentle close + drift-guard either holds them or declines them gently.)
        # OVERRIDE: an explicit grasp_height_m (a deliberate equator grip) opts in to
        # attempting the sphere anyway — the auto-decline only guards the auto-derive path.
        if dinfo.get("shape") == "sphere" and grasp_height_m is None:
            return MoveResult(False,
                              f"Grasp declined (not attempted): the object is a SPHERE. "
                              f"A smooth sphere is a single contact point for flat "
                              f"parallel pads — it rolls and coasts away on the lightest "
                              f"touch, so any grasp attempt would bat it past the never-"
                              f"bat budget. This is a fundamental flat-pad limitation; "
                              f"declined gently without disturbing it. {dinfo}",
                              self.backend.get_state())

        # NEVER-BAT, geometry-based early decline (P0): a SHORT, WIDE box cannot be
        # center-gripped by the flat pads. Gripped at its low center anchor, the static
        # pad lands mid-face on the wide top and the deeper seat rung LEVERS it sideways
        # — a bat (measured: a 48-60mm short box is flung 24-53mm in ONE seat rung,
        # before any per-rung drift-abort can fire). Narrower boxes the pads CAN bracket
        # (<=44mm) and tall wide boxes (gripped high on the body) are unaffected; only
        # the un-center-grippable short wide face is declined, up front WITHOUT touching
        # it (0mm displacement) — the honest never-bat outcome.
        if dinfo.get("low_grip_wide"):
            return MoveResult(False,
                              f"Grasp declined (not attempted): the object is a SHORT, "
                              f"WIDE box ({dinfo['obj_w_mm']:.0f}mm wide, "
                              f"{dinfo['obj_h_mm']:.0f}mm tall — gripped at its low "
                              f"center). A wide face this short cannot be center-gripped "
                              f"by the flat parallel pads: the off-center pad lands "
                              f"mid-face and the seating rung would lever it sideways, "
                              f"batting it past the never-bat budget. This is a "
                              f"fundamental flat-pad limitation; declined gently without "
                              f"disturbing it. {dinfo}",
                              self.backend.get_state())

        # NEVER-BAT, geometry-based early decline (P0): a box NARROWER than the validated
        # 30mm cube cannot be closed on gently — the flat moving pad travels from the wide
        # pre-open gap onto the small object and shoves it 12-19mm (a bat), at or over the
        # 15mm ceiling, at every seed that latches (verified both cells). A tall narrow box
        # topples ~38mm. There is no safe seed, so a sub-cube box is declined up front
        # WITHOUT touching it (0mm displacement) — the honest never-bat outcome.
        if dinfo.get("narrow_box"):
            return MoveResult(False,
                              f"Grasp declined (not attempted): the object is a NARROW box "
                              f"({dinfo['obj_w_mm']:.0f}mm wide — narrower than the 30mm "
                              f"cube the flat pads are calibrated for). Closing the wide-"
                              f"open jaws onto an object this small shoves it 12-19mm past "
                              f"the never-bat budget before the pads can latch; a tall "
                              f"narrow box topples further. This is a fundamental flat-pad "
                              f"limitation; declined gently without disturbing it. {dinfo}",
                              self.backend.get_state())

        # NEVER-BAT, geometry-based early decline (P0): a SHORT cylinder gripped at its
        # low center is a rolling round band the flat pads cannot hold — each one-pad
        # close + seat-lift drags it (measured: a transient 15mm walk, right at the
        # ceiling, before the budget guard can stop the sweep). A TALL cylinder is still
        # attempted (gripped high it presents a graspable band, and the gentle close +
        # drift-guard hold it or decline it gently under budget). So only the short roller
        # is declined up front WITHOUT touching it (0mm displacement).
        if dinfo.get("low_grip_round"):
            return MoveResult(False,
                              f"Grasp declined (not attempted): the object is a SHORT "
                              f"cylinder ({dinfo['obj_w_mm']:.0f}mm dia, "
                              f"{dinfo['obj_h_mm']:.0f}mm tall — gripped at its low "
                              f"center). A short round band rolls and drags under the flat "
                              f"parallel pads, walking it toward the never-bat budget on "
                              f"each one-pad close. This is a fundamental flat-pad "
                              f"limitation; declined gently without disturbing it. {dinfo}",
                              self.backend.get_state())

        def log(_m):  # feedback_pick logs to a file; default discard (no harness sink)
            if log_sink is not None:
                log_sink(_m)
        self._carry_log = log

        is_pos = y >= 0
        pos_inner_lowpan = False
        if is_pos:
            # Gated +y wrist_flex (steeper wf=42 vs the validated wf=30 basin). Two
            # disjoint +y regions need the steeper, more-vertical descent/close:
            #   (a) SHORT radius (r < _G_POS_INNER_R): the static finger arcs inward and
            #       clips the cube's outer +x top edge during the shallow-wf descent.
            #   (b) HIGH |pan| (> _G_POS_STEEP_PAN): the off-axis +y close basin only
            #       seats BOTH pads under the steeper wf (shallow wf one-pads / misses).
            # The validated LOW-pan, LONGER-radius +y column keeps wf=30 untouched — the
            # steeper wf shifts that basin and regresses those cells (and the grid).
            r_xy = math.hypot(x, y)
            pan_deg = abs(self._grasp_pan_deg(x, y))
            inner = (r_xy < _G_POS_INNER_R) or (
                pan_deg > _G_POS_STEEP_PAN and r_xy < _G_POS_STEEP_RMAX) or (
                pan_deg > _G_POS_MID_PAN and _G_POS_MID_RMIN <= r_xy < _G_POS_STEEP_RMAX)
            wf = _G_POS_WF_INNER if inner else _G_POS_WF
            # SHORT-radius, LOW-pan +y (the near-axis inner column, e.g. cell 17
            # (0.195,+0.023) and cell 20 (0.196,+0.010)): at the base seed -0.010 the
            # jaw's lateral target sits right in the descending finger's arc for the
            # y~+0.02 row, so the OPEN jaw brushes the cube on the diagonal approach and
            # the descend drift-aborts BEFORE any intended contact (measured: cell 17
            # brushed 5.1mm above the cube). A slightly more-negative seed places the
            # jaw just clear of that arc -> approach 0.0mm, descend clean, both pads
            # close ~4mm (measured across wf 42-50). The seed only shifts the lateral
            # target (no extra motion) so it is drift-free. Gated tight (r<_G_POS_INNER_R
            # AND |pan|<_G_POS_LOWPAN) so it touches only this near-axis inner pair and
            # NO workspace-grid cell (grid's nearest +y cell is |pan|=11 -> excluded).
            pos_inner_lowpan = (r_xy < _G_POS_INNER_R and pan_deg < _G_POS_LOWPAN)
        else:
            wf = _G_NEG_WF
        # width-adaptive close-basin seed (narrow objects need a more-negative lat)
        if pos_inner_lowpan:
            base_seed = _G_FINGER_LAT_POS_INNER
        elif is_pos:
            base_seed = _G_FINGER_LAT_POS
        else:
            base_seed = _G_FINGER_LAT_NEG
        finger_lat0 = base_seed + lat_offset
        sweep = _G_RETRY_LAT_STEP_POS if is_pos else _G_RETRY_LAT_STEP_NEG

        self._grasp_rehome()
        # HONEST route: trust_coords drives the grasp from the PERCEIVED (x,y,z);
        # legacy default reads the live object (validated 97% benchmark path).
        lx, ly, lz = (x, y, z) if trust_coords else self._grasp_read_cube()
        cx, cy, cz = lx, ly, lz
        ox0, oy0 = cx, cy                          # ORIGINAL object xy — the never-bat datum
        self._grasp_prepos(cx, cy, wf, is_pos, preopen_pct, yaw_offset_deg)

        def _migration_mm() -> float:
            cu = self.backend.get_state().cube_position_m
            # object still on the table -> displacement from where it started; once it
            # is grasped+aloft this is meaningless (it moves with the gripper), so the
            # budget check is only consulted while the object is on the table.
            return math.hypot(cu["x"] - ox0, cu["y"] - oy0) * 1000.0

        def _decline(reason: str) -> MoveResult:
            """End the grasp GENTLY: open, lift the EE clear, re-home, and report.
            Used the instant the object MOVES (rolls/slips) — chasing it would bat it."""
            self._grasp_lift_clear(wf)
            self._grasp_rehome()
            return MoveResult(False, reason, self.backend.get_state())

        def _try_carry(cx_, ty_) -> Optional[MoveResult]:
            """Carry a FIRM grip. Returns a success MoveResult, or None if the grip
            was lost on the lift (set down gently) so the caller declines."""
            if self._grasp_carry_lift(cx_, ty_, dinfo.get("obj_h_mm", 30.0), is_pos=is_pos):
                return MoveResult(True,
                                  f"Grasped (firm grip, finger_lat seed {finger_lat0:+.4f}, "
                                  f"{dinfo})", self.backend.get_state())
            return None

        # CONFIDENCE-DRIVEN grasp (Task #21). For each lateral seed: gentle descent +
        # IK-hold close, then check_grip. FIRM -> carry straight away (NO big re-verify
        # motion — that is what shook good grips loose). MARGINAL/EMPTY -> a bounded
        # series of TINY LOCAL re-seats (open partway, lift a few mm, shift <=4mm
        # laterally at the current height, re-close) — NOT a re-home/lift-clear/re-
        # approach. Stop the instant the grip is firm. NEVER-BAT (P0): a drift-abort or
        # a close-shove ends the grasp gently (the object rolled/slipped — chasing it
        # bats it); cumulative migration is capped at the budget.
        for attempt in range(len(sweep)):
            finger_lat = finger_lat0 + sweep[attempt]
            # Re-use the PERCEIVED coords on the honest route (do NOT re-read truth).
            lx, ly, lz = (x, y, z) if trust_coords else self._grasp_read_cube()
            cx, cy, cz = lx, ly, lz
            firm, aborted, info = self._grasp_attempt_gentle(
                cx, cy, cz, wf, finger_lat, preopen_pct, log, grasp_z=grasp_z)
            log(f"    attempt {attempt} finger_lat={finger_lat:+.4f} "
                f"firm={firm} aborted={aborted} quality={info.get('quality')}")

            # COMMIT-ON-GRIP: the moment the jaws close on the cube (any pad contact),
            # carry it out — firm OR marginal — with NO re-seat dance and NO opening a
            # held grip. A pre-grip bump (jaws still open) or a clean empty close sweeps
            # to the next seed, and does so WITHOUT re-homing (stay near the object).
            if _G_COMMIT_ON_GRIP:
                _grip = info.get("grip", {})
                _contact = (firm or bool(_grip.get("both_pads"))
                            or info.get("lifted_grasping")
                            or _grip.get("quality") == "marginal")
                if aborted or not _contact:
                    # no grip yet (bumped on descent, or empty close) -> next seed, no re-home
                    if attempt + 1 < len(sweep):
                        self._grasp_lift_clear(wf)
                        self._grasp_prepos(cx, cy, wf, is_pos, preopen_pct, yaw_offset_deg)
                        continue
                    return _decline(
                        f"Grasp declined: no grip after {attempt+1} seed(s) "
                        f"({'bumped at '+str(info.get('stage')) if aborted else 'empty close'}). "
                        f"{dinfo}")
                # gripped -> COMMIT and carry (never open a held cube on a marginal read)
                res = _try_carry(cx, info["ty"])
                if res is not None:
                    return res
                return MoveResult(False,
                    f"Committed after grip but the cube genuinely fell during the lift. "
                    f"{dinfo}", self.backend.get_state())

            # A drift-abort during descend/seat means the object MOVED on contact (it
            # rolls/slips). Retrying just shoves it again -> decline gently.
            if aborted:
                return _decline(
                    f"Grasp declined: object moved on contact (drift-abort at the "
                    f"{info.get('stage')} stage — it rolls/slips under the flat pads). "
                    f"Stopped gently after {attempt+1} attempt(s) without chasing it. "
                    f"{dinfo}")

            # FIRM grip -> carry straight away (no re-verify lift).
            if firm:
                res = _try_carry(cx, info["ty"])
                if res is not None:
                    return res
                log("    lost grasp during staged carry -> set down gently, decline")
                return _decline(
                    f"Grasp declined: object slipped out of the parallel pads during the "
                    f"lift (marginal grip the flat pads could not hold securely). It was "
                    f"set down gently from a low height, not dropped. {dinfo}")

            # NOT firm. Two cases:
            #  (a) we HAVE a two-pad grip but it is marginal/unstable -> a TINY LOCAL
            #      re-seat (open partway, lift clear, shift <=4mm laterally over the top,
            #      re-close) can firm it WITHOUT the far re-home. Stop the instant firm.
            #  (b) a ONE-PAD catch or empty close -> re-seating in place would re-close on
            #      the same off-center contact (the re-seat shift drifts-aborts when one
            #      pad is still touching), so sweep the next lateral SEED via a fresh
            #      gentle over-the-top re-approach. The widened -y sweep reaches both the
            #      more-negative (-0.017/-0.020) and less-negative (-0.010/-0.008) seeds,
            #      so the short-radius cells that seat both pads only off the base seed get
            #      a clean both-pads close within the never-bat budget (no in-place drag).
            ty = info["ty"]
            g = info.get("grip", {})
            have_grip = bool(g.get("both_pads"))
            if have_grip:
                # nudge toward more-negative finger_lat first (the measured basin
                # direction), bracketing with a + step if that overshoots.
                nudge_dirs = [-1.0, +1.0]
                for k in range(_G_NUDGE_MAX):
                    if _migration_mm() > _G_BAT_BUDGET_MM:
                        return MoveResult(False,
                                          f"Grasp declined: object migrated "
                                          f"{_migration_mm():.1f}mm — past the "
                                          f"{_G_BAT_BUDGET_MM:.0f}mm never-bat budget. Stopped "
                                          f"gently before it moves further. {dinfo}",
                                          self.backend.get_state())
                    lat_delta = nudge_dirs[k % len(nudge_dirs)] * (_G_NUDGE_LAT_MM / 1000.0)
                    rfirm, rabort, rdrift, rinfo = self._grasp_reseat(
                        cx, ty, cz if grasp_z is None else grasp_z, wf,
                        lat_delta, ox0, oy0, log)
                    if rabort:
                        return _decline(
                            f"Grasp declined: object moved {rdrift:.1f}mm during a tiny "
                            f"re-seat ({rinfo.get('stage')}) — it rolls/slips under the flat "
                            f"pads. Stopped gently without chasing it. {dinfo}")
                    ty = rinfo["ty"]
                    if rfirm:
                        res = _try_carry(cx, ty)
                        if res is not None:
                            return res
                        return _decline(
                            f"Grasp declined: re-seated to a firm grip but it slipped on the "
                            f"lift (the flat pads could not hold it securely). Set down "
                            f"gently. {dinfo}")
            # Miss, or tiny re-seats exhausted. If sweep values remain, do a fresh gentle
            # over-the-top re-approach at the next seed (lift clear + re-prepos); the
            # budget guard caps cumulative migration.
            if attempt + 1 < len(sweep):
                self._grasp_lift_clear(wf)
                self._grasp_rehome()
                walk = _migration_mm()
                if walk > _G_BAT_BUDGET_MM:
                    return MoveResult(False,
                                      f"Grasp declined: object migrated {walk:.1f}mm over "
                                      f"{attempt+1} attempt(s) — past the {_G_BAT_BUDGET_MM:.0f}mm "
                                      f"never-bat budget. Stopped gently. {dinfo}",
                                      self.backend.get_state())
                self._grasp_prepos(cx, cy, wf, is_pos, preopen_pct, yaw_offset_deg)

        # Exhausted -> give up GENTLY. No pick_object fallback by design (its wrist-snap
        # close flings the object when it misses, violating never-bat).
        return MoveResult(False,
                          f"Grasp failed gently after {len(sweep)} seeds + tiny re-seats "
                          f"(gentle-only, never-bat). {dinfo}",
                          self.backend.get_state())

    def execute_intuitive_move(self,
                               move_gripper_up_mm: float = 0,
                               move_gripper_forward_mm: float = 0,
                               tilt_gripper_down_angle: float = 0,
                               rotate_gripper_counterclockwise_angle: float = 0,
                               rotate_robot_left_angle: float = 0,
                               gain: float = 0.5,
                               lock_pan: bool = True) -> MoveResult:
        """Intuitive delta move: spatial mm offsets + direct joint angle deltas.

        Positive rotate_robot_left_angle turns the base toward +y (left).
        Positive tilt_gripper_down_angle tips the gripper nose downward.
        Positive move_gripper_forward_mm moves in the arm's current facing direction.
        """
        with self._lock:
            n = self.ik.n_arm
            ga = self._ga(self.backend.get_state().gripper_openness_pct)

            # 1. Base rotation (shoulder_pan delta).
            # Positive left_angle → more negative pan (since positive pan faces -y).
            if rotate_robot_left_angle != 0:
                lo, hi = float(self.ik.model.jnt_range[0, 0]), float(self.ik.model.jnt_range[0, 1])
                start_pan = float(self.ik.data.qpos[0])
                end_pan = np.clip(start_pan - np.radians(rotate_robot_left_angle), lo, hi)
                g_ctrl = float(self.ik.data.ctrl[5])
                for s in range(80):
                    t = (s + 1) / 80
                    ctrl = self.ik.data.qpos[:n].copy()
                    ctrl[0] = start_pan + (end_pan - start_pan) * t
                    self.backend.apply_control(np.append(ctrl, g_ctrl))
                    self.backend.step()

            # 2. Cartesian forward/up IK.
            # lock_pan=True (default): only shoulder_lift + elbow_flex active.
            # lock_pan=False: shoulder_pan also free — needed for descending into +y positions.
            if move_gripper_up_mm != 0 or move_gripper_forward_mm != 0:
                pan = float(self.ik.data.qpos[0])
                ee = self.ik.get_ee_position()
                dx = (move_gripper_forward_mm / 1000.0) * np.cos(pan)
                dy = (move_gripper_forward_mm / 1000.0) * (-np.sin(pan))
                dz = move_gripper_up_mm / 1000.0
                target = ee + np.array([dx, dy, dz])
                locked = _LOCKED_PAN_AND_WRIST if lock_pan else _LOCKED_WRIST
                self._move(target, ga, max_steps=400, gain=gain, locked=locked)

            # 3. Wrist tilt and roll deltas.
            if tilt_gripper_down_angle != 0 or rotate_gripper_counterclockwise_angle != 0:
                wf_lo, wf_hi = float(self.ik.model.jnt_range[3, 0]), float(self.ik.model.jnt_range[3, 1])
                wr_lo, wr_hi = float(self.ik.model.jnt_range[4, 0]), float(self.ik.model.jnt_range[4, 1])
                start_wf = float(self.ik.data.qpos[3])
                start_wr = float(self.ik.data.qpos[4])
                end_wf = np.clip(start_wf + np.radians(tilt_gripper_down_angle), wf_lo, wf_hi)
                end_wr = np.clip(start_wr + np.radians(rotate_gripper_counterclockwise_angle), wr_lo, wr_hi)
                g_ctrl = float(self.ik.data.ctrl[5])
                for s in range(80):
                    t = (s + 1) / 80
                    ctrl = self.ik.data.qpos[:n].copy()
                    ctrl[3] = start_wf + (end_wf - start_wf) * t
                    ctrl[4] = start_wr + (end_wr - start_wr) * t
                    self.backend.apply_control(np.append(ctrl, g_ctrl))
                    self.backend.step()

        return MoveResult(True, "Intuitive move applied", self.backend.get_state())

    def run_pick_and_place(self, cube_pos: Optional[np.ndarray] = None,
                           container_pos: Optional[np.ndarray] = None) -> MoveResult:
        self.backend.reset(cube_pos, container_pos)
        self.backend.on_pick_and_place_start()

        with self._lock:
            cube, container = self.backend.get_object_positions()
            locked = _LOCKED_WRIST

            # Phase 1 — open gripper
            for _ in range(20):
                self._step(self.ik.get_ee_position(), _GRIPPER_OPEN)

            # Phase 2 — above cube
            above = cube.copy()
            above[2] += _GRASP_Z_OFFSET + _APPROACH_Z
            above[1] += _FINGER_Y_OFFSET
            self._move(above, _GRIPPER_OPEN)

            # Phase 3 — lower to grasp
            grasp_pos = cube.copy()
            grasp_pos[2] += _GRASP_Z_OFFSET
            grasp_pos[1] += _FINGER_Y_OFFSET
            self._move(grasp_pos, _GRIPPER_OPEN, tol=0.008)

            # Phase 4 — close gripper (IK-active throughout)
            contact_step = contact_gripper = None
            grasp_gripper = _GRIPPER_CLOSED
            for step in range(300):
                if contact_step is None:
                    gripper = _GRIPPER_OPEN - 2.0 * min(step / 250, 1.0)
                else:
                    tgt = max(contact_gripper - _TIGHTEN, -1.0)
                    gripper = contact_gripper + (tgt - contact_gripper) * min((step - contact_step) / 100, 1.0)

                ctrl = self.ik.step_toward_target(grasp_pos, gripper_action=gripper,
                                                   gain=0.5, locked_joints=locked)
                ctrl[3] = np.pi / 2
                ctrl[4] = np.pi / 2
                self.backend.apply_control(ctrl)
                self.backend.step()

                if self.backend.is_grasping() and contact_step is None:
                    contact_step, contact_gripper = step, gripper
                if contact_step is not None:
                    tgt = max(contact_gripper - _TIGHTEN, -1.0)
                    if gripper <= tgt + 0.01:
                        grasp_gripper = gripper
                        break
            else:
                if contact_step is not None:
                    grasp_gripper = gripper

            # Phase 5 — lift (interpolated)
            lift_pos = cube.copy()
            lift_pos[2] = _LIFT_Z
            for s in range(300):
                t = min(s / 250, 1.0)
                tgt = grasp_pos + (lift_pos - grasp_pos) * t
                ctrl = self.ik.step_toward_target(tgt, gripper_action=grasp_gripper,
                                                   gain=0.3, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # Phase 6 — hold at lift
            for _ in range(100):
                ctrl = self.ik.step_toward_target(lift_pos, gripper_action=grasp_gripper,
                                                   gain=0.3, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # Phase 7 — arc to above container
            above_ctr = container.copy()
            above_ctr[2] = _ABOVE_CTR_Z
            self._move(above_ctr, grasp_gripper, max_steps=500, tol=0.025)

            # Phase 8 — descend into container
            place_pos = container.copy()
            place_pos[2] = _PLACE_Z
            for s in range(250):
                t = min(s / 200, 1.0)
                tgt = above_ctr + (place_pos - above_ctr) * t
                ctrl = self.ik.step_toward_target(tgt, gripper_action=grasp_gripper,
                                                   gain=0.4, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # Phase 9 — release
            for s in range(50):
                g = grasp_gripper + (_GRIPPER_OPEN - grasp_gripper) * (s / 50)
                ctrl = self.ik.step_toward_target(place_pos, gripper_action=g,
                                                   gain=0.5, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

            # Phase 10 — withdraw
            for s in range(150):
                t = s / 150
                tgt = place_pos + (above_ctr - place_pos) * t
                ctrl = self.ik.step_toward_target(tgt, gripper_action=_GRIPPER_OPEN,
                                                   gain=0.3, locked_joints=locked)
                self.backend.apply_control(ctrl)
                self.backend.step()

        return MoveResult(True, "Pick-and-place complete", self.backend.get_state())
