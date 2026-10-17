"""Shared robot definition used by every robot in this repo.

Each robot directory (``robots/<name>/``) carries a ``robot.json`` written by the
importer with the physical facts (joints, ranges, armature, init pose, feet). The
Python side adds what RL needs: actuator gains, posture tolerances, how far/fast
the robot should be asked to walk, and a few size-dependent scalings.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from chris_robots import ROBOTS_DIR

# mjlab's G1 recipe: PD gains derived from reflected inertia, 10 Hz, zeta = 2.
NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def pd_from_armature(armature: float) -> tuple[float, float]:
  kp = armature * NATURAL_FREQ**2
  kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
  return kp, kd


@dataclass(frozen=True)
class ActuatorGroup:
  """One motor type: which joints it drives and how it behaves."""

  name: str
  joints: tuple[str, ...]
  """Regexes matched against joint names."""
  effort_limit: float
  armature: float
  kp: float | None = None
  kd: float | None = None
  """If kp/kd are None they're derived from the armature (see pd_from_armature)."""

  def gains(self) -> tuple[float, float]:
    kp, kd = pd_from_armature(self.armature)
    return (self.kp if self.kp is not None else kp, self.kd if self.kd is not None else kd)

  def to_cfg(self) -> BuiltinPositionActuatorCfg:
    kp, kd = self.gains()
    return BuiltinPositionActuatorCfg(
      target_names_expr=self.joints,
      stiffness=kp,
      damping=kd,
      effort_limit=self.effort_limit,
      armature=self.armature,
    )


@dataclass(frozen=True)
class PostureStd:
  """Per-joint posture tolerance (std of a Gaussian reward) for each speed regime.

  Keys are regexes; every joint must match exactly one key in each dict.
  """

  standing: dict[str, float]
  walking: dict[str, float]
  running: dict[str, float]


# Default posture tolerances (rad) for the "animated character" gait: pattern ->
# (walking std, running std). Deliberately loose on the legs and shoulders so the
# posture term only vetoes silly poses, not big strides. Standing is always tight.
DYNAMIC_POSTURE: dict[str, tuple[float, float]] = {
  r".*hip_pitch.*": (0.7, 1.2),
  r".*hip_roll.*": (0.3, 0.4),
  r".*hip_yaw.*": (0.3, 0.4),
  r".*knee.*": (0.9, 1.4),
  r".*ankle.*": (0.3, 0.4),  # tighter: loose ankles let the policy stand on foot edges
  r".*shoulder_pitch.*": (0.8, 1.2),
  # Arms swing front/back only: lateral shoulder axes and wrists stay near default.
  r".*shoulder_roll.*": (0.1, 0.12),
  r".*shoulder_yaw.*": (0.1, 0.12),
  r".*elbow.*": (0.5, 0.8),  # pitch-type elbows; lateral elbows are tightened in __post_init__
  r".*wrist.*": (0.15, 0.2),
  r".*gripper.*": (0.05, 0.05),
}
_FALLBACK_POSTURE = (0.3, 0.5)
# Loose enough that the Gaussian pose reward still has a gradient when an arm is left
# 0.5 rad off after a walk (0.05 saturated to ~0 and the policy ignored it).
STANDING_POSTURE_STD = 0.15


def dynamic_posture(
  joint_names: tuple[str, ...],
  scale: float = 1.0,
  overrides: dict[str, tuple[float, float]] | None = None,
) -> PostureStd:
  """Build a PostureStd (explicit per-joint keys) from DYNAMIC_POSTURE.

  ``overrides`` maps exact joint names to (walking, running) stds and wins over the
  regex table.
  """
  walking: dict[str, float] = {}
  running: dict[str, float] = {}
  for n in joint_names:
    if overrides and n in overrides:
      w, r = overrides[n]
    else:
      hits = [p for p in DYNAMIC_POSTURE if re.fullmatch(p, n)]
      if len(hits) > 1:
        raise ValueError(f"joint {n} matches several posture patterns: {hits}")
      w, r = DYNAMIC_POSTURE[hits[0]] if hits else _FALLBACK_POSTURE
    walking[n], running[n] = w * scale, r * scale
  return PostureStd(standing={".*": STANDING_POSTURE_STD}, walking=walking, running=running)


@dataclass
class RobotDef:
  name: str
  actuators: tuple[ActuatorGroup, ...]
  action_scale: float | dict[str, float]
  """Joint-position action scale (rad per unit action)."""
  walk_vel: float
  """Nominal walking speed (m/s). Sets tracking std, lateral/backward ranges, speed
  regime thresholds and the eval 'walk' segment."""
  max_lin_vel: float
  """Top (running) forward speed command (m/s)."""
  max_ang_vel: float = 1.0
  posture: PostureStd | None = None
  """None -> dynamic_posture(joint_names)."""
  posture_scale: float = 1.0
  foot_clearance: float = 0.08
  """Target swing-foot height (m)."""
  push_vel: float = 0.5
  """Magnitude of random base velocity pushes (m/s)."""
  com_offset: float = 0.02
  """Base CoM randomization range (m)."""
  init_bend: float | None = None
  """Override the importer's knees-bent init pose (radians of knee flexion / 2)."""
  extra_init_joint_pos: dict[str, float] = field(default_factory=dict)
  """Additional init joint positions (regex -> value), e.g. arm pose."""
  base_body: str = "base"
  foot_friction: float = 0.8
  arm_swing_gain: float = 1.0
  """Cosmetic: reward each shoulder pitch for following the opposite hip pitch with
  this gain (0 disables the term)."""
  lean_at_run: float = 0.2
  """Forward torso lean (rad) rewarded at max_lin_vel, scaled linearly with the
  commanded forward speed."""
  gait_period: float | None = None
  """Gait clock period while walking (s, one full left+right cycle). None ->
  0.9 * sqrt(standing_height) (Froude scaling of a human ~0.9 s stride at 1 m hip)."""
  gait_period_run: float | None = None
  """Gait clock period at max_lin_vel. None -> 0.65 * gait_period."""
  hip_adduction_min: float = 0.0
  """Sim-only: widen the adduction side (the smaller side) of every hip-roll joint range
  to at least this many radians. Several of these robots only allow 0.1-0.2 rad of
  adduction, which makes it impossible to shift the pelvis over the stance foot, and the
  policy learns a shuffle instead of a walk."""

  # ---- derived from robot.json ---------------------------------------------
  def __post_init__(self) -> None:
    self.dir: Path = ROBOTS_DIR / self.name
    self.xml: Path = self.dir / f"{self.name}.xml"
    assert self.xml.exists(), self.xml
    self.info: dict = json.loads((self.dir / "robot.json").read_text())
    self.joint_names: tuple[str, ...] = tuple(self.info["joints"])
    # Feet in *model* order. mjlab resolves name patterns in model order (contact
    # sensor primaries, SceneEntityCfg site ids), so the height-scan frames and
    # everything else per-foot must use the same order or rewards pair the wrong feet.
    if "foot_order" in self.info:
      self.foot_order: tuple[str, ...] = tuple(self.info["foot_order"])
    else:  # older robot.json: derive it from the compiled model
      model = mujoco.MjModel.from_xml_path(str(self.xml))
      self.foot_order = tuple(
        sorted(
          ("left", "right"),
          key=lambda s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot"),
        )
      )
    self.foot_sites: tuple[str, ...] = tuple(f"{s}_foot" for s in self.foot_order)
    self.foot_geoms: tuple[str, ...] = tuple(f"{s}_foot_collision" for s in self.foot_order)
    self.foot_bodies: tuple[str, ...] = tuple(self.info["feet"][s]["body"] for s in self.foot_order)
    self.standing_height: float = float(self.info["standing_base_height_m"])
    self.display_name: str = self.info["display_name"]
    if self.posture is None:
      # Elbows that don't flex in the sagittal plane (axis not ~y) get the lateral
      # treatment: they should stay still rather than flap sideways.
      pitch_elbows = set(self.pitch_elbows())
      overrides = {
        n: (0.1, 0.12) for n in self.joint_names if "elbow" in n and n not in pitch_elbows
      }
      self.posture = dynamic_posture(self.joint_names, self.posture_scale, overrides)
    if self.gait_period is None:
      self.gait_period = 0.9 * math.sqrt(self.standing_height)
    if self.gait_period_run is None:
      self.gait_period_run = 0.65 * self.gait_period
    if self.init_bend is not None:
      # Re-derive the knees-bent init pose with a different bend (lazy import: the
      # importer module imports this package).
      from chris_robots.tools.import_kscale import compute_init_pose

      model = mujoco.MjModel.from_xml_path(str(self.xml))
      self.info["init_pose"] = compute_init_pose(model, bend=self.init_bend)

  def init_qpos(self, model: mujoco.MjModel) -> list[float]:
    """Full qpos (free joint + hinges in model order) of the init pose."""
    pose = self.info["init_pose"]
    joint_pos = dict(pose["joint_pos"])
    for pattern, value in self.extra_init_joint_pos.items():
      for name in self.joint_names:
        if re.fullmatch(pattern, name):
          joint_pos[name] = value
    qpos = list(pose["base_pos"]) + [1.0, 0.0, 0.0, 0.0]
    for i in range(model.njnt):
      if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
        qpos.append(joint_pos.get(model.joint(i).name, 0.0))
    return qpos

  def foot_up_axes(self) -> tuple[tuple[float, float, float], ...]:
    """World-up expressed in each foot site's frame at the init pose (foot_order)."""
    model = mujoco.MjModel.from_xml_path(str(self.xml))
    data = mujoco.MjData(model)
    data.qpos[:] = self.init_qpos(model)
    mujoco.mj_forward(model, data)
    out = []
    for name in self.foot_sites:
      sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
      R = data.site_xmat[sid].reshape(3, 3)
      up = R.T @ np.array([0.0, 0.0, 1.0])
      out.append(tuple(float(x) for x in up))
    return tuple(out)

  def _find_joint(self, side: str, part: str) -> str | None:
    for n in self.joint_names:
      if side in n and part in n:
        return n
    return None

  def _axis_y(self, joint: str) -> float:
    return float(self.info["joint_info"][joint]["world_axis_at_init"][1])

  def pitch_elbows(self) -> tuple[str, ...]:
    """Elbow joints whose axis is ~lateral (they flex in the sagittal plane)."""
    return tuple(n for n in self.joint_names if "elbow" in n and abs(self._axis_y(n)) >= 0.7)

  def lateral_arm_joints(self) -> tuple[str, ...]:
    """Arm joints that move the arm sideways/twist it: should stay quiet while walking."""
    pitch_elbows = set(self.pitch_elbows())
    out = []
    for n in self.joint_names:
      if "shoulder_roll" in n or "shoulder_yaw" in n or "wrist" in n:
        out.append(n)
      elif "elbow" in n and n not in pitch_elbows:
        out.append(n)
    return tuple(out)

  def arm_swing_pairs(self) -> tuple[tuple[str, str, float, float], ...]:
    """(shoulder_pitch, opposite hip_pitch, shoulder sign, hip sign) per side.

    Signs are sign(world joint axis . y) at the init pose, so that
    ``-sign * (q - q_default)`` means "limb forward" for every pitch joint.
    """
    pairs = []
    for arm_side, leg_side in (("left", "right"), ("right", "left")):
      sh = self._find_joint(arm_side, "shoulder_pitch")
      hip = self._find_joint(leg_side, "hip_pitch")
      if sh is None or hip is None:
        return ()
      s_sh, s_hip = self._axis_y(sh), self._axis_y(hip)
      if abs(s_sh) < 0.7 or abs(s_hip) < 0.7:
        return ()
      pairs.append((sh, hip, float(np.sign(s_sh)), float(np.sign(s_hip))))
    return tuple(pairs)

  def elbow_swing_pairs(self) -> tuple[tuple[str, str, float, float], ...]:
    """(elbow, same-side shoulder_pitch, elbow sign, shoulder sign) for pitch-type elbows.

    Convention as above: ``-sign * (q - q_default)`` = forearm/arm forward, so the
    elbow flexes a little as the arm swings forward and extends on the back-swing.
    """
    pairs = []
    for side in ("left", "right"):
      el = self._find_joint(side, "elbow")
      sh = self._find_joint(side, "shoulder_pitch")
      if el is None or sh is None or el not in self.pitch_elbows() or abs(self._axis_y(sh)) < 0.7:
        continue
      pairs.append((el, sh, float(np.sign(self._axis_y(el))), float(np.sign(self._axis_y(sh)))))
    return tuple(pairs)

  @property
  def task_id(self) -> str:
    return "Chris-Velocity-Flat-" + "".join(p.capitalize() for p in self.name.split("_"))

  def get_spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(self.xml))
    self.apply_joint_overrides(spec)
    return spec

  def apply_joint_overrides(self, spec: mujoco.MjSpec) -> None:
    """Apply sim-only joint range tweaks to a spec containing this robot."""
    if self.hip_adduction_min > 0:
      for j in spec.joints:
        if re.fullmatch(r".*hip_roll.*", j.name):
          lo, hi = (float(x) for x in j.range)
          if abs(lo) < abs(hi):
            lo = min(lo, -self.hip_adduction_min)
          else:
            hi = max(hi, self.hip_adduction_min)
          j.range = [lo, hi]

  def init_state(self) -> EntityCfg.InitialStateCfg:
    pose = self.info["init_pose"]
    joint_pos = dict(pose["joint_pos"])
    joint_pos.update(self.extra_init_joint_pos)
    return EntityCfg.InitialStateCfg(
      pos=tuple(pose["base_pos"]),
      joint_pos=joint_pos,
      joint_vel={".*": 0.0},
    )

  def collision_cfg(self) -> CollisionCfg:
    # Feet only: no self collisions, no body/terrain collisions except the soles.
    # Lean and fast; fine for a sim-only / web demo.
    return CollisionCfg(
      geom_names_expr=self.foot_geoms,
      contype=0,
      conaffinity=1,
      condim=3,
      priority=1,
      friction=(self.foot_friction,),
    )

  def articulation(self) -> EntityArticulationInfoCfg:
    return EntityArticulationInfoCfg(
      actuators=tuple(a.to_cfg() for a in self.actuators),
      soft_joint_pos_limit_factor=0.9,
    )

  def entity_cfg(self) -> EntityCfg:
    return EntityCfg(
      init_state=self.init_state(),
      collisions=(self.collision_cfg(),),
      spec_fn=self.get_spec,
      articulation=self.articulation(),
    )

  def describe(self) -> str:
    lines = [f"{self.display_name} ({self.name}): {len(self.joint_names)} DoF, task {self.task_id}"]
    for a in self.actuators:
      kp, kd = a.gains()
      lines.append(
        f"  {a.name:12s} kp={kp:7.2f} kd={kd:6.2f} tau={a.effort_limit:6.1f} arm={a.armature:.4f} joints={a.joints}"
      )
    return "\n".join(lines)
