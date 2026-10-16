"""Shared robot definition used by every robot in this repo.

Each robot directory (``robots/<name>/``) carries a ``robot.json`` written by the
importer with the physical facts (joints, ranges, armature, init pose, feet). The
Python side adds what RL needs: actuator gains, posture tolerances, how far/fast
the robot should be asked to walk, and a few size-dependent scalings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
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


@dataclass
class RobotDef:
  name: str
  actuators: tuple[ActuatorGroup, ...]
  posture: PostureStd
  action_scale: float | dict[str, float]
  """Joint-position action scale (rad per unit action)."""
  max_lin_vel: float
  """Max forward speed command used in training/play (m/s). Sets the scale of all
  velocity-related ranges, stds and curriculum stages."""
  max_ang_vel: float = 1.0
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
  foot_sites: tuple[str, str] = ("left_foot", "right_foot")
  foot_geoms: tuple[str, str] = ("left_foot_collision", "right_foot_collision")
  foot_friction: float = 0.8
  body_ang_vel_weight: float = -0.05
  angular_momentum_weight: float = -0.02

  # ---- derived from robot.json ---------------------------------------------
  def __post_init__(self) -> None:
    self.dir: Path = ROBOTS_DIR / self.name
    self.xml: Path = self.dir / f"{self.name}.xml"
    assert self.xml.exists(), self.xml
    self.info: dict = json.loads((self.dir / "robot.json").read_text())
    self.joint_names: tuple[str, ...] = tuple(self.info["joints"])
    self.foot_bodies: tuple[str, ...] = tuple(
      self.info["feet"][s]["body"] for s in ("left", "right")
    )
    self.standing_height: float = float(self.info["standing_base_height_m"])
    self.display_name: str = self.info["display_name"]

  @property
  def task_id(self) -> str:
    return "Chris-Velocity-Flat-" + "".join(p.capitalize() for p in self.name.split("_"))

  def get_spec(self) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(self.xml))

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
