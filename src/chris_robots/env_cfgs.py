"""Velocity-tracking environment for any RobotDef, on flat ground.

Starts from mjlab's ``make_velocity_env_cfg()`` (the G1/Go1 recipe) and plugs in a
robot. Everything size-dependent (speeds, foot clearance, pushes, CoM
randomization) is scaled from the RobotDef so the same code serves a 1.4 m,
36 kg humanoid and a 0.4 m, 3 kg one.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from chris_robots import mdp as chris_mdp
from chris_robots.robots.base import RobotDef

STEPS_PER_ITER = 24  # rsl-rl num_steps_per_env


def make_flat_env_cfg(robot: RobotDef, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  # ---- sim settings (flat terrain is cheap) ----------------------------------
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": robot.entity_cfg()}

  # ---- flat terrain: drop the terrain generator + height scan -----------------
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  # ---- sensors -----------------------------------------------------------------
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in robot.foot_sites)
      sensor.pattern = RingPatternCfg.single_ring(
        radius=0.03 * robot.standing_height / 0.8, num_samples=6
      )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=robot.foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  # ---- actions -------------------------------------------------------------------
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = robot.action_scale

  # ---- viewer / viz --------------------------------------------------------------
  cfg.viewer.body_name = robot.base_body
  cfg.viewer.distance = max(1.5, 2.5 * robot.standing_height)
  cfg.viewer.elevation = -10.0

  v = robot.max_lin_vel
  w = robot.max_ang_vel

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = robot.standing_height + 0.25
  twist.ranges.lin_vel_x = (-v, v)
  twist.ranges.lin_vel_y = (-0.6 * v, 0.6 * v)
  twist.ranges.ang_vel_z = (-0.5 * w, 0.5 * w)

  # ---- events (domain randomization) ---------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = (robot.base_body,)
  cfg.events["base_com"].params["ranges"] = {
    0: (-robot.com_offset, robot.com_offset),
    1: (-robot.com_offset, robot.com_offset),
    2: (-1.2 * robot.com_offset, 1.2 * robot.com_offset),
  }
  p = robot.push_vel
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-p, p),
    "y": (-p, p),
    "z": (-0.8 * p, 0.8 * p),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
  }

  # ---- rewards ---------------------------------------------------------------------
  cfg.rewards["pose"].params["std_standing"] = dict(robot.posture.standing)
  cfg.rewards["pose"].params["std_walking"] = dict(robot.posture.walking)
  cfg.rewards["pose"].params["std_running"] = dict(robot.posture.running)
  # Speed regime thresholds scale with the robot's speed range.
  cfg.rewards["pose"].params["walking_threshold"] = 0.05 * v
  cfg.rewards["pose"].params["running_threshold"] = 1.5 * v

  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25) * v
  cfg.rewards["upright"].params["asset_cfg"].body_names = (robot.base_body,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (robot.base_body,)
  cfg.rewards["body_ang_vel"].weight = robot.body_ang_vel_weight
  cfg.rewards["angular_momentum"].weight = robot.angular_momentum_weight

  # Gait shaping. With mjlab's defaults our flat box soles learn a fast double-support
  # shuffle (~1 cm foot lift, no lateral weight shift): the swing-height penalty is
  # worth ~0.01/step against a 2.0/step velocity reward. Lean harder on clearance and
  # add Isaac Lab's biped air-time reward, which pays for long single-stance phases.
  # Durations scale with leg length (a 0.35 m robot's natural cadence is ~1.7x faster
  # than a 1 m one).
  swing_t = 0.4 * math.sqrt(robot.standing_height)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = robot.foot_sites
  cfg.rewards["foot_clearance"].params["target_height"] = robot.foot_clearance
  cfg.rewards["foot_clearance"].weight = -4.0
  cfg.rewards["foot_swing_height"].params["target_height"] = robot.foot_clearance
  cfg.rewards["foot_swing_height"].weight = -1.0
  for name in ("foot_clearance", "foot_swing_height", "foot_slip", "soft_landing"):
    cfg.rewards[name].params["command_threshold"] = 0.1 * v
  cfg.rewards.pop("air_time")
  cfg.rewards["air_time_biped"] = RewardTermCfg(
    func=chris_mdp.feet_air_time_biped,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": swing_t,
      "command_name": "twist",
      "command_threshold": 0.1 * v,
    },
  )

  # ---- curriculum: widen the command range once the robot can walk ---------------
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-v, v), "ang_vel_z": (-0.5 * w, 0.5 * w)},
    {"step": 2500 * STEPS_PER_ITER, "lin_vel_x": (-v, 1.5 * v), "ang_vel_z": (-w, w)},
  ]

  # ---- play-mode overrides ---------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist.ranges.lin_vel_x = (-v, 1.5 * v)
    twist.ranges.ang_vel_z = (-w, w)

  return cfg
