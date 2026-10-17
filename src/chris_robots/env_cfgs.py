"""Velocity-tracking environment for any RobotDef, on flat ground.

Starts from mjlab's ``make_velocity_env_cfg()`` (the G1/Go1 recipe) and plugs in a
robot. Everything size-dependent (speeds, foot clearance, pushes, CoM
randomization) is scaled from the RobotDef so the same code serves a 1.4 m,
36 kg humanoid and a 0.4 m, 3 kg one.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
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

  v = robot.walk_vel  # nominal walking speed: the unit for thresholds/stds
  run = robot.max_lin_vel
  w = robot.max_ang_vel

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = robot.standing_height + 0.25
  twist.ranges.lin_vel_x = (-v, run)
  twist.ranges.lin_vel_y = (-0.6 * v, 0.6 * v)
  twist.ranges.ang_vel_z = (-w, w)

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
  # Design goal: an *animated character*, not a controller. These policies only ever
  # run in a browser, so we trade robustness for drama: big strides, visible foot
  # lift, arm swing, a forward lean that grows with speed, and a torso that is
  # allowed to move. Falling is cheap (episode ends) rather than penalised.
  assert robot.posture is not None
  cfg.rewards["pose"].params["std_standing"] = dict(robot.posture.standing)
  cfg.rewards["pose"].params["std_walking"] = dict(robot.posture.walking)
  cfg.rewards["pose"].params["std_running"] = dict(robot.posture.running)
  cfg.rewards["pose"].params["walking_threshold"] = 0.1 * v
  cfg.rewards["pose"].params["running_threshold"] = 1.6 * v
  cfg.rewards["pose"].weight = 0.5

  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5 * v

  # No "keep the torso level" terms: upright -> speed-dependent lean, and the
  # angular-velocity / angular-momentum penalties go entirely.
  cfg.rewards.pop("upright")
  cfg.rewards.pop("body_ang_vel")
  cfg.rewards.pop("angular_momentum")
  cfg.rewards["lean"] = RewardTermCfg(
    func=chris_mdp.lean,
    weight=1.0,
    params={
      "command_name": "twist",
      "run_vel": run,
      "lean_at_run": robot.lean_at_run,
      "std": 0.35,
    },
  )
  cfg.rewards["action_rate_l2"].weight = -0.05

  # Gait: a phase clock prescribes *when* each foot is on the ground (left and right
  # half a period apart, duty factor 0.6 walking -> 0.4 running) and how high the
  # swing foot should arc. Free-form shaping terms (air time, stride length) were
  # exploited: the policy planted one foot, hovered the other and skated on a foot
  # corner. A schedule leaves no such option, and the gait's period/symmetry are then
  # design parameters instead of emergent accidents.
  left_index = robot.foot_order.index("left")
  cfg.commands["gait_clock"] = chris_mdp.GaitClockCommandCfg(
    walk_vel=v,
    run_vel=run,
    period_walk=robot.gait_period,
    period_run=robot.gait_period_run,
    duty_walk=0.6,
    duty_run=0.4,
    standing_threshold=0.1 * v,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["gait_clock"] = ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "gait_clock"}
    )
  for name in ("foot_clearance", "foot_swing_height", "air_time"):
    cfg.rewards.pop(name)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = robot.foot_sites
  cfg.rewards["foot_slip"].weight = -0.5
  for name in ("foot_slip", "soft_landing"):
    cfg.rewards[name].params["command_threshold"] = 0.1 * v
  cfg.rewards["gait_schedule"] = RewardTermCfg(
    func=chris_mdp.gait_contact_schedule,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "clock_name": "gait_clock",
      "left_index": left_index,
    },
  )
  cfg.rewards["gait_swing"] = RewardTermCfg(
    func=chris_mdp.gait_swing_height,
    weight=1.0,
    params={
      "sensor_name": "foot_height_scan",
      "clock_name": "gait_clock",
      "left_index": left_index,
      "target_height": robot.foot_clearance,
    },
  )
  cfg.rewards["foot_flat"] = RewardTermCfg(
    func=chris_mdp.foot_flat,
    weight=-5.0,
    params={"site_names": robot.foot_sites, "up_local": robot.foot_up_axes()},
  )

  pairs = robot.arm_swing_pairs() if robot.arm_swing_gain > 0 else ()
  if pairs:
    cfg.rewards["arm_swing"] = RewardTermCfg(
      func=chris_mdp.arm_swing,
      weight=0.5,
      params={
        "pairs": pairs,
        "gain": robot.arm_swing_gain,
        "std": 0.3,
        "command_name": "twist",
        "command_threshold": 0.1 * v,
      },
    )

  # ---- curriculum: start at a jog, open up to the full sprint range early ----------
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-v, 0.5 * run), "ang_vel_z": (-w, w)},
    {"step": 800 * STEPS_PER_ITER, "lin_vel_x": (-v, run), "ang_vel_z": (-w, w)},
  ]

  # ---- play-mode overrides ---------------------------------------------------------
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist.ranges.lin_vel_x = (-v, run)
    twist.ranges.ang_vel_z = (-w, w)

  return cfg
