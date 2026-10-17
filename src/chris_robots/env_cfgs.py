"""Velocity-tracking environment for any RobotDef, on flat ground.

Starts from mjlab's ``make_velocity_env_cfg()`` (the G1/Go1 recipe) and plugs in a
robot. Everything size-dependent (speeds, foot clearance, pushes, CoM
randomization) is scaled from the RobotDef so the same code serves a 1.4 m,
36 kg humanoid and a 0.4 m, 3 kg one.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains import config as terrains
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from chris_robots import mdp as chris_mdp
from chris_robots.robots.base import RobotDef

STEPS_PER_ITER = 24  # rsl-rl num_steps_per_env


def make_flat_env_cfg(
  robot: RobotDef,
  play: bool = False,
  amp: bool = False,
  amp_hybrid: bool = False,
  mimic: bool = False,
  rough: bool = False,
  agile: bool = False,
) -> ManagerBasedRlEnvCfg:
  """``amp=True`` swaps the hand-designed animation layer (gait clock, arm swing,
  torso sway, bob, pose prior, lean) for an adversarial motion prior trained on
  retargeted human mocap (see ``chris_robots.amp``). ``amp_hybrid=True`` keeps the
  animation layer and *adds* the motion prior as a lower-weight style nudge.
  ``mimic=True`` replaces the animation layer with tracking of a mocap gait library
  (see ``chris_robots.tools.gaitlib``): the policy observes the reference pose for
  the commanded speed and stride phase and is rewarded for matching it. ``rough=True``
  (with ``mimic``) walks it on gentle, blind-walkable uneven ground instead of a plane.
  ``agile=True`` (with ``mimic``) practises what the website asks for most: slow walks
  and turns in place, tracked tighter (``_agile``)."""
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
  # More "stop" commands: braking from a sprint is where the small robots fall.
  twist.rel_standing_envs = 0.2

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
  cfg.rewards["action_rate_l2"].weight = robot.action_rate_weight

  if mimic:
    if rough:
      _gentle_terrain(cfg, robot)
    cfg = _finish_mimic(cfg, robot, play)
    if agile:
      _agile(cfg, robot, play)
    return cfg
  if amp:
    return _finish_amp(cfg, robot, play)

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
  left_sign, separation = robot.foot_lateral()
  cfg.rewards["feet_apart"] = RewardTermCfg(
    func=chris_mdp.feet_apart,
    weight=-1.0,
    params={
      "site_names": robot.foot_sites,
      "left_index": left_index,
      "left_sign": left_sign,
      "separation_m": separation,
      "min_frac": 0.8,
    },
  )
  # ---- animation layer: torso sway / pelvic rotation / vertical bob from the clock ----
  cfg.rewards.pop("track_angular_velocity")
  cfg.rewards["track_yaw_sway"] = RewardTermCfg(
    func=chris_mdp.track_yaw_sway,
    weight=2.0,
    params={
      "command_name": "twist",
      "clock_name": "gait_clock",
      "left_sign": left_sign,
      "yaw_deg": robot.sway_yaw_deg,
      "std": math.sqrt(0.5),
    },
  )
  cfg.rewards["torso_sway"] = RewardTermCfg(
    func=chris_mdp.torso_sway,
    weight=1.0,
    params={
      "clock_name": "gait_clock",
      "left_sign": left_sign,
      "roll_deg": robot.sway_roll_deg,
      "std": 0.08,
    },
  )
  cfg.rewards["base_bob"] = RewardTermCfg(
    func=chris_mdp.base_bob,
    weight=0.5,
    params={
      "clock_name": "gait_clock",
      "amplitude_m": robot.bob_frac * robot.standing_height,
      "std": 0.5 * robot.bob_frac * robot.standing_height * 4.0 * math.pi / robot.gait_period,
    },
  )
  cfg.rewards["foot_flat"] = RewardTermCfg(
    func=chris_mdp.foot_flat,
    weight=-2.0,  # rad of stance-sole tilt beyond 3 deg: 10 deg tilt -> -0.25/step/foot
    params={
      "site_names": robot.foot_sites,
      "up_local": robot.foot_up_axes(),
      "sensor_name": "feet_ground_contact",
      "dead_zone_deg": 3.0,
    },
  )

  pairs = robot.arm_swing_pairs() if robot.arm_swing_gain > 0 else ()
  if pairs:
    cfg.rewards["arm_swing"] = RewardTermCfg(
      func=chris_mdp.arm_swing,
      weight=0.5,
      params={
        "pairs": pairs,
        "gain": robot.arm_swing_gain,
        "std": 0.2,
        "command_name": "twist",
        "command_threshold": 0.1 * v,
        # Elbows flex a little on the forward swing (pitch-type elbows only).
        "elbow_pairs": robot.elbow_swing_pairs(),
        "elbow_gain": 0.5,
      },
    )
  # Arms swing front/back, not sideways: penalise shoulder roll/yaw, wrist and
  # sideways-elbow deviation from the default pose.
  lateral = robot.lateral_arm_joints()
  if lateral:
    cfg.rewards["arm_lateral"] = RewardTermCfg(
      func=chris_mdp.joint_deviation_l2,
      weight=-2.0,
      params={"joint_names": lateral},
    )

  if amp_hybrid:
    cfg.rewards["amp_style"] = _amp_term(robot, play, weight=1.0)

  # ---- curriculum: start at a jog, open up to the full sprint range early ----------
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-v, 0.5 * run), "ang_vel_z": (-w, w)},
    {"step": 400 * STEPS_PER_ITER, "lin_vel_x": (-v, run), "ang_vel_z": (-w, w)},
  ]

  return _apply_play(cfg, robot, play)


def _apply_play(cfg: ManagerBasedRlEnvCfg, robot: RobotDef, play: bool) -> ManagerBasedRlEnvCfg:
  if play:
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist.ranges.lin_vel_x = (-robot.walk_vel, robot.max_lin_vel)
    twist.ranges.ang_vel_z = (-robot.max_ang_vel, robot.max_ang_vel)
  return cfg


def _amp_term(robot: RobotDef, play: bool, weight: float) -> RewardTermCfg:
  from chris_robots import amp as chris_amp
  from chris_robots.tools.mocap import DATA_DIR

  return RewardTermCfg(
    func=chris_amp.amp_style,
    weight=weight,
    params={
      "motion_file": str(DATA_DIR / f"{robot.name}.npz"),
      "foot_sites": robot.foot_sites,
      "standing_height": robot.standing_height,
      "train": not play,
    },
  )


_ROLE_WEIGHTS = {
  "hip_pitch": 1.0,
  "knee": 1.0,
  "hip_roll": 1.0,
  "ankle_pitch": 0.6,
  "ankle_roll": 0.4,
  "hip_yaw": 0.4,
  "shoulder_pitch": 1.0,
  "elbow": 0.6,
  "shoulder_roll": 0.5,
  "shoulder_yaw": 0.3,
}


def mimic_joint_weights(robot: RobotDef) -> tuple[float, ...]:
  """Per-joint weight of the pose-tracking reward, by anatomical role (model order)."""
  from chris_robots.tools.mocap import canonical_role

  out = []
  for name in robot.joint_names:
    key = canonical_role(name)
    out.append(_ROLE_WEIGHTS.get(key[1], 0.2) if key else 0.2)
  return tuple(out)


def _gentle_terrain(cfg: ManagerBasedRlEnvCfg, robot: RobotDef) -> None:
  """Uneven ground a blind policy can walk, every size scaled to the robot: tilted
  plates (up to 6 deg, +-2% of standing height), a grid of small steps (up to 2.4%),
  and flat patches. Covers the website's bumps (lib/sim/terrain.ts: about +-4% of
  standing height, over a much longer wavelength).

  Only boxes and convex meshes: MuJoCo Warp's heightfield contacts don't hold a box
  foot (a flat policy that never falls on a plane fell ~44 times a minute on a flat
  heightfield), so the mjlab heightfield terrains are out."""
  cfg.sim.nconmax = 48
  cfg.sim.njmax = 400
  h = robot.standing_height
  edge = 0.25 * h
  # Tiles scale with the robot so the geom count doesn't grow for small robots: the
  # raycast sensors' BVH spans every terrain geom in every world, and ~12k geoms x
  # 4096 worlds is already 14 GB (and the most Warp will build).
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(8.0 * min(1.0, h),) * 2,
      border_width=10.0 * h,
      num_rows=8,
      num_cols=8,
      sub_terrains={
        "flat": terrains.flat(proportion=0.3),
        "plates": terrains.tilted_grid(
          proportion=0.4,
          grid_width=0.6 * h,
          tilt_range_deg=6.0,
          height_range=0.04 * h,
          platform_width=h,
          border_width=edge,
          floor_depth=0.5,
        ),
        "steps": terrains.box_random_grid(
          proportion=0.3,
          grid_width=0.4 * h,
          grid_height_range=(0.0, 0.012 * h),
          platform_width=h,
          border_width=edge,
          merge_similar_heights=True,
          height_merge_threshold=0.003 * h,
        ),
      },
    ),
  )
  cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
    func=velocity_mdp.out_of_terrain_bounds, time_out=True
  )


def _finish_mimic(cfg: ManagerBasedRlEnvCfg, robot: RobotDef, play: bool) -> ManagerBasedRlEnvCfg:
  """Mocap-tracking variant: velocity tracking + gait-library pose/lean/sway tracking."""
  from chris_robots.tools.mocap import DATA_DIR

  v = robot.walk_vel
  run = robot.max_lin_vel
  w = robot.max_ang_vel
  for name in ("pose", "lean", "foot_clearance", "foot_swing_height", "air_time"):
    cfg.rewards.pop(name, None)
  cfg.commands["gait_clock"] = chris_mdp.MotionClockCommandCfg(
    gaitlib_file=str(DATA_DIR / f"{robot.name}_gaitlib.npz"),
    standing_threshold=0.1 * v,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["gait_clock"] = ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "gait_clock"}
    )
    cfg.observations[group].terms["motion_ref"] = ObservationTermCfg(
      func=chris_mdp.motion_ref, params={"command_name": "gait_clock"}
    )
  cfg.rewards["ref_joint_pos"] = RewardTermCfg(
    func=chris_mdp.ref_joint_pos,
    weight=4.0,
    params={"command_name": "gait_clock", "weights": mimic_joint_weights(robot), "std": 0.25},
  )
  cfg.rewards["ref_gravity"] = RewardTermCfg(
    func=chris_mdp.ref_gravity, weight=1.0, params={"command_name": "gait_clock", "std": 0.06}
  )
  cfg.rewards.pop("track_angular_velocity")
  cfg.rewards["ref_yaw_rate"] = RewardTermCfg(
    func=chris_mdp.ref_yaw_rate,
    weight=1.5,
    params={"command_name": "gait_clock", "twist_name": "twist", "std": math.sqrt(0.5)},
  )
  left_index = robot.foot_order.index("left")
  cfg.rewards["gait_schedule"] = RewardTermCfg(
    func=chris_mdp.gait_contact_schedule,
    weight=1.5,
    params={
      "sensor_name": "feet_ground_contact",
      "clock_name": "gait_clock",
      "left_index": left_index,
      "margin": 0.05,
    },
  )
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = robot.foot_sites
  cfg.rewards["foot_slip"].weight = -0.5
  for name in ("foot_slip", "soft_landing"):
    cfg.rewards[name].params["command_threshold"] = 0.1 * v
  left_sign, separation = robot.foot_lateral()
  cfg.rewards["feet_apart"] = RewardTermCfg(
    func=chris_mdp.feet_apart,
    weight=-1.0,
    params={
      "site_names": robot.foot_sites,
      "left_index": left_index,
      "left_sign": left_sign,
      "separation_m": separation,
      "min_frac": 0.6,
    },
  )
  cfg.rewards["foot_flat"] = RewardTermCfg(
    func=chris_mdp.foot_flat,
    weight=-1.0,
    params={
      "site_names": robot.foot_sites,
      "up_local": robot.foot_up_axes(),
      "sensor_name": "feet_ground_contact",
      "dead_zone_deg": 5.0,
    },
  )
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-v, 0.5 * run), "ang_vel_z": (-w, w)},
    {"step": 400 * STEPS_PER_ITER, "lin_vel_x": (-v, run), "ang_vel_z": (-w, w)},
  ]
  return _apply_play(cfg, robot, play)


def _agile(cfg: ManagerBasedRlEnvCfg, robot: RobotDef, play: bool) -> None:
  """Slow walking and turning, for a mimic policy fine-tuned from a full-range one.

  The full-range recipe tracks speed with a std of half a walk and turns with
  sqrt(0.5) rad/s, so creeping at 0.3 m/s or turning at half the asked rate costs
  little, and uniform commands up to a sprint rarely ask for either. Here 30 % of the
  moving envs amble (up to 0.9 walk speeds, turning a bit) and 15 % turn in place, a
  turn in place tracks a short stepping stride instead of a full forward one
  (``MotionClockCommandCfg.turn_stride``), slow commands keep a real, shorter stride
  (``slow_stride``), and the tracking stds grow with the command, so missing a slow
  walk or a gentle turn costs as much as missing a run."""
  v = robot.walk_vel
  w = robot.max_ang_vel
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  fields = {f: getattr(twist, f) for f in twist.__dataclass_fields__}
  cfg.commands["twist"] = chris_mdp.MixedVelocityCommandCfg(
    **fields,
    rel_slow_envs=0.0 if play else 0.3,
    slow_lin_vel_x=(0.12 * v, 0.9 * v),
    slow_ang_vel_z=(-0.6 * w, 0.6 * w),
    rel_turn_envs=0.0 if play else 0.15,
    turn_ang_vel_z=(0.3 * w, w),
    turn_lin_vel_x=(-0.05 * v, 0.1 * v),
  )
  clock = cfg.commands["gait_clock"]
  assert isinstance(clock, chris_mdp.MotionClockCommandCfg)
  clock.turn_stride = 0.45
  clock.slow_stride = True
  track = cfg.rewards["track_linear_velocity"]
  track.func = chris_mdp.track_lin_vel_rel
  track.params = {"command_name": track.params["command_name"], "std": 0.06 * v, "std_rel": 0.2}
  cfg.rewards["ref_yaw_rate"].params.update(std=0.15, std_rel=0.2)
  cfg.rewards["ref_yaw_rate"].weight = 2.5


def _finish_amp(cfg: ManagerBasedRlEnvCfg, robot: RobotDef, play: bool) -> ManagerBasedRlEnvCfg:
  """AMP variant: task rewards + regularisers + the adversarial style reward."""
  v = robot.walk_vel
  run = robot.max_lin_vel
  w = robot.max_ang_vel
  # Everything "how to move" comes from the data now.
  for name in ("pose", "lean", "foot_clearance", "foot_swing_height", "air_time"):
    cfg.rewards.pop(name, None)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = robot.foot_sites
  cfg.rewards["foot_slip"].weight = -0.5
  for name in ("foot_slip", "soft_landing"):
    cfg.rewards[name].params["command_threshold"] = 0.1 * v
  left_sign, separation = robot.foot_lateral()
  cfg.rewards["feet_apart"] = RewardTermCfg(
    func=chris_mdp.feet_apart,
    weight=-1.0,
    params={
      "site_names": robot.foot_sites,
      "left_index": robot.foot_order.index("left"),
      "left_sign": left_sign,
      "separation_m": separation,
      "min_frac": 0.6,
    },
  )
  cfg.rewards["foot_flat"] = RewardTermCfg(
    func=chris_mdp.foot_flat,
    weight=-2.0,
    params={
      "site_names": robot.foot_sites,
      "up_local": robot.foot_up_axes(),
      "sensor_name": "feet_ground_contact",
      "dead_zone_deg": 3.0,
    },
  )
  lateral = robot.lateral_arm_joints()
  if lateral:
    cfg.rewards["arm_lateral"] = RewardTermCfg(
      func=chris_mdp.joint_deviation_l2, weight=-1.0, params={"joint_names": lateral}
    )
  cfg.rewards["amp_style"] = _amp_term(robot, play, weight=robot.amp_weight)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-v, 0.5 * run), "ang_vel_z": (-w, w)},
    {"step": 400 * STEPS_PER_ITER, "lin_vel_x": (-v, run), "ang_vel_z": (-w, w)},
  ]
  return _apply_play(cfg, robot, play)
