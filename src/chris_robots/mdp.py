"""Extra reward terms on top of mjlab's velocity task."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

# ------------------------------------------------------------------ velocity commands


@dataclass(kw_only=True)
class MixedVelocityCommandCfg(UniformVelocityCommandCfg):
  """Uniform velocity commands, plus a share of slow walks and turns in place.

  Uniform sampling over a sprint-sized range leaves little practice where a browser
  robot spends its time: ambling up to the cursor and pivoting toward it. Of the
  resampled envs, ``rel_slow_envs`` walk forward at ``slow_lin_vel_x`` while turning
  within ``slow_ang_vel_z``, and ``rel_turn_envs`` turn in place at a yaw rate whose
  magnitude is in ``turn_ang_vel_z`` (with a little forward drift). Neither kind
  steers by heading."""

  rel_slow_envs: float = 0.0
  slow_lin_vel_x: tuple[float, float] = (0.1, 0.6)
  slow_ang_vel_z: tuple[float, float] = (-0.8, 0.8)
  rel_turn_envs: float = 0.0
  turn_ang_vel_z: tuple[float, float] = (0.4, 1.5)
  turn_lin_vel_x: tuple[float, float] = (-0.05, 0.1)

  def build(self, env: ManagerBasedRlEnv) -> MixedVelocityCommand:
    return MixedVelocityCommand(self, env)


class MixedVelocityCommand(UniformVelocityCommand):
  cfg: MixedVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    c = self.cfg
    # Shares of the envs that aren't standing.
    roll = torch.rand(len(env_ids), device=self.device)
    roll[self.is_standing_env[env_ids]] = 2.0
    slow = env_ids[roll < c.rel_slow_envs]
    turn = env_ids[(roll >= c.rel_slow_envs) & (roll < c.rel_slow_envs + c.rel_turn_envs)]
    u = lambda n, lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi)  # noqa: E731
    if len(slow):
      self.vel_command_b[slow, 0] = u(len(slow), *c.slow_lin_vel_x)
      self.vel_command_b[slow, 1] = 0.0
      self.vel_command_b[slow, 2] = u(len(slow), *c.slow_ang_vel_z)
    if len(turn):
      sign = torch.where(torch.rand(len(turn), device=self.device) < 0.5, -1.0, 1.0)
      self.vel_command_b[turn, 0] = u(len(turn), *c.turn_lin_vel_x)
      self.vel_command_b[turn, 1] = 0.0
      self.vel_command_b[turn, 2] = sign * u(len(turn), *c.turn_ang_vel_z)
    both = torch.cat([slow, turn])
    if len(both):
      self.is_heading_env[both] = False
      self.is_forward_env[both] = False
      self.is_world_env[both] = False
      self.vel_command_w[both] = self.vel_command_b[both]

# ------------------------------------------------------------------ gait clock (phase)


@dataclass(kw_only=True)
class GaitClockCommandCfg(CommandTermCfg):
  """A phase variable that drives a periodic contact schedule (Siekmann et al. 2021).

  phi in [0, 1) advances by dt / period each step while a velocity is commanded and
  freezes when standing. Left foot phase = phi, right foot phase = phi + 0.5. A foot
  is scheduled to be in stance while its phase < duty. Period and duty interpolate
  from the walk values to the run values as the commanded speed goes from walk_vel
  to run_vel. The command exposed to the policy is [sin(2 pi phi), cos(2 pi phi)].
  """

  twist_name: str = "twist"
  walk_vel: float = 0.8
  run_vel: float = 2.5
  period_walk: float = 0.9
  period_run: float = 0.6
  duty_walk: float = 0.6
  duty_run: float = 0.4
  standing_threshold: float = 0.08
  period_jitter: float = 0.1
  """Per-episode random scale of the period in [1 - jitter, 1 + jitter] (gait variety)."""
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env: ManagerBasedRlEnv) -> GaitClockCommand:
    return GaitClockCommand(self, env)


class GaitClockCommand(CommandTerm):
  cfg: GaitClockCommandCfg

  def __init__(self, cfg: GaitClockCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.period = torch.full((self.num_envs,), cfg.period_walk, device=self.device)
    self.duty = torch.full((self.num_envs,), cfg.duty_walk, device=self.device)
    self.moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.period_scale = torch.ones(self.num_envs, device=self.device)
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self._env_ref = env

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)
    j = self.cfg.period_jitter
    self.period_scale[env_ids] = 1.0 + j * (
      2.0 * torch.rand(len(env_ids), device=self.device) - 1.0
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    twist = self._env_ref.command_manager.get_command(self.cfg.twist_name)
    assert twist is not None
    speed = torch.norm(twist[:, :2], dim=1) + 0.3 * torch.abs(twist[:, 2])
    s = torch.clamp(
      (speed - self.cfg.walk_vel) / max(1e-6, self.cfg.run_vel - self.cfg.walk_vel), 0.0, 1.0
    )
    self.period = (
      self.cfg.period_walk + s * (self.cfg.period_run - self.cfg.period_walk)
    ) * self.period_scale
    self.duty = self.cfg.duty_walk + s * (self.cfg.duty_run - self.cfg.duty_walk)
    self.moving = speed > self.cfg.standing_threshold
    if env_ids is None:  # per-step path: advance the clock
      dt = self._env_ref.step_dt
      self.phase = torch.where(
        self.moving, torch.remainder(self.phase + dt / self.period, 1.0), self.phase
      )
    two_pi_phi = 2.0 * math.pi * self.phase
    self._command[:, 0] = torch.sin(two_pi_phi)
    self._command[:, 1] = torch.cos(two_pi_phi)

  def foot_phase(self, left_index: int) -> torch.Tensor:
    """Per-foot phase [B, 2] in sensor order; the left foot gets offset 0."""
    offsets = torch.tensor([0.0 if i == left_index else 0.5 for i in range(2)], device=self.device)
    return torch.remainder(self.phase.unsqueeze(1) + offsets, 1.0)


@dataclass(kw_only=True)
class MotionClockCommandCfg(GaitClockCommandCfg):
  """Gait clock whose period, duty and a full-body reference pose come from a mocap
  gait library (``uv run gaitlib``), looked up by commanded speed and phase.

  Exposes ``ref_q`` [B, J] (entity joint order), ``ref_grav`` [B, 2] (projected
  gravity x/y) and ``ref_yaw_rate`` [B] (stride-locked pelvis yaw rate). Backward
  commands play the stride in reverse. Below the slowest table row the reference
  blends to the standing pose. Mirrors ``gaitlib.sample`` exactly.
  """

  gaitlib_file: str
  period_jitter: float = 0.08
  turn_stride: float = 1.0
  """Stride amplitude (fraction of the slowest row) when turning in place; it grows to
  the full stride as the linear speed reaches the slowest row. 1.0 = the original
  lookup, where a turn in place got a full forward stride to track."""
  slow_stride: bool = False
  """Below the slowest row keep a real, shorter stride instead of blending linearly to
  standing: amplitude sqrt(speed / slowest), the way human step length shrinks, faded
  in between 15 % and 30 % of the slowest row's speed. Without it a slow walk or a
  gentle turn tracks an all-but-standing pose, and the policy learns to stand."""

  def build(self, env: ManagerBasedRlEnv) -> MotionClockCommand:
    return MotionClockCommand(self, env)


def load_gaitlib(path: str, joint_names: list[str] | tuple[str, ...], device) -> dict:
  """Gait-library tensors with joints reordered to ``joint_names``."""
  import numpy as np

  lib = np.load(path)
  names = [str(n) for n in lib["joint_names"]]
  order = [names.index(n) for n in joint_names]
  f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
  yaw = lib["yaw"]  # [S, P]
  n_phase = yaw.shape[1]
  return {
    "speeds": f(lib["speeds"]),
    "period": f(lib["period"]),
    "duty": f(lib["duty"]),
    "q": f(lib["q"][..., order]),  # [S, P, J]
    "grav": f(lib["grav"]),  # [S, P, 2]
    # d(yaw)/d(phase), central difference on the periodic table
    "dyaw": f((np.roll(yaw, -1, axis=1) - np.roll(yaw, 1, axis=1)) * n_phase / 2.0)[..., None],
    "stand_q": f(lib["stand_q"][order]),
  }


def motion_reference(lib: dict, twist: torch.Tensor, phase: torch.Tensor) -> dict:
  """Gait-library lookup for commands ``twist`` [B, 3] at clock phase [B].

  Speed s = |v_xy| + 0.3 |w_z| picks (and linearly blends) two table rows; phase
  picks (and blends) two samples. Backward commands play the stride in reverse via
  psi = (duty - phase) mod 1, which keeps each foot's stance window where the clock
  schedule puts it. Below the slowest row the pose blends to standing (weight w).
  The browser (lib/sim/gait.ts) implements exactly this.
  """
  sp = lib["speeds"]
  speed = torch.norm(twist[:, :2], dim=1) + 0.3 * torch.abs(twist[:, 2])
  s = torch.clamp(speed, float(sp[0]), float(sp[-1]))
  i = torch.clamp(torch.searchsorted(sp, s) - 1, 0, len(sp) - 2)
  a = (s - sp[i]) / (sp[i + 1] - sp[i])
  period = torch.lerp(lib["period"][i], lib["period"][i + 1], a)
  duty = torch.lerp(lib["duty"][i], lib["duty"][i + 1], a)
  backward = twist[:, 0] < -0.5 * torch.norm(twist[:, :2], dim=1)
  psi = torch.where(backward, torch.remainder(duty - phase, 1.0), phase)
  n_phase = lib["q"].shape[1]
  x = psi * n_phase
  k = torch.floor(x).long() % n_phase
  k1 = (k + 1) % n_phase
  b = (x - torch.floor(x)).unsqueeze(1)
  a1 = a.unsqueeze(1)

  def at(t: torch.Tensor) -> torch.Tensor:  # t [S, P, D] -> [B, D]
    r0 = (1 - b) * t[i, k] + b * t[i, k1]
    r1 = (1 - b) * t[i + 1, k] + b * t[i + 1, k1]
    return (1 - a1) * r0 + a1 * r1

  if lib.get("slow_stride", 0.0):
    on = torch.clamp((speed - 0.15 * sp[0]) / (0.15 * sp[0]), 0.0, 1.0)
    w = on * on * (3.0 - 2.0 * on) * torch.sqrt(torch.clamp(speed / sp[0], 0.0, 1.0))
  else:
    u = torch.clamp((speed - 0.25 * sp[0]) / (0.75 * sp[0]), 0.0, 1.0)
    w = u * u * (3.0 - 2.0 * u)
  m0 = float(lib.get("turn_stride", 1.0))
  if m0 < 1.0:
    # Stepping in place to turn: a shorter stride until the robot is actually walking.
    ul = torch.clamp(torch.norm(twist[:, :2], dim=1) / sp[0], 0.0, 1.0)
    w = w * (m0 + (1.0 - m0) * ul * ul * (3.0 - 2.0 * ul))
  stand = lib["stand_q"]
  direction = torch.where(backward, -1.0, 1.0)
  return {
    "speed": speed,
    "period": period,
    "duty": duty,
    "weight": w,
    "q": stand + w.unsqueeze(1) * (at(lib["q"]) - stand),
    "grav": w.unsqueeze(1) * at(lib["grav"]),
    "dyaw_dphase": w * direction * at(lib["dyaw"])[:, 0],
  }


class MotionClockCommand(GaitClockCommand):
  cfg: MotionClockCommandCfg

  def __init__(self, cfg: MotionClockCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    robot: Entity = env.scene["robot"]
    self.lib = load_gaitlib(cfg.gaitlib_file, robot.joint_names, self.device)
    self.lib["turn_stride"] = cfg.turn_stride
    self.lib["slow_stride"] = float(cfg.slow_stride)
    J = len(robot.joint_names)
    self.ref_q = self.lib["stand_q"].expand(self.num_envs, J).clone()
    self.ref_grav = torch.zeros(self.num_envs, 2, device=self.device)
    self.ref_yaw_rate = torch.zeros(self.num_envs, device=self.device)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    twist = self._env_ref.command_manager.get_command(self.cfg.twist_name)
    assert twist is not None
    ref = motion_reference(self.lib, twist, self.phase)
    self.period = ref["period"] * self.period_scale
    self.duty = ref["duty"]
    self.moving = ref["speed"] > self.cfg.standing_threshold
    if env_ids is None:
      dt = self._env_ref.step_dt
      self.phase = torch.where(
        self.moving, torch.remainder(self.phase + dt / self.period, 1.0), self.phase
      )
      ref = motion_reference(self.lib, twist, self.phase)
    two_pi_phi = 2.0 * math.pi * self.phase
    self._command[:, 0] = torch.sin(two_pi_phi)
    self._command[:, 1] = torch.cos(two_pi_phi)
    self.ref_q = ref["q"]
    self.ref_grav = ref["grav"]
    self.ref_yaw_rate = ref["dyaw_dphase"] / self.period


def _motion(env: ManagerBasedRlEnv, name: str) -> MotionClockCommand:
  term = env.command_manager.get_term(name)
  assert isinstance(term, MotionClockCommand)
  return term


def motion_ref(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Observation: the reference pose relative to the default pose [B, J]."""
  robot: Entity = env.scene["robot"]
  return _motion(env, command_name).ref_q - robot.data.default_joint_pos


class ref_joint_pos:
  """exp(-sum_j w_j (q_j - ref_j)^2 / sum_j w_j / std^2): track the mocap pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.w = torch.tensor(cfg.params["weights"], device=env.device, dtype=torch.float32)

  def __call__(
    self, env: ManagerBasedRlEnv, command_name: str, weights: tuple[float, ...], std: float
  ) -> torch.Tensor:
    del weights
    robot: Entity = env.scene["robot"]
    err = torch.square(robot.data.joint_pos - _motion(env, command_name).ref_q)
    return torch.exp(-(err * self.w).sum(dim=1) / self.w.sum() / std**2)


def ref_gravity(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  """Track the mocap torso lean and side sway through projected gravity x/y."""
  robot: Entity = env.scene["robot"]
  g = robot.data.projected_gravity_b[:, :2]
  err = torch.sum(torch.square(g - _motion(env, command_name).ref_grav), dim=1)
  return torch.exp(-err / std**2)


def ref_yaw_rate(
  env: ManagerBasedRlEnv, command_name: str, twist_name: str, std: float, std_rel: float = 0.0
) -> torch.Tensor:
  """Yaw-rate tracking with the mocap's stride-locked pelvis yaw on top. The std is
  ``std + std_rel * |w_z|``: with ``std_rel`` a small std makes slow turns count."""
  robot: Entity = env.scene["robot"]
  twist = env.command_manager.get_command(twist_name)
  assert twist is not None
  target = twist[:, 2] + _motion(env, command_name).ref_yaw_rate
  s = std + std_rel * torch.abs(twist[:, 2])
  return torch.exp(-torch.square(robot.data.root_link_ang_vel_b[:, 2] - target) / s**2)


def track_lin_vel_rel(
  env: ManagerBasedRlEnv, command_name: str, std: float, std_rel: float
) -> torch.Tensor:
  """``track_linear_velocity`` with a std of ``std + std_rel * |v_cmd|``, so creeping
  at half the asked speed costs as much at 0.15 m/s as at a run."""
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = robot.data.root_link_lin_vel_b
  err = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1) + torch.square(actual[:, 2])
  s = std + std_rel * torch.norm(command[:, :2], dim=1)
  return torch.exp(-err / s**2)


def _clock(env: ManagerBasedRlEnv, name: str) -> GaitClockCommand:
  term = env.command_manager.get_term(name)
  assert isinstance(term, GaitClockCommand)
  return term


def gait_contact_schedule(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  clock_name: str,
  left_index: int,
  margin: float = 0.03,
) -> torch.Tensor:
  """+1 per foot whose contact state matches the clock's stance/swing schedule.

  Within ``margin`` (in phase units) of a transition either state is accepted.
  While standing (clock not moving) the schedule is "both feet down".
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  in_contact = sensor.data.current_contact_time > 0.0  # [B, 2]
  clock = _clock(env, clock_name)
  phi = clock.foot_phase(left_index)
  duty = clock.duty.unsqueeze(1)
  sched_stance = phi < duty
  near_edge = (torch.abs(phi - duty) < margin) | (phi < margin) | (phi > 1.0 - margin)
  ok = (in_contact == sched_stance) | near_edge
  moving = clock.moving.unsqueeze(1)
  ok = torch.where(moving, ok, in_contact)
  return ok.float().mean(dim=1)


def gait_swing_height(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  clock_name: str,
  left_index: int,
  target_height: float,
) -> torch.Tensor:
  """Bell-shaped swing-foot height target over the scheduled swing phase."""
  height_sensor: TerrainHeightSensor = env.scene[sensor_name]
  h = height_sensor.data.heights  # [B, 2]
  clock = _clock(env, clock_name)
  phi = clock.foot_phase(left_index)
  duty = clock.duty.unsqueeze(1)
  u = torch.clamp((phi - duty) / torch.clamp(1.0 - duty, min=1e-3), 0.0, 1.0)
  target = target_height * torch.sin(math.pi * u)
  in_swing = (phi >= duty) & clock.moving.unsqueeze(1)
  err = torch.square(h - target) / (0.5 * target_height) ** 2
  r = torch.exp(-err) * in_swing.float()
  return r.sum(dim=1) / torch.clamp(in_swing.float().sum(dim=1), min=1.0)


class foot_flat:
  """Penalise stance-foot sole tilt (radians beyond a dead-zone), summed over feet.

  Tilt = angle between the sole up-axis and world z. Only feet in ground contact are
  penalised (the swing foot may pitch for a natural heel-strike). The up axis is read
  off the foot sites' orientation at the init pose, so it works for any foot frame.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    ids, _ = asset.find_sites(list(cfg.params["site_names"]), preserve_order=True)
    self.site_ids = torch.tensor(ids, device=env.device)
    # World-up expressed in each site's frame at the init pose (computed offline by
    # the env config from the MuJoCo model, since entity data may not be valid yet).
    self.up_local = torch.tensor(cfg.params["up_local"], device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    site_names: tuple[str, ...],
    up_local: tuple,
    sensor_name: str | None = None,
    dead_zone_deg: float = 3.0,
  ) -> torch.Tensor:
    del site_names, up_local
    asset: Entity = env.scene["robot"]
    q = asset.data.site_quat_w[:, self.site_ids]  # [B, 2, 4]
    up_w = quat_apply(q.reshape(-1, 4), self.up_local.repeat(q.shape[0], 1)).reshape(
      q.shape[0], -1, 3
    )
    tilt = torch.acos(up_w[..., 2].clamp(-1.0, 1.0))  # [B, 2] rad
    excess = (tilt - math.radians(dead_zone_deg)).clamp(min=0.0)
    if sensor_name is not None:
      sensor: ContactSensor = env.scene[sensor_name]
      assert sensor.data.current_contact_time is not None
      in_contact = sensor.data.current_contact_time > 0.0  # [B, 2]; sensor order == model order
      excess = excess * in_contact.float()
    return excess.sum(dim=1)


class arm_swing:
  """Reward contralateral arm swing: each shoulder pitch follows the opposite hip.

  ``pairs`` is a tuple of ``(shoulder_joint, hip_joint, shoulder_sign, hip_sign)``
  where the signs are ``sign(world_axis . y)`` of each pitch joint at the init pose,
  so ``-sign * (q - q_default)`` is "limb forward" for both. The target is
  ``forward_arm = gain * forward_opposite_leg``; reward = exp(-mean(err^2) / std^2),
  only while a velocity is commanded. Purely cosmetic: it is what makes a walk read
  as a walk instead of a robot sliding its legs under a frozen torso.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    default = asset.data.default_joint_pos
    assert default is not None
    self.default = default
    pairs = cfg.params["pairs"]
    sh_ids, hip_ids, sh_sign, hip_sign = [], [], [], []
    for sh, hip, s_sh, s_hip in pairs:
      sh_ids.append(asset.find_joints(sh)[0][0])
      hip_ids.append(asset.find_joints(hip)[0][0])
      sh_sign.append(float(s_sh))
      hip_sign.append(float(s_hip))
    self.sh_ids = torch.tensor(sh_ids, device=env.device)
    self.hip_ids = torch.tensor(hip_ids, device=env.device)
    self.sh_sign = torch.tensor(sh_sign, device=env.device)
    self.hip_sign = torch.tensor(hip_sign, device=env.device)
    # Optional elbow coupling: (elbow, same-side shoulder_pitch, elbow sign, shoulder sign).
    el_ids, el_sh_ids, el_sign, el_sh_sign = [], [], [], []
    for el, sh, s_el, s_sh in cfg.params.get("elbow_pairs", ()):
      el_ids.append(asset.find_joints(el)[0][0])
      el_sh_ids.append(asset.find_joints(sh)[0][0])
      el_sign.append(float(s_el))
      el_sh_sign.append(float(s_sh))
    self.el_ids = torch.tensor(el_ids, device=env.device, dtype=torch.long)
    self.el_sh_ids = torch.tensor(el_sh_ids, device=env.device, dtype=torch.long)
    self.el_sign = torch.tensor(el_sign, device=env.device)
    self.el_sh_sign = torch.tensor(el_sh_sign, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    pairs: tuple,
    gain: float,
    std: float,
    command_name: str,
    command_threshold: float,
    elbow_pairs: tuple = (),
    elbow_gain: float = 0.5,
  ) -> torch.Tensor:
    del pairs, elbow_pairs  # Resolved in __init__.
    asset: Entity = env.scene["robot"]
    rel = asset.data.joint_pos - self.default
    fwd_arm = -self.sh_sign * rel[:, self.sh_ids]
    fwd_leg = -self.hip_sign * rel[:, self.hip_ids]
    err = fwd_arm - gain * fwd_leg
    if len(self.el_ids):
      fwd_forearm = -self.el_sign * rel[:, self.el_ids]
      fwd_same_arm = -self.el_sh_sign * rel[:, self.el_sh_ids]
      err = torch.cat([err, fwd_forearm - elbow_gain * fwd_same_arm], dim=1)
    # Always on: when standing the legs are at default, so the target is "arms at
    # default" too, which stops the arms freezing mid-swing after a stop.
    del command_name, command_threshold
    return torch.exp(-torch.mean(err**2, dim=1) / std**2)


class joint_deviation_l2:
  """Mean squared deviation from the default pose over a set of joints (penalty).

  Used to keep the lateral arm axes (shoulder roll/yaw, wrists, sideways elbows) quiet
  so that arm motion reads as front/back swing rather than flapping.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    ids, _ = asset.find_joints(list(cfg.params["joint_names"]), preserve_order=True)
    self.ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    default = asset.data.default_joint_pos
    assert default is not None
    self.default = default

  def __call__(self, env: ManagerBasedRlEnv, joint_names: tuple[str, ...]) -> torch.Tensor:
    del joint_names
    asset: Entity = env.scene["robot"]
    rel = asset.data.joint_pos[:, self.ids] - self.default[:, self.ids]
    return torch.mean(rel**2, dim=1)


def _moving(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return ((torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > threshold).float()


def lean(
  env: ManagerBasedRlEnv,
  command_name: str,
  run_vel: float,
  lean_at_run: float,
  std: float,
) -> torch.Tensor:
  """Replace 'upright' with a speed-dependent forward lean.

  Target pitch = lean_at_run * clamp(cmd_vx, 0) / run_vel, so the torso stays level
  when standing and leans into the run at top speed. Roll is pulled towards zero
  with the same tolerance. In the base frame, gravity's x component is +sin(pitch)
  for a nose-down pitch.
  """
  asset: Entity = env.scene["robot"]
  g = asset.data.projected_gravity_b
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = torch.sin(lean_at_run * torch.clamp(command[:, 0], min=0.0) / run_vel)
  err = torch.square(g[:, 0] - target)  # roll is shaped by torso_sway, not pulled to zero
  return torch.exp(-err / std**2)


# ------------------------------------------------------------------ animation layer
# Sway targets driven by the gait clock. Amplitudes follow published walking kinematics
# (pelvic transverse rotation ~8 deg total growing with speed, pelvic list ~10-15 deg,
# COM vertical excursion ~5 cm p-p = 3% of height, step width 8-10 cm), exaggerated a
# little because these robots have no waist: the whole upper body is the pelvis.


class torso_sway:
  """Roll toward the stance leg, following the clock: g_y target = sin(roll) * cos(2 pi (phi - duty/2)).

  Peaks over the left foot at left mid-stance and over the right foot half a cycle
  later. Zero when standing (clock frozen, target still well defined but we gate it).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.left_sign = float(cfg.params["left_sign"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    clock_name: str,
    left_sign: float,
    roll_deg: float,
    std: float,
  ) -> torch.Tensor:
    del left_sign
    asset: Entity = env.scene["robot"]
    clock = _clock(env, clock_name)
    phase = clock.phase - 0.5 * clock.duty
    target = self.left_sign * math.sin(math.radians(roll_deg)) * torch.cos(2.0 * math.pi * phase)
    target = torch.where(clock.moving, target, torch.zeros_like(target))
    g_y = asset.data.projected_gravity_b[:, 1]
    return torch.exp(-torch.square(g_y - target) / std**2)


class track_yaw_sway:
  """Yaw-rate tracking with a stride-synchronous pelvic rotation on top.

  Pelvis yaw = -Psi cos(2 pi phi) (swing-side hip forward, i.e. right hip forward at
  right touchdown), so the extra yaw rate is Psi * 2 pi / T * sin(2 pi phi). Replaces
  mjlab's track_angular_velocity, which would otherwise penalise any sway.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.left_sign = float(cfg.params["left_sign"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    clock_name: str,
    left_sign: float,
    yaw_deg: float,
    std: float,
  ) -> torch.Tensor:
    del left_sign
    asset: Entity = env.scene["robot"]
    clock = _clock(env, clock_name)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    amp = math.radians(yaw_deg) * 2.0 * math.pi / clock.period
    sway = self.left_sign * amp * torch.sin(2.0 * math.pi * clock.phase)
    sway = torch.where(clock.moving, sway, torch.zeros_like(sway))
    target = command[:, 2] + sway
    actual = asset.data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(actual - target) / std**2)


def base_bob(
  env: ManagerBasedRlEnv,
  clock_name: str,
  amplitude_m: float,
  std: float,
) -> torch.Tensor:
  """Vertical COM bob: highest at mid single-support, two cycles per stride.

  Tracked through the base vertical velocity so no absolute height is prescribed:
  h = A cos(4 pi (phi - duty/2))  =>  dh/dt = -A 4 pi / T sin(4 pi (phi - duty/2)).
  """
  asset: Entity = env.scene["robot"]
  clock = _clock(env, clock_name)
  phase = clock.phase - 0.5 * clock.duty
  target = -amplitude_m * 4.0 * math.pi / clock.period * torch.sin(4.0 * math.pi * phase)
  target = torch.where(clock.moving, target, torch.zeros_like(target))
  vz = asset.data.root_link_lin_vel_w[:, 2]
  return torch.exp(-torch.square(vz - target) / std**2)


class feet_apart:
  """Penalise a catwalk stance: lateral foot separation (in the base frame) below
  ``min_frac`` of the init-pose separation, normalised so 1 = feet on one line."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    ids, _ = asset.find_sites(list(cfg.params["site_names"]), preserve_order=True)
    self.site_ids = torch.tensor(ids, device=env.device)
    self.left_sign = float(cfg.params["left_sign"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    site_names: tuple[str, str],
    left_index: int,
    left_sign: float,
    separation_m: float,
    min_frac: float = 0.8,
  ) -> torch.Tensor:
    del site_names, left_sign
    asset: Entity = env.scene["robot"]
    p = asset.data.site_pos_w[:, self.site_ids]  # [B, 2, 3]
    d = quat_apply_inverse(asset.data.root_link_quat_w, p[:, left_index] - p[:, 1 - left_index])
    lateral = self.left_sign * d[:, 1]  # + = left foot on the left, - = crossed
    return torch.clamp(min_frac * separation_m - lateral, min=0.0) / separation_m


class stride:
  """Reward fore-aft separation of the feet (normalised by ``cap``) while moving.

  Dense version of a step-length reward: averaged over a gait cycle it grows with
  stride length, so the policy prefers long strides to a quick shuffle at the same
  speed. Separation is measured along the base heading.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene["robot"]
    ids, _ = asset.find_sites(list(cfg.params["site_names"]), preserve_order=True)
    assert len(ids) == 2, ids
    self.site_ids = torch.tensor(ids, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    site_names: tuple[str, ...],
    cap: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    del site_names
    asset: Entity = env.scene["robot"]
    pos = asset.data.site_pos_w[:, self.site_ids, :2]  # [B, 2, 2]
    delta = pos[:, 0] - pos[:, 1]
    heading = asset.data.heading_w
    fwd = torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)
    sep = torch.abs(torch.sum(delta * fwd, dim=1))
    return torch.clamp(sep / cap, max=1.0) * _moving(env, command_name, command_threshold)


def feet_air_time_biped(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Reward long single-stance phases (Isaac Lab's ``feet_air_time_positive_biped``).

  While exactly one foot is on the ground, the reward is the shorter of (swing foot's
  air time, stance foot's contact time), clamped at ``threshold``. Double support and
  flight earn nothing, so a fast double-support shuffle is worth less than a proper
  step-swing-step walk. Only active when a non-trivial velocity is commanded.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.current_air_time is not None and data.current_contact_time is not None
  air_time = data.current_air_time
  contact_time = data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = in_contact.int().sum(dim=1) == 1
  reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  reward = torch.clamp(reward, max=threshold)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold
  return reward * moving.float()
