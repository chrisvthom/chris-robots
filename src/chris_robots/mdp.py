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
from mjlab.utils.lab_api.math import quat_apply

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
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self._env_ref = env

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    twist = self._env_ref.command_manager.get_command(self.cfg.twist_name)
    assert twist is not None
    speed = torch.norm(twist[:, :2], dim=1) + 0.3 * torch.abs(twist[:, 2])
    s = torch.clamp(
      (speed - self.cfg.walk_vel) / max(1e-6, self.cfg.run_vel - self.cfg.walk_vel), 0.0, 1.0
    )
    self.period = self.cfg.period_walk + s * (self.cfg.period_run - self.cfg.period_walk)
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
  """Penalise sole tilt: 1 - (sole up-axis . world z), averaged over both feet.

  The up axis is read off the foot sites' orientation at the init pose, so it works
  for any foot frame convention.
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
    angle_scale: float = 1.0,
  ) -> torch.Tensor:
    del site_names, up_local
    asset: Entity = env.scene["robot"]
    q = asset.data.site_quat_w[:, self.site_ids]  # [B, 2, 4]
    up_w = quat_apply(q.reshape(-1, 4), self.up_local.repeat(q.shape[0], 1)).reshape(
      q.shape[0], -1, 3
    )
    return (1.0 - up_w[..., 2]).mean(dim=1) * angle_scale


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
    reward = torch.exp(-torch.mean(err**2, dim=1) / std**2)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold
    return reward * moving.float()


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
  err = torch.square(g[:, 0] - target) + torch.square(g[:, 1])
  return torch.exp(-err / std**2)


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
