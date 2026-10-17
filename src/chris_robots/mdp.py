"""Extra reward terms on top of mjlab's velocity task."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensor


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

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    pairs: tuple,
    gain: float,
    std: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    del pairs  # Resolved in __init__.
    asset: Entity = env.scene["robot"]
    rel = asset.data.joint_pos - self.default
    fwd_arm = -self.sh_sign * rel[:, self.sh_ids]
    fwd_leg = -self.hip_sign * rel[:, self.hip_ids]
    err = fwd_arm - gain * fwd_leg
    reward = torch.exp(-torch.mean(err**2, dim=1) / std**2)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold
    return reward * moving.float()


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
