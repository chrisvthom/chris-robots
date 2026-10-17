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
