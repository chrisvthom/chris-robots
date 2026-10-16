"""Extra reward terms on top of mjlab's velocity task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor


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
