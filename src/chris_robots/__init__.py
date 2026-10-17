"""Chris Thomas' robots: MuJoCo models + mjlab walking tasks.

Importing this package registers every task with mjlab (via the ``mjlab.tasks``
entry point declared in ``pyproject.toml``), so ``uv run train <task>`` and
``uv run play <task>`` just work. Task ids: ``Chris-Velocity-Flat-<Robot>``.
"""

import os
from pathlib import Path

# Repo layout: <repo>/robots/<name>/<name>.xml. Resolve relative to this file so it
# works from an editable install; override with CHRIS_ROBOTS_DIR if you vendor the
# assets elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTS_DIR = Path(os.environ.get("CHRIS_ROBOTS_DIR", _REPO_ROOT / "robots"))
POLICIES_DIR = _REPO_ROOT / "policies"

__all__ = ["ROBOTS_DIR", "POLICIES_DIR", "register_tasks"]


def register_tasks() -> list[str]:
  """Register one flat-ground velocity task per robot. Idempotent."""
  from mjlab.tasks.registry import list_tasks, register_mjlab_task
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

  from chris_robots.env_cfgs import make_flat_env_cfg
  from chris_robots.rl_cfg import make_ppo_runner_cfg
  from chris_robots.robots import ALL_ROBOTS

  registered = []
  existing = set(list_tasks())
  for robot in ALL_ROBOTS:
    # "<task>" is plain PPO; "<task>-Sym" adds left/right mirror augmentation + loss.
    # "<task>-Amp" replaces the hand-made animation layer with an adversarial motion
    # prior from retargeted mocap (needs ``uv run mocap <robot>`` first).
    # "<task>-SymAmp" keeps the animation layer and adds the prior as a style nudge.
    # "<task>-Mimic" tracks a mocap gait library (needs ``uv run gaitlib <robot>``).
    # "<task>-MimicRough" is the same on gentle uneven ground.
    # "<task>-MimicAgile" fine-tunes a mimic policy for slow walks and turns in place.
    variants = (
      (robot.task_id, False, {}),
      (f"{robot.task_id}-Sym", True, {}),
      (f"{robot.task_id}-Amp", True, {"amp": True}),
      (f"{robot.task_id}-SymAmp", True, {"amp_hybrid": True}),
      (f"{robot.task_id}-Mimic", True, {"mimic": True}),
      (f"{robot.task_id}-MimicRough", True, {"mimic": True, "rough": True}),
      (f"{robot.task_id}-MimicAgile", True, {"mimic": True, "agile": True}),
    )
    for task_id, symmetry, kw in variants:
      if task_id in existing:
        continue
      register_mjlab_task(
        task_id=task_id,
        env_cfg=make_flat_env_cfg(robot, **kw),
        play_env_cfg=make_flat_env_cfg(robot, play=True, **kw),
        rl_cfg=make_ppo_runner_cfg(robot, symmetry=symmetry),
        runner_cls=VelocityOnPolicyRunner,
      )
      registered.append(task_id)
  return registered


# Entry-point import side effect (what mjlab expects).
register_tasks()
