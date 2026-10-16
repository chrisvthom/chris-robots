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
    if robot.task_id in existing:
      continue
    register_mjlab_task(
      task_id=robot.task_id,
      env_cfg=make_flat_env_cfg(robot),
      play_env_cfg=make_flat_env_cfg(robot, play=True),
      rl_cfg=make_ppo_runner_cfg(robot),
      runner_cls=VelocityOnPolicyRunner,
    )
    registered.append(robot.task_id)
  return registered


# Entry-point import side effect (what mjlab expects).
register_tasks()
