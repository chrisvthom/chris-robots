"""Build each task's environment on the GPU and step it with dummy actions.

Catches config errors (bad regexes, missing sites/geoms), NaNs, and gives a quick
feel for the reward scale before burning GPU hours.

  uv run smoke-test                  # all robots
  uv run smoke-test kbot --steps 200 --agent random
"""

from __future__ import annotations

import argparse
import time

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import chris_robots  # noqa: F401  (registers tasks)
from chris_robots.robots import ALL_ROBOTS, ROBOTS_BY_NAME


def run(robot_name: str, num_envs: int, steps: int, agent: str, device: str) -> None:
  robot = ROBOTS_BY_NAME[robot_name]
  print(f"\n=== {robot.task_id} ===")
  print(robot.describe())
  cfg = load_env_cfg(robot.task_id)
  cfg.scene.num_envs = num_envs
  t0 = time.time()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  print(
    f"env built in {time.time() - t0:.1f}s; obs actor dim="
    f"{env.observation_manager.group_obs_dim['actor']} actions={env.action_manager.total_action_dim}"
  )
  obs, _ = env.reset()
  n_act = env.action_manager.total_action_dim
  rew_sum = {}
  fell = 0
  term_counts: dict[str, int] = {}
  t0 = time.time()
  for i in range(steps):
    if agent == "zero":
      act = torch.zeros(num_envs, n_act, device=device)
    else:
      act = torch.randn(num_envs, n_act, device=device) * 0.5
    obs, rew, terminated, truncated, info = env.step(act)
    for k, v in env.reward_manager._episode_sums.items():  # noqa: SLF001
      rew_sum[k] = float(v.mean())
    fell += int(terminated.sum())
    for name in env.termination_manager.active_terms:
      term_counts[name] = term_counts.get(name, 0) + int(
        env.termination_manager.get_term(name).sum()
      )
    if not torch.isfinite(rew).all() or not torch.isfinite(obs["actor"]).all():
      raise RuntimeError(f"NaN/inf at step {i}")
  dt = time.time() - t0
  print(
    f"stepped {steps}x{num_envs} envs in {dt:.1f}s ({steps * num_envs / dt:,.0f} env-steps/s); "
    f"terminated={fell} by term: {term_counts}"
  )
  base_h = env.scene["robot"].data.root_link_pos_w[:, 2]
  print(
    f"base height after {steps} steps: mean={float(base_h.mean()):.3f} min={float(base_h.min()):.3f} "
    f"(standing={robot.standing_height:.3f})"
  )
  print("episode reward sums (mean over envs):")
  for k, v in sorted(rew_sum.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k:24s} {v:+.3f}")
  env.close()


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("robots", nargs="*", default=[r.name for r in ALL_ROBOTS])
  p.add_argument("--num-envs", type=int, default=64)
  p.add_argument("--steps", type=int, default=100)
  p.add_argument("--agent", choices=("zero", "random"), default="zero")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  for r in args.robots:
    run(r, args.num_envs, args.steps, args.agent, args.device)


if __name__ == "__main__":
  main()
