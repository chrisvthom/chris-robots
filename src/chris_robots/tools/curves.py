"""Print a compact table of training curves from a run's tensorboard log.

Usage: uv run curves kbot [--run NAME] [--rows 12]
"""

from __future__ import annotations

import argparse

from chris_robots import _REPO_ROOT

LOGS_DIR = _REPO_ROOT / "logs" / "rsl_rl"

COLUMNS = {
  "Train/mean_reward": "reward",
  "Train/mean_episode_length": "ep_len",
  "Episode_Termination/fell_over": "fell",
  "Metrics/twist/error_vel_xy": "err_xy",
  "Metrics/twist/error_vel_yaw": "err_yaw",
  "Metrics/peak_height_mean": "peak_h",
  "Metrics/air_time_mean": "air_t",
  "Episode_Reward/track_linear_velocity": "r_linvel",
  "Episode_Reward/pose": "r_pose",
  "Episode_Reward/foot_clearance": "r_clear",
  "Episode_Reward/action_rate_l2": "r_actrate",
  "Episode_Reward/air_time_biped": "r_airbip",
  "Episode_Reward/stride": "r_stride",
  "Episode_Reward/lean": "r_lean",
  "Episode_Reward/arm_swing": "r_arms",
  "Train/mean_action_std": "act_std",
}


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("robot")
  p.add_argument("--run", default=None, help="run dir name (default: latest)")
  p.add_argument("--rows", type=int, default=12)
  args = p.parse_args()

  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  exp_dir = LOGS_DIR / f"{args.robot}_velocity"
  run = exp_dir / args.run if args.run else sorted(d for d in exp_dir.iterdir() if d.is_dir())[-1]
  ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
  ea.Reload()
  tags = set(ea.Tags()["scalars"])
  cols = [(t, n) for t, n in COLUMNS.items() if t in tags]
  series = {t: {e.step: e.value for e in ea.Scalars(t)} for t, _ in cols}
  steps = sorted(series[cols[0][0]])
  stride = max(1, len(steps) // args.rows)
  picks = steps[::stride]
  if picks[-1] != steps[-1]:
    picks.append(steps[-1])

  print(f"{run.name}  ({len(steps)} iterations)")
  print(f"{'iter':>6} " + " ".join(f"{n:>9}" for _, n in cols))
  for s in picks:
    print(f"{s:>6} " + " ".join(f"{series[t].get(s, float('nan')):9.3f}" for t, _ in cols))


if __name__ == "__main__":
  main()
