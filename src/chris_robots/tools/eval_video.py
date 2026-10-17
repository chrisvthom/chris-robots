"""Render evaluation videos of a policy at specific training checkpoints.

    uv run eval-video kbot                       # latest run, checkpoints 500/1000/.../latest
    uv run eval-video kbot --checkpoints 300,800  # specific iterations
    uv run eval-video kbot --run <dir> --checkpoints latest

For each checkpoint we export an ONNX actor (cached under .cache/eval/), build a
standalone bundle (same code path as export-web) and roll it out through a scripted
command sequence: stand, walk, turn, sidestep, fast walk. Output per checkpoint under
media/evals/<robot>/: <run>_<iter>.mp4, .gif and a .png contact sheet.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chris_robots import _REPO_ROOT
from chris_robots.robots import ROBOTS_BY_NAME
from chris_robots.robots.base import RobotDef
from chris_robots.tools.export_web import BundleRunner, find_run_dir, write_bundle

EVAL_CACHE = _REPO_ROOT / ".cache" / "eval"
MEDIA_DIR = _REPO_ROOT / "media" / "evals"
FONT = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"


# --------------------------------------------------------------------------- onnx from .pt


def export_checkpoints_to_onnx(robot: RobotDef, run_dir: Path, ckpts: list[Path]) -> list[Path]:
  """Export ONNX for each checkpoint (cached). Builds one small env + runner."""
  out_dir = EVAL_CACHE / robot.name / run_dir.name
  out_dir.mkdir(parents=True, exist_ok=True)
  outs = [out_dir / f"{c.stem}.onnx" for c in ckpts]
  todo = [(c, o) for c, o in zip(ckpts, outs, strict=True) if not o.exists()]
  if not todo:
    return outs

  import torch  # noqa: F401  (torch before mjlab)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import chris_robots  # noqa: F401  (registers tasks)

  env_cfg = load_env_cfg(robot.task_id, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(robot.task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(robot.task_id)(wrapped, asdict(agent_cfg), device="cuda:0")
  metadata = get_base_metadata(env, run_dir.name)
  for ckpt, out in todo:
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    runner.export_policy_to_onnx(str(out.parent), out.name)
    attach_metadata_to_onnx(str(out), metadata)
    print(f"  exported {out.relative_to(_REPO_ROOT)}")
  env.close()
  return outs


def pick_checkpoints(run_dir: Path, spec: str) -> list[Path]:
  available = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not available:
    raise FileNotFoundError(f"no checkpoints in {run_dir}")
  by_iter = {int(p.stem.split("_")[1]): p for p in available}
  latest = available[-1]
  if spec == "auto":
    iters = sorted(
      {i for i in by_iter if i > 0 and i % 500 == 0} | {int(latest.stem.split("_")[1])}
    )
    return [by_iter[i] for i in iters]
  picks = []
  for tok in spec.split(","):
    tok = tok.strip()
    if tok == "latest":
      picks.append(latest)
    else:
      want = int(tok)
      # nearest available checkpoint at or below the requested iteration
      cands = [i for i in by_iter if i <= want]
      picks.append(by_iter[max(cands)] if cands else available[0])
  return list(dict.fromkeys(picks))


# --------------------------------------------------------------------------- rollout


def command_script(robot: RobotDef) -> list[tuple[str, float, np.ndarray]]:
  """(label, duration s, [vx, vy, wz]) segments: stand, walk, turn, sidestep, jog, sprint."""
  v, run, w = robot.walk_vel, robot.max_lin_vel, robot.max_ang_vel
  jog = 0.5 * (v + run)
  return [
    ("stand", 1.5, np.array([0.0, 0.0, 0.0])),
    (f"walk {v:.2f} m/s", 3.5, np.array([v, 0.0, 0.0])),
    ("turn left", 3.0, np.array([0.5 * v, 0.0, 0.6 * w])),
    ("sidestep", 2.0, np.array([0.0, 0.5 * v, 0.0])),
    (f"jog {jog:.2f} m/s", 3.0, np.array([jog, 0.0, 0.0])),
    (f"sprint {run:.2f} m/s", 3.5, np.array([run, 0.0, 0.0])),
    ("stop", 1.5, np.array([0.0, 0.0, 0.0])),
  ]


def render_eval(
  bundle: Path,
  robot: RobotDef,
  title: str,
  out_stem: Path,
  fps: int = 25,
  width: int = 720,
  height: int = 450,
) -> dict:
  runner = BundleRunner(bundle)
  m, d = runner.model, runner.data
  dt = runner.cfg["sim"]["control_dt"]
  every = max(1, round(1.0 / (fps * dt)))
  # Brighter than the default scene so dark robots read well in small videos.
  m.vis.headlight.ambient[:] = 0.45
  m.vis.headlight.diffuse[:] = 0.75
  m.vis.headlight.specular[:] = 0.1
  renderer = mujoco.Renderer(m, height=height, width=width)
  # Free camera that we re-aim at mid-body every frame (a tracking camera would look
  # at the base, wasting half the frame above the head). Distance frames the whole
  # robot with a 25% margin.
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  mujoco.mj_forward(m, d)
  robot_geoms = np.flatnonzero(m.geom_bodyid != 0)
  top = float(np.max(d.geom_xpos[robot_geoms, 2] + m.geom_rbound[robot_geoms]))
  mid_z = 0.5 * top
  cam.distance = (0.625 * top) / math.tan(math.radians(m.vis.global_.fovy / 2))
  cam.azimuth, cam.elevation = 150, -10
  lookat = np.array([0.0, 0.0, mid_z])
  font = ImageFont.truetype(FONT, 22)
  font_small = ImageFont.truetype(FONT, 17)

  frames: list[np.ndarray] = []
  frame_meta: list[str] = []
  stats: dict[str, dict] = {}
  foot_sites = [
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in robot.foot_order
  ]
  fell_at = None
  t = 0.0
  for label, duration, cmd in command_script(robot):
    seg_vel, seg_clear = [], []
    start = runner.base_pos().copy()
    n = int(duration / dt)
    for i in range(n):
      runner.step(cmd.astype(np.float32))
      t += dt
      seg_vel.append(d.sensordata[runner.velo][:2].copy())
      seg_clear.append(max(d.site_xpos[s][2] for s in foot_sites))
      if fell_at is None and (
        runner.upright() < 0.5
        or runner.base_pos()[2] < 0.4 * runner.cfg["sim"]["standing_base_height_m"]
      ):
        fell_at = t
      if i % every == 0:
        # Smoothly follow the base in xy (first-order lag hides the step bobbing).
        lookat[:2] += 0.25 * (runner.base_pos()[:2] - lookat[:2])
        cam.lookat[:] = lookat
        renderer.update_scene(d, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width, 34], fill=(0, 0, 0))
        draw.text((10, 6), title, font=font, fill=(255, 255, 255))
        cmd_txt = f"{label}   cmd: vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}   t={t:4.1f}s"
        draw.rectangle([0, height - 28, width, height], fill=(0, 0, 0))
        draw.text((10, height - 24), cmd_txt, font=font_small, fill=(230, 230, 230))
        if fell_at is not None:
          draw.text((width - 90, 6), "FELL", font=font, fill=(255, 80, 80))
        frames.append(np.asarray(img))
        frame_meta.append(f"{label}  t={t:.1f}s")
    vel = np.array(seg_vel)
    disp = runner.base_pos()[:2] - start[:2]
    stats[label] = {
      "cmd": cmd.round(2).tolist(),
      "mean_vel_b": vel.mean(0).round(2).tolist(),
      "dist_m": float(np.linalg.norm(disp).round(2)),
      "max_foot_clearance_m": float(np.max(seg_clear).round(3)),
    }
  renderer.close()

  mp4 = out_stem.with_suffix(".mp4")
  proc = subprocess.Popen(
    [
      "ffmpeg",
      "-y",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-s",
      f"{width}x{height}",
      "-r",
      str(fps),
      "-i",
      "-",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "-crf",
      "22",
      str(mp4),
    ],
    stdin=subprocess.PIPE,
  )
  assert proc.stdin is not None
  for f in frames:
    proc.stdin.write(f.tobytes())
  proc.stdin.close()
  proc.wait()

  gif = out_stem.with_suffix(".gif")
  small = [
    Image.fromarray(f).resize((width // 2, height // 2), Image.LANCZOS).quantize(96)
    for f in frames[::2]
  ]
  small[0].save(
    gif, save_all=True, append_images=small[1:], duration=int(2000 / fps), loop=0, optimize=True
  )

  # Contact sheet: full-resolution centre crops, 3 frames per moving segment.
  crop_w = int(width * 0.42)
  x0 = (width - crop_w) // 2
  sheet_frames = []
  seg_bounds, acc = [], 0
  for label, duration, _ in command_script(robot):
    n = int(duration / dt) // every
    if label not in ("stand", "stop"):
      seg_bounds.append((acc, acc + n))
    acc += n
  per_seg = 3
  bar = 34
  for a, b in seg_bounds:
    for j in range(per_seg):
      k = min(a + (b - a) * (2 * j + 1) // (2 * per_seg), len(frames) - 1)
      crop = Image.fromarray(frames[k]).crop((x0, bar, x0 + crop_w, height))
      draw = ImageDraw.Draw(crop)
      draw.rectangle([0, crop.height - 28, crop_w, crop.height], fill=(0, 0, 0))
      draw.text((8, crop.height - 24), frame_meta[k], font=font_small, fill=(230, 230, 230))
      sheet_frames.append(crop)
  cols = per_seg * 2
  rows = (len(sheet_frames) + cols - 1) // cols
  crop_h = height - bar
  sheet = Image.new("RGB", (cols * crop_w, bar + rows * crop_h))
  ImageDraw.Draw(sheet).text((10, 6), title, font=font, fill=(255, 255, 255))
  for i, f in enumerate(sheet_frames):
    sheet.paste(f, ((i % cols) * crop_w, bar + (i // cols) * crop_h))
  sheet.save(out_stem.with_suffix(".png"))

  return {
    "fell_at_s": fell_at,
    "segments": stats,
    "mp4": str(mp4),
    "gif": str(gif),
    "sheet": str(out_stem.with_suffix(".png")),
  }


def main() -> None:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("robot")
  p.add_argument("--run", default=None)
  p.add_argument(
    "--checkpoints", default="auto", help="comma-separated iterations, 'latest', or 'auto'"
  )
  args = p.parse_args()

  robot = ROBOTS_BY_NAME[args.robot]
  run_dir = (
    find_run_dir(robot, args.run)
    if args.run
    else sorted(d for d in (find_run_dir(robot, None).parent).iterdir() if d.is_dir())[-1]
  )
  ckpts = pick_checkpoints(run_dir, args.checkpoints)
  print(f"== {robot.display_name}: run {run_dir.name}, checkpoints {[c.stem for c in ckpts]}")
  onnxs = export_checkpoints_to_onnx(robot, run_dir, ckpts)

  out_dir = MEDIA_DIR / robot.name
  out_dir.mkdir(parents=True, exist_ok=True)
  for ckpt, onnx in zip(ckpts, onnxs, strict=True):
    it = int(ckpt.stem.split("_")[1])
    bundle = EVAL_CACHE / robot.name / run_dir.name / f"bundle_{it}"
    write_bundle(robot, onnx, bundle, f"{run_dir.name}/{ckpt.stem}", copy_assets=False)
    title = f"{robot.display_name}  -  {run_dir.name}  -  iteration {it}"
    res = render_eval(bundle, robot, title, out_dir / f"{run_dir.name}_{it:05d}")
    fell = f"FELL at {res['fell_at_s']:.1f}s" if res["fell_at_s"] else "no fall"
    print(f"-- iteration {it}: {fell}")
    for label, s in res["segments"].items():
      print(
        f"     {label:16s} cmd={s['cmd']}  mean v_b={s['mean_vel_b']}  dist={s['dist_m']:.2f} m"
        f"  max foot clearance={s['max_foot_clearance_m']:.3f} m"
      )
    print(f"     {Path(res['mp4']).relative_to(_REPO_ROOT)}  |  {Path(res['sheet']).name}")


if __name__ == "__main__":
  main()
