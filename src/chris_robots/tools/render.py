"""Render still images of each robot (for the README and for eyeballing imports).

Usage:
  uv run render-robots                 # all robots, default pose -> media/<name>.png
  uv run render-robots kbot --init     # use the mjlab init_state keyframe instead of qpos0
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from chris_robots import ROBOTS_DIR

MEDIA_DIR = ROBOTS_DIR.parent / "media"


def _bbox(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  for i in range(model.ngeom):
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
      continue
    r = model.geom_rbound[i]
    lo = np.minimum(lo, data.geom_xpos[i] - r)
    hi = np.maximum(hi, data.geom_xpos[i] + r)
  return lo, hi


def render_scene(
  xml_path: Path,
  out_path: Path,
  *,
  width: int = 960,
  height: int = 1080,
  azimuth: float = 135.0,
  elevation: float = -12.0,
  keyframe: str | None = None,
  show_sites: bool = False,
) -> None:
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, height)
  data = mujoco.MjData(model)
  if keyframe is not None:
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    assert kid >= 0, f"keyframe {keyframe!r} not found"
    mujoco.mj_resetDataKeyframe(model, data, kid)
  mujoco.mj_forward(model, data)

  lo, hi = _bbox(model, data)
  center = (lo + hi) / 2
  size = hi - lo
  fovy = math.radians(float(model.vis.global_.fovy))
  tan_v = math.tan(fovy / 2)
  tan_h = tan_v * (width / height)
  diag_xy = math.hypot(size[0], size[1])
  dist = max(size[2] * 1.1 / (2 * tan_v), diag_xy * 1.1 / (2 * tan_h)) + diag_xy / 2

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = center
  cam.distance = dist
  cam.azimuth = azimuth
  cam.elevation = elevation

  opt = mujoco.MjvOption()
  if show_sites:
    opt.sitegroup[5] = 1
  else:
    opt.sitegroup[5] = 0

  renderer = mujoco.Renderer(model, height=height, width=width)
  renderer.update_scene(data, camera=cam, scene_option=opt)
  img = renderer.render()
  renderer.close()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(img).save(out_path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "robots", nargs="*", default=sorted(p.name for p in ROBOTS_DIR.iterdir() if p.is_dir())
  )
  parser.add_argument(
    "--init", action="store_true", help="use the 'init_state' keyframe if present"
  )
  parser.add_argument("--sites", action="store_true", help="draw group-5 sites (imu, feet)")
  parser.add_argument("--out", type=Path, default=MEDIA_DIR)
  args = parser.parse_args()
  for name in args.robots:
    scene = ROBOTS_DIR / name / "scene.xml"
    out = args.out / f"{name}.png"
    render_scene(scene, out, keyframe="init_state" if args.init else None, show_sites=args.sites)
    print(f"wrote {out}")


if __name__ == "__main__":
  main()
