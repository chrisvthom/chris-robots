"""Bundle a trained policy into a self-contained folder for the web.

Output layout (``policies/<robot>/``)::

  scene.xml       MJCF with floor, lights, the robot, <position> actuators (one per
                  policy action, in action order) and an ``init_state`` keyframe.
  assets/         meshes referenced by scene.xml
  policy.onnx     actor network (obs normalizer included). input "obs" -> output "actions"
  policy.json     everything the runtime needs to build the observation vector and
                  turn actions into actuator targets (see ``build_policy_json``).
  preview.mp4/gif short rollout of the bundle driven by the policy (if --preview)

The bundle only needs MuJoCo (mujoco-wasm in the browser) and an ONNX runtime; mjlab
is not required to run it. ``--check`` proves that by rolling the bundle out here
with plain ``mujoco`` + ``onnxruntime``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import onnx
from mjlab.utils.spec import create_position_actuator

from chris_robots import _REPO_ROOT, POLICIES_DIR
from chris_robots.robots import ROBOTS_BY_NAME
from chris_robots.robots.base import RobotDef

LOGS_DIR = _REPO_ROOT / "logs" / "rsl_rl"

# Must match mjlab's velocity task (velocity_env_cfg.py).
SIM_DT = 0.005
DECIMATION = 4


# --------------------------------------------------------------------------- runs


def find_run_dir(robot: RobotDef, run: str | None) -> Path:
  exp_dir = LOGS_DIR / f"{robot.name}_velocity"
  if run is not None:
    p = Path(run)
    if p.is_dir():
      return p
    if (exp_dir / run).is_dir():
      return exp_dir / run
    raise FileNotFoundError(f"no run '{run}' under {exp_dir}")
  runs = sorted(d for d in exp_dir.iterdir() if d.is_dir() and list(d.glob("*.onnx")))
  if not runs:
    raise FileNotFoundError(f"no runs with an exported .onnx under {exp_dir}")
  return runs[-1]


def read_onnx_metadata(onnx_path: Path) -> dict:
  model = onnx.load(str(onnx_path))
  raw = {e.key: e.value for e in model.metadata_props}

  def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")] if s else []

  def strs(s: str) -> list[str]:
    return s.split(",") if s else []

  meta = {
    "joint_names": strs(raw["joint_names"]),
    "joint_stiffness": floats(raw["joint_stiffness"]),
    "joint_damping": floats(raw["joint_damping"]),
    "default_joint_pos": floats(raw["default_joint_pos"]),
    "observation_names": strs(raw["observation_names"]),
    "observation_terms_scale": [
      [float(v) for v in x.split(";")] if ";" in x else float(x)
      for x in strs(raw["observation_terms_scale"])
    ],
    "observation_terms_history_length": [
      int(float(x)) for x in strs(raw["observation_terms_history_length"])
    ],
    "observation_terms_clip": [
      [float(v) for v in x.split(";")] for x in strs(raw["observation_terms_clip"])
    ],
    "action_scale": floats(raw["action_scale"]),
  }
  inp = model.graph.input[0]
  out = model.graph.output[0]
  meta["input_name"] = inp.name
  meta["output_name"] = out.name
  meta["obs_dim"] = int(inp.type.tensor_type.shape.dim[-1].dim_value)
  meta["action_dim"] = int(out.type.tensor_type.shape.dim[-1].dim_value)
  if len(meta["action_scale"]) == 1:
    meta["action_scale"] = meta["action_scale"] * meta["action_dim"]
  return meta


# --------------------------------------------------------------------------- scene


def build_scene_spec(robot: RobotDef, meta: dict) -> mujoco.MjSpec:
  """robots/<name>/scene.xml + feet-only collisions + position actuators + keyframe."""
  spec = mujoco.MjSpec.from_file(str(robot.dir / "scene.xml"))
  robot.apply_joint_overrides(spec)
  _apply_sim_options(spec, robot)

  # Same collision structure mjlab applied during training. mjlab applies this to the
  # robot-only spec; here the floor is in the same spec, so restore it afterwards.
  robot.collision_cfg().edit_spec(spec)
  floor = spec.geom("floor")
  floor.contype, floor.conaffinity = 1, 0

  # One <position> actuator per policy action, in action order, with the exact
  # gains the policy was trained with (read back from the ONNX metadata).
  for jname, kp, kd in zip(
    meta["joint_names"], meta["joint_stiffness"], meta["joint_damping"], strict=False
  ):
    group = next(a for a in robot.actuators if _matches_any(jname, a.joints))
    create_position_actuator(
      spec,
      jname,
      stiffness=kp,
      damping=kd,
      effort_limit=group.effort_limit,
      armature=group.armature,
    )

  # Keyframe: init pose + ctrl = default joint targets so the robot holds still at t=0.
  key = spec.key("init_state")
  key.ctrl = np.array(meta["default_joint_pos"])
  spec.compile()
  return spec


def _apply_sim_options(spec: mujoco.MjSpec, robot: RobotDef) -> None:
  """Bake the training-time <option> values into the bundle.

  mjlab overrides the model's options at load time (0.005 s implicitfast, Newton
  solver, ...). The K-Scale MJCFs carry their own <option timestep="0.002">, so a
  bundle that keeps them would run 2.5x slow motion and see different dynamics.
  """
  from mjlab.sim.sim import _CONE_MAP, _INTEGRATOR_MAP, _JACOBIAN_MAP, _SOLVER_MAP

  from chris_robots.env_cfgs import make_flat_env_cfg

  mj = make_flat_env_cfg(robot, play=True).sim.mujoco
  o = spec.option
  o.timestep = mj.timestep
  o.integrator = _INTEGRATOR_MAP[mj.integrator]
  o.solver = _SOLVER_MAP[mj.solver]
  o.cone = _CONE_MAP[mj.cone]
  o.jacobian = _JACOBIAN_MAP[mj.jacobian]
  o.impratio = mj.impratio
  o.iterations = mj.iterations
  o.tolerance = mj.tolerance
  o.ls_iterations = mj.ls_iterations
  o.ls_tolerance = mj.ls_tolerance
  o.ccd_iterations = mj.ccd_iterations
  o.gravity = list(mj.gravity)
  assert abs(mj.timestep - SIM_DT) < 1e-9, "SIM_DT out of sync with mjlab config"


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
  import re

  return any(re.fullmatch(p, name) for p in patterns)


# --------------------------------------------------------------------------- json


def build_policy_json(robot: RobotDef, meta: dict, model: mujoco.MjModel, source_name: str) -> dict:
  n = meta["action_dim"]
  obs_sizes = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": n,
    "joint_vel": n,
    "actions": n,
    "command": 3,
  }
  layout, offset = [], 0
  for name, scale, hist, clip in zip(
    meta["observation_names"],
    meta["observation_terms_scale"],
    meta["observation_terms_history_length"],
    meta["observation_terms_clip"],
    strict=False,
  ):
    if name not in obs_sizes:
      raise ValueError(f"unknown observation term {name!r}; add it to obs_sizes/export")
    size = obs_sizes[name] * max(1, hist)
    layout.append(
      {"name": name, "offset": offset, "size": size, "scale": scale, "history": hist, "clip": clip}
    )
    offset += size
  if offset != meta["obs_dim"]:
    raise ValueError(f"obs layout sums to {offset} but policy expects {meta['obs_dim']}")

  qpos_adr = [
    int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
    for j in meta["joint_names"]
  ]
  qvel_adr = [
    int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
    for j in meta["joint_names"]
  ]
  v = robot.max_lin_vel
  w = robot.max_ang_vel
  return {
    "robot": robot.name,
    "display_name": robot.display_name,
    "source_run": source_name,
    "files": {"scene": "scene.xml", "policy": "policy.onnx"},
    "sim": {
      "timestep": SIM_DT,
      "decimation": DECIMATION,
      "control_dt": SIM_DT * DECIMATION,
      "keyframe": "init_state",
      "base_body": robot.base_body,
      "standing_base_height_m": robot.standing_height,
    },
    "sensors": {
      "base_lin_vel": "imu_lin_vel",
      "base_ang_vel": "imu_ang_vel",
      "note": "projected_gravity = R_base^T * [0,0,-1]; read the base orientation from the freejoint quaternion",
    },
    "joints": {
      "names": meta["joint_names"],
      "qpos_adr": qpos_adr,
      "qvel_adr": qvel_adr,
      "default_pos": meta["default_joint_pos"],
      "kp": meta["joint_stiffness"],
      "kd": meta["joint_damping"],
      "effort_limit": [
        next(a for a in robot.actuators if _matches_any(j, a.joints)).effort_limit
        for j in meta["joint_names"]
      ],
    },
    "policy": {
      "input": meta["input_name"],
      "output": meta["output_name"],
      "obs_dim": meta["obs_dim"],
      "action_dim": n,
      "observation_layout": layout,
      "action_scale": meta["action_scale"],
      "note": "ctrl[i] = default_pos[i] + action_scale[i] * action[i]; ctrl index == action index == joints.names index",
    },
    "command": {
      "order": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
      "ranges": {
        "lin_vel_x": [-v, 1.5 * v],
        "lin_vel_y": [-0.6 * v, 0.6 * v],
        "ang_vel_z": [-w, w],
      },
      "suggested_walk": [0.6 * v, 0.0, 0.0],
    },
  }


# --------------------------------------------------------------------------- runtime (check)


class BundleRunner:
  """Minimal, mjlab-free runtime. This is what the browser has to replicate."""

  def __init__(self, bundle_dir: Path):
    import onnxruntime as ort

    self.cfg = json.loads((bundle_dir / "policy.json").read_text())
    self.model = mujoco.MjModel.from_xml_path(str(bundle_dir / self.cfg["files"]["scene"]))
    self.data = mujoco.MjData(self.model)
    self.sess = ort.InferenceSession(
      str(bundle_dir / self.cfg["files"]["policy"]), providers=["CPUExecutionProvider"]
    )
    j = self.cfg["joints"]
    self.qpos_adr = np.array(j["qpos_adr"])
    self.qvel_adr = np.array(j["qvel_adr"])
    self.default = np.array(j["default_pos"], dtype=np.float32)
    self.scale = np.array(self.cfg["policy"]["action_scale"], dtype=np.float32)
    self.gyro = self._sensor_slice(self.cfg["sensors"]["base_ang_vel"])
    self.velo = self._sensor_slice(self.cfg["sensors"]["base_lin_vel"])
    self.base_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg["sim"]["base_body"]
    )
    self.reset()

  def _sensor_slice(self, name: str) -> slice:
    sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = self.model.sensor_adr[sid]
    return slice(adr, adr + self.model.sensor_dim[sid])

  def reset(self) -> None:
    mujoco.mj_resetDataKeyframe(
      self.model, self.data, self.model.key(self.cfg["sim"]["keyframe"]).id
    )
    mujoco.mj_forward(self.model, self.data)
    self.last_action = np.zeros_like(self.default)

  def observe(self, command: np.ndarray) -> np.ndarray:
    d = self.data
    quat = d.xquat[self.base_id]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    gravity_b = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    parts = {
      "base_lin_vel": d.sensordata[self.velo],
      "base_ang_vel": d.sensordata[self.gyro],
      "projected_gravity": gravity_b,
      "joint_pos": d.qpos[self.qpos_adr] - self.default,
      "joint_vel": d.qvel[self.qvel_adr],
      "actions": self.last_action,
      "command": command,
    }
    obs = np.zeros(self.cfg["policy"]["obs_dim"], dtype=np.float32)
    for term in self.cfg["policy"]["observation_layout"]:
      x = np.asarray(parts[term["name"]], dtype=np.float32) * np.float32(term["scale"])
      x = np.clip(x, term["clip"][0], term["clip"][1])
      obs[term["offset"] : term["offset"] + term["size"]] = x
    return obs

  def step(self, command: np.ndarray) -> None:
    obs = self.observe(command)
    action = self.sess.run(None, {self.cfg["policy"]["input"]: obs[None]})[0][0]
    self.last_action = action
    self.data.ctrl[:] = self.default + self.scale * action
    for _ in range(self.cfg["sim"]["decimation"]):
      mujoco.mj_step(self.model, self.data)

  def base_pos(self) -> np.ndarray:
    return self.data.xpos[self.base_id].copy()

  def upright(self) -> float:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, self.data.xquat[self.base_id])
    return float(rot.reshape(3, 3)[2, 2])  # cos(tilt)


def check_bundle(bundle_dir: Path, seconds: float, command: tuple[float, float, float]) -> bool:
  """Roll the bundle out and print a short gait report."""
  runner = BundleRunner(bundle_dir)
  m, d = runner.model, runner.data
  cmd = np.array(command, dtype=np.float32)
  dt = runner.cfg["sim"]["control_dt"]
  start = runner.base_pos()
  steps = int(seconds / dt)
  fell = False
  min_height = 0.4 * runner.cfg["sim"]["standing_base_height_m"]

  foot_sites = [
    mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")
  ]
  names = runner.cfg["joints"]["names"]
  roll_idx = [i for i, n in enumerate(names) if "hip_roll" in n]
  foot_z = np.zeros((steps, 2))
  hip_roll = np.zeros((steps, len(roll_idx)))
  for i in range(steps):
    runner.step(cmd)
    foot_z[i] = [d.site_xpos[s][2] for s in foot_sites]
    hip_roll[i] = d.qpos[runner.qpos_adr[roll_idx]] if roll_idx else 0.0
    if runner.upright() < 0.5 or runner.base_pos()[2] < min_height:
      fell = True
      print(
        f"  FELL at t={i * dt:.2f}s (upright={runner.upright():.2f}, h={runner.base_pos()[2]:.2f})"
      )
      foot_z, hip_roll = foot_z[: i + 1], hip_roll[: i + 1]
      break

  end = runner.base_pos()
  dist = end[:2] - start[:2]
  # Swing phases: foot more than 5 mm above its resting height.
  rest = foot_z[: max(1, int(0.2 / dt))].min(0)
  peaks, n_swings = [], 0
  for f in range(2):
    up = foot_z[:, f] > rest[f] + 0.005
    edges = np.flatnonzero(np.diff(up.astype(int)) == 1)
    n_swings += len(edges)
    for s in edges:
      rest_of = ~up[s + 1 :]
      e = s + 1 + int(np.argmax(rest_of)) if rest_of.any() else len(up)
      peaks.append(foot_z[s:e, f].max() - rest[f])
  cadence = n_swings / 2 / max(1e-6, len(foot_z) * dt)  # steps per second per foot
  sway = float(np.ptp(hip_roll, axis=0).mean()) if roll_idx else float("nan")
  print(
    f"  check: cmd={command} -> moved dx={dist[0]:+.2f} dy={dist[1]:+.2f} m in {seconds:.0f}s"
    f" (avg {np.linalg.norm(dist) / seconds:.2f} m/s), final height {end[2]:.2f}, upright={runner.upright():.2f}"
  )
  print(
    f"  gait:  {n_swings} swings, cadence {cadence:.2f} steps/s/foot, mean peak clearance "
    f"{np.mean(peaks) if peaks else 0:.3f} m, hip roll range {sway:.2f} rad"
  )
  return not fell


def render_preview(
  bundle_dir: Path,
  seconds: float,
  command: tuple[float, float, float],
  fps: int = 25,
  width: int = 640,
  height: int = 400,
) -> None:
  """Write preview.mp4 (ffmpeg) and a small preview.gif next to the bundle."""
  import subprocess

  from PIL import Image

  runner = BundleRunner(bundle_dir)
  cmd = np.array(command, dtype=np.float32)
  renderer = mujoco.Renderer(runner.model, height=height, width=width)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = runner.base_id
  cam.distance = max(1.5, 2.6 * runner.cfg["sim"]["standing_base_height_m"])
  cam.azimuth, cam.elevation = 135, -15
  frames = []
  control_dt = runner.cfg["sim"]["control_dt"]
  every = max(1, round(1.0 / (fps * control_dt)))
  for i in range(int(seconds / control_dt)):
    runner.step(cmd)
    if i % every == 0:
      renderer.update_scene(runner.data, camera=cam)
      frames.append(renderer.render().copy())
  renderer.close()

  mp4 = bundle_dir / "preview.mp4"
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
      "23",
      str(mp4),
    ],
    stdin=subprocess.PIPE,
  )
  assert proc.stdin is not None
  for f in frames:
    proc.stdin.write(f.tobytes())
  proc.stdin.close()
  proc.wait()

  gif = bundle_dir / "preview.gif"
  small = [
    Image.fromarray(f).resize((width // 2, height // 2), Image.LANCZOS).quantize(64)
    for f in frames[::2]
  ]
  small[0].save(
    gif, save_all=True, append_images=small[1:], duration=int(2000 / fps), loop=0, optimize=True
  )
  print(
    f"  wrote {mp4.name} ({mp4.stat().st_size / 1e6:.1f} MB) and {gif.name} ({gif.stat().st_size / 1e6:.1f} MB), {len(frames)} frames"
  )


# --------------------------------------------------------------------------- main


def write_bundle(
  robot: RobotDef, onnx_src: Path, out: Path, source_name: str, copy_assets: bool = True
) -> dict:
  """Write a standalone bundle for ``onnx_src`` into ``out`` (replacing it). Returns policy.json."""
  meta = read_onnx_metadata(onnx_src)
  if list(meta["joint_names"]) != list(robot.joint_names):
    raise ValueError("policy joint order differs from robot.json joint order")
  if out.exists():
    shutil.rmtree(out)
  out.mkdir(parents=True)

  spec = build_scene_spec(robot, meta)
  model = spec.compile()
  if copy_assets:
    spec.meshdir = "assets"
    shutil.copytree(robot.dir / "assets", out / "assets")
  else:  # scratch bundles (evals) just point at the repo's meshes
    spec.meshdir = str(robot.dir / "assets")
  (out / "scene.xml").write_text(spec.to_xml())
  shutil.copy2(onnx_src, out / "policy.onnx")
  policy_json = build_policy_json(robot, meta, model, source_name)
  (out / "policy.json").write_text(json.dumps(policy_json, indent=2))
  return policy_json


def export(
  robot: RobotDef, run: str | None, out_root: Path, check: bool, gif: bool, seconds: float
) -> Path:
  run_dir = find_run_dir(robot, run)
  onnx_src = sorted(run_dir.glob("*.onnx"))[-1]
  print(f"== {robot.display_name}: {run_dir.name} / {onnx_src.name}")
  out = out_root / robot.name
  write_bundle(robot, onnx_src, out, run_dir.name)
  size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
  print(f"  wrote {out} ({size_mb:.1f} MB)")

  cmd = tuple(json.loads((out / "policy.json").read_text())["command"]["suggested_walk"])
  ok = True
  if check:
    ok = check_bundle(out, seconds, cmd)
    print(
      "  OK: policy walks in the standalone bundle"
      if ok
      else "  WARNING: robot fell in the standalone bundle"
    )
  if gif:
    render_preview(out, seconds, cmd)
  return out


def main() -> None:
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument(
    "robots",
    nargs="*",
    default=list(ROBOTS_BY_NAME),
    help="robot names (default: all with a trained run)",
  )
  p.add_argument(
    "--run", default=None, help="run dir name or path (default: latest run with an .onnx)"
  )
  p.add_argument("--out", type=Path, default=POLICIES_DIR)
  p.add_argument(
    "--no-check", action="store_true", help="skip the standalone mujoco+onnxruntime rollout"
  )
  p.add_argument(
    "--gif",
    "--preview",
    dest="gif",
    action="store_true",
    help="also write preview.mp4 + preview.gif",
  )
  p.add_argument("--seconds", type=float, default=8.0)
  args = p.parse_args()

  for name in args.robots:
    robot = ROBOTS_BY_NAME[name]
    try:
      export(robot, args.run, args.out, check=not args.no_check, gif=args.gif, seconds=args.seconds)
    except FileNotFoundError as e:
      print(f"== {robot.display_name}: skipped ({e})")


if __name__ == "__main__":
  main()
