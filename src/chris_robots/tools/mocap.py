"""Retarget LAFAN1 human mocap (via Unitree's G1 retargeting) onto one of our robots.

``uv run mocap kbot`` downloads the walk/run/sprint clips of
``lvhaidong/LAFAN1_Retargeting_Dataset`` (Unitree G1, 30 fps, CC BY-NC-ND 4.0 -- the
data is *not* committed, see ``data/``), maps the G1 joint angles onto the robot's
joints, and writes ``data/mocap/<robot>.npz`` with the AMP feature trajectory at the
policy rate (50 Hz).

Retargeting is joint-space: for every robot joint we find the G1 joint with the same
role (``left hip pitch`` ...) and solve ``q_robot = s * (q_g1 - phi)`` where ``s`` is the
sign of the dot product of the two joint axes at the zero pose and ``phi`` is the angle
(about that axis) between the two robots' limb-segment directions at the zero pose.
That handles mirrored axis conventions and differing rest poses (the G1's forearm
points forward at q=0, ours hang down). Lengths are scaled by hip-to-ankle leg length,
time by its square root (Froude scaling), and the root height is shifted so the lowest
foot touches the ground.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import mujoco
import numpy as np

from chris_robots import ROBOTS_DIR
from chris_robots.robots import ROBOTS_BY_NAME as ROBOTS
from chris_robots.robots import RobotDef

DATA_DIR = ROBOTS_DIR.parent / "data" / "mocap"
MEDIA_DIR = ROBOTS_DIR.parent / "media" / "mocap"
HF_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
OUT_FPS = 50
IN_FPS = 30

# Canonical joint roles; every G1 joint has exactly one.
_G1_ROLE = {
  "hip_pitch": "hip_pitch",
  "hip_roll": "hip_roll",
  "hip_yaw": "hip_yaw",
  "knee": "knee",
  "ankle_pitch": "ankle_pitch",
  "ankle_roll": "ankle_roll",
  "shoulder_pitch": "shoulder_pitch",
  "shoulder_roll": "shoulder_roll",
  "shoulder_yaw": "shoulder_yaw",
  "elbow": "elbow",
  "wrist_roll": "wrist_roll",
  "wrist_pitch": "wrist_pitch",
  "wrist_yaw": "wrist_yaw",
}
# Where each role's limb segment ends (role of the next joint, or a foot site).
_DISTAL = {
  "hip_pitch": "knee",
  "hip_roll": "knee",
  "hip_yaw": "knee",
  "knee": "ankle_pitch",
  "ankle_pitch": "foot",
  "shoulder_pitch": "elbow",
  "shoulder_roll": "elbow",
  "shoulder_yaw": "elbow",
  "elbow": "wrist",
}
_PITCH_ROLES = {"hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow"}


def _g1_xml() -> Path:
  import mjlab

  return Path(mjlab.__file__).parent / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"


def canonical_role(joint_name: str) -> tuple[str, str] | None:
  """Map a joint name to ``(side, role)`` or None if it has no human counterpart."""
  n = joint_name.lower()
  if "left" in n or re.search(r"(^|_)l(_|$)", n):
    side = "left"
  elif "right" in n or re.search(r"(^|_)r(_|$)", n):
    side = "right"
  else:
    return None
  n = re.sub(r"_\d+$", "", n)  # kbot's motor-size suffix
  for role in (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
  ):
    if role in n:
      return side, role
  if "ankle" in n:
    return side, "ankle_pitch"
  if "wrist" in n:
    return side, "wrist_yaw"
  if "knee" in n:
    return side, "knee"
  return None


class _Chain:
  """Zero-pose geometry of a robot: joint anchors/axes and foot sites by (side, role)."""

  def __init__(self, model: mujoco.MjModel, foot_sites: tuple[str, str] | None = None):
    self.model = model
    self.data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, self.data)
    self.joints: dict[tuple[str, str], int] = {}
    self.names: list[str] = []
    for j in range(model.njnt):
      if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
      name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
      self.names.append(name)
      key = canonical_role(name)
      if key is not None and key not in self.joints:
        self.joints[key] = j
    self.feet: dict[str, np.ndarray] = {}
    if foot_sites is None:
      foot_sites = ("left_foot", "right_foot")
    for s in foot_sites:
      sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
      side = "left" if "left" in s.lower() else "right"
      self.feet[side] = self.data.site_xpos[sid].copy()
    self.root_z = float(self.data.xpos[1][2]) if model.nbody > 1 else 0.0

  def axis(self, key: tuple[str, str]) -> np.ndarray:
    return self.data.xaxis[self.joints[key]].copy()

  def anchor(self, key: tuple[str, str]) -> np.ndarray:
    return self.data.xanchor[self.joints[key]].copy()

  def distal(self, key: tuple[str, str]) -> np.ndarray | None:
    side, role = key
    target = _DISTAL.get(role)
    if target is None:
      return None
    if target == "foot":
      return self.feet[side]
    if target == "wrist":
      for r in ("wrist_roll", "wrist_pitch", "wrist_yaw"):
        if (side, r) in self.joints:
          return self.anchor((side, r))
      return None
    if (side, target) in self.joints:
      return self.anchor((side, target))
    return None

  def leg_length(self) -> float:
    return float(
      np.linalg.norm(self.anchor(("left", "hip_pitch")) - self.anchor(("left", "ankle_pitch")))
    )


def build_mapping(robot: _Chain, g1: _Chain) -> list[tuple[str, int, float, float]]:
  """Return ``(robot_joint_name, g1_column, sign, offset)`` for every mappable joint."""
  g1_order = [
    mujoco.mj_id2name(g1.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, g1.model.njnt)
  ]
  out = []
  for key, j in robot.joints.items():
    if key not in g1.joints:
      continue
    a_r = robot.axis(key)
    a_g = g1.axis(key)
    dot = float(np.dot(a_r, a_g))
    if abs(dot) < 0.7:
      print(f"  skip {key}: axes differ ({a_r.round(2)} vs {a_g.round(2)})")
      continue
    sign = 1.0 if dot > 0 else -1.0
    phi = 0.0
    if key[1] in _PITCH_ROLES:
      d_r = robot.distal(key)
      d_g = g1.distal(key)
      if d_r is not None and d_g is not None:
        d_r = d_r - robot.anchor(key)
        d_g = d_g - g1.anchor(key)
        d_r -= a_g * np.dot(d_r, a_g)
        d_g -= a_g * np.dot(d_g, a_g)
        if np.linalg.norm(d_r) > 1e-6 and np.linalg.norm(d_g) > 1e-6:
          d_r /= np.linalg.norm(d_r)
          d_g /= np.linalg.norm(d_g)
          phi = math.atan2(float(np.dot(np.cross(d_g, d_r), a_g)), float(np.dot(d_g, d_r)))
    name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_JOINT, j)
    col = g1_order.index(mujoco.mj_id2name(g1.model, mujoco.mjtObj.mjOBJ_JOINT, g1.joints[key]))
    out.append((name, col, sign, phi))
  return out


def _quat_wxyz_from_xyzw(q: np.ndarray) -> np.ndarray:
  return q[:, [3, 0, 1, 2]]


def _quat_normalize(q: np.ndarray) -> np.ndarray:
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quat_inv(q: np.ndarray) -> np.ndarray:
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  )


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate world vectors ``v`` into the frame of unit quaternions ``q`` (wxyz)."""
  qv = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
  return _quat_mul(_quat_mul(_quat_inv(q), qv), q)[..., 1:]


def _ang_vel_body(q: np.ndarray, dt: float) -> np.ndarray:
  """Body-frame angular velocity from a quaternion sequence (wxyz)."""
  q0 = q[:-1]
  q1 = q[1:]
  dq = _quat_mul(_quat_inv(q0), q1)
  dq = np.where(dq[..., :1] < 0, -dq, dq)
  ang = 2.0 * np.arctan2(np.linalg.norm(dq[..., 1:], axis=-1), dq[..., 0])
  axis = dq[..., 1:] / np.maximum(np.linalg.norm(dq[..., 1:], axis=-1, keepdims=True), 1e-9)
  w = axis * ang[..., None] / dt
  return np.concatenate([w, w[-1:]], axis=0)


def _resample(x: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
  return np.stack([np.interp(t_out, t_in, x[:, i]) for i in range(x.shape[1])], axis=1)


def retarget_clip(
  csv_path: Path,
  robot: RobotDef,
  chain: _Chain,
  mapping: list[tuple[str, int, float, float]],
  length_scale: float,
  time_scale: float,
) -> dict[str, np.ndarray]:
  raw = np.loadtxt(csv_path, delimiter=",").astype(np.float64)
  n_in = raw.shape[0]
  t_in = np.arange(n_in) / IN_FPS * time_scale
  t_out = np.arange(0.0, t_in[-1], 1.0 / OUT_FPS)
  root_pos = _resample(raw[:, :3], t_in, t_out) * length_scale
  root_quat = _quat_normalize(_resample(_quat_wxyz_from_xyzw(raw[:, 3:7]), t_in, t_out))
  g1_q = _resample(raw[:, 7:], t_in, t_out)

  model = chain.model
  nj = len(chain.names)
  joint_pos = np.zeros((len(t_out), nj))
  idx = {n: i for i, n in enumerate(chain.names)}
  for name, col, sign, phi in mapping:
    q = sign * (g1_q[:, col] - phi)
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if model.jnt_limited[j]:
      lo, hi = model.jnt_range[j]
      q = np.clip(q, lo, hi)
    joint_pos[:, idx[name]] = q

  # FK for foot positions, and to put the lowest foot on the floor.
  data = mujoco.MjData(model)
  foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in robot.foot_sites]
  qadr = [
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in chain.names
  ]
  foot_w = np.zeros((len(t_out), len(foot_ids), 3))
  for k in range(len(t_out)):
    data.qpos[:3] = root_pos[k]
    data.qpos[3:7] = root_quat[k]
    for i, a in enumerate(qadr):
      data.qpos[a] = joint_pos[k, i]
    mujoco.mj_kinematics(model, data)
    foot_w[k] = data.site_xpos[foot_ids]
  sole = float(np.min(chain.data.site_xpos[foot_ids][:, 2]))  # site height above floor at q=0
  lowest = foot_w[:, :, 2].min(axis=1)
  shift = np.percentile(lowest, 3.0) - sole
  root_pos[:, 2] -= shift
  foot_w[:, :, 2] -= shift

  dt = 1.0 / OUT_FPS
  joint_vel = np.gradient(joint_pos, dt, axis=0)
  lin_vel_w = np.gradient(root_pos, dt, axis=0)
  lin_vel_b = _quat_rotate_inv(root_quat, lin_vel_w)
  ang_vel_b = _ang_vel_body(root_quat, dt)
  gravity_b = _quat_rotate_inv(root_quat, np.tile(np.array([0.0, 0.0, -1.0]), (len(t_out), 1)))
  foot_b = np.stack(
    [_quat_rotate_inv(root_quat, foot_w[:, i] - root_pos) for i in range(len(foot_ids))], axis=1
  )
  return {
    "root_pos": root_pos.astype(np.float32),
    "root_quat": root_quat.astype(np.float32),
    "joint_pos": joint_pos.astype(np.float32),
    "joint_vel": joint_vel.astype(np.float32),
    "root_lin_vel_b": lin_vel_b.astype(np.float32),
    "root_ang_vel_b": ang_vel_b.astype(np.float32),
    "gravity_b": gravity_b.astype(np.float32),
    "foot_pos_b": foot_b.reshape(len(t_out), -1).astype(np.float32),
    "root_height": root_pos[:, 2:3].astype(np.float32),
  }


def download_clips(patterns: tuple[str, ...] = ("walk", "run", "sprint")) -> list[Path]:
  from huggingface_hub import list_repo_files, snapshot_download

  files = list_repo_files(HF_REPO, repo_type="dataset")
  wanted = sorted(f for f in files if f.startswith("g1/") and any(p in f for p in patterns))
  local = Path(
    snapshot_download(
      HF_REPO, repo_type="dataset", allow_patterns=wanted, local_dir=DATA_DIR / "lafan1_g1"
    )
  )
  return [local / f for f in wanted]


def _lit_model(robot: RobotDef, size: int) -> mujoco.MjModel:
  """Robot MJCF plus a floor and lights, for offscreen rendering."""
  spec = mujoco.MjSpec.from_file(str(robot.xml))
  floor = spec.worldbody.add_geom()
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size[:] = [20.0, 20.0, 0.1]
  floor.rgba[:] = [0.55, 0.6, 0.68, 1.0]
  light = spec.worldbody.add_light()
  light.pos[:] = [1.0, -2.0, 3.0 * robot.standing_height + 1.0]
  light.dir[:] = [-0.3, 0.6, -1.0]
  light.castshadow = True
  spec.visual.headlight.ambient[:] = [0.45, 0.45, 0.45]
  spec.visual.headlight.diffuse[:] = [0.6, 0.6, 0.6]
  spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, size)
  spec.visual.global_.offheight = max(spec.visual.global_.offheight, size)
  return spec.compile()


def render_playback(
  robot: RobotDef,
  chain: _Chain,
  clip: dict[str, np.ndarray],
  out_png: Path,
  n_frames: int = 10,
  start_s: float | None = None,
  mp4_s: float = 0.0,
) -> None:
  """Contact sheet (and optional mp4) of the retargeted clip played back kinematically.

  Picks the window around the fastest 1.5 s of the clip unless ``start_s`` is given.
  """
  import subprocess

  from PIL import Image

  size = 400
  model = _lit_model(robot, size)
  data = mujoco.MjData(model)
  qadr = [
    model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in chain.names
  ]
  renderer = mujoco.Renderer(model, height=size, width=size)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.azimuth = 135
  cam.elevation = -12
  cam.distance = 3.0 * robot.standing_height
  T = clip["joint_pos"].shape[0]
  if start_s is None:
    speed = np.linalg.norm(clip["root_lin_vel_b"][:, :2], axis=1)
    win = int(OUT_FPS * 1.5)
    smooth = np.convolve(speed, np.ones(win) / win, mode="valid")
    k0 = int(np.argmax(smooth))
  else:
    k0 = int(start_s * OUT_FPS)

  def pose(k: int) -> None:
    data.qpos[:3] = clip["root_pos"][k]
    data.qpos[3:7] = clip["root_quat"][k]
    for i, a in enumerate(qadr):
      data.qpos[a] = clip["joint_pos"][k, i]
    mujoco.mj_kinematics(model, data)
    cam.lookat[:] = [*clip["root_pos"][k, :2], 0.5 * robot.standing_height]

  ks = np.linspace(k0, k0 + OUT_FPS * 1.5, n_frames).astype(int).clip(0, T - 1)
  tiles = []
  for k in ks:
    pose(int(k))
    renderer.update_scene(data, camera=cam)
    tiles.append(renderer.render().copy())
  cols = 5
  rows = math.ceil(len(tiles) / cols)
  sheet = Image.new("RGB", (size * cols, size * rows))
  for i, t in enumerate(tiles):
    sheet.paste(Image.fromarray(t), ((i % cols) * size, (i // cols) * size))
  out_png.parent.mkdir(parents=True, exist_ok=True)
  sheet.save(out_png)

  if mp4_s > 0:
    mp4 = out_png.with_suffix(".mp4")
    proc = subprocess.Popen(
      [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size}x{size}", "-r", str(OUT_FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(mp4),
      ],
      stdin=subprocess.PIPE,
    )  # fmt: skip
    assert proc.stdin is not None
    k1 = min(T, k0 + int(mp4_s * OUT_FPS))
    for k in range(max(0, k0 - OUT_FPS), k1):
      pose(k)
      renderer.update_scene(data, camera=cam)
      proc.stdin.write(renderer.render().tobytes())
    proc.stdin.close()
    proc.wait()
  renderer.close()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("robot", choices=sorted(ROBOTS))
  ap.add_argument("--clips", nargs="*", default=("walk", "run", "sprint"))
  ap.add_argument("--render", action="store_true", help="write a playback contact sheet")
  args = ap.parse_args()
  robot = ROBOTS[args.robot]

  chain = _Chain(mujoco.MjModel.from_xml_path(str(robot.xml)), robot.foot_sites)
  g1 = _Chain(mujoco.MjModel.from_xml_path(str(_g1_xml())))
  mapping = build_mapping(chain, g1)
  print(f"{robot.name}: mapped {len(mapping)}/{len(chain.names)} joints")
  for name, col, sign, phi in mapping:
    print(f"  {name:32s} <- g1[{col:2d}]  sign {sign:+.0f}  offset {math.degrees(phi):6.1f} deg")
  length_scale = chain.leg_length() / g1.leg_length()
  time_scale = math.sqrt(length_scale)
  print(f"  leg length ratio {length_scale:.3f}, time scale {time_scale:.3f}")

  paths = download_clips(tuple(args.clips))
  clips = []
  for p in paths:
    c = retarget_clip(p, robot, chain, mapping, length_scale, time_scale)
    c["clip"] = np.full(c["joint_pos"].shape[0], len(clips), dtype=np.int32)
    clips.append(c)
    speed = np.linalg.norm(c["root_lin_vel_b"][:, :2], axis=1)
    print(
      f"  {p.stem:22s} {c['joint_pos'].shape[0] / OUT_FPS:6.1f} s  "
      f"speed mean {speed.mean():.2f} max {np.percentile(speed, 95):.2f} m/s  "
      f"height {c['root_height'].mean():.3f} m"
    )
  out = {k: np.concatenate([c[k] for c in clips], axis=0) for k in clips[0]}
  out["joint_names"] = np.array(chain.names)
  out["clip_names"] = np.array([p.stem for p in paths])
  out["fps"] = np.array(OUT_FPS)
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(DATA_DIR / f"{robot.name}.npz", **out)
  print(f"wrote {DATA_DIR / f'{robot.name}.npz'}: {out['joint_pos'].shape[0]} frames")
  if args.render:
    for stem in ("walk1_subject5", "run1_subject2", "sprint1_subject2"):
      if stem in [p.stem for p in paths]:
        i = [p.stem for p in paths].index(stem)
        render_playback(
          robot, chain, clips[i], MEDIA_DIR / f"{robot.name}_{stem}.png", mp4_s=8.0
        )
    print(f"rendered playback sheets/mp4s to {MEDIA_DIR}")


if __name__ == "__main__":
  main()
