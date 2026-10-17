"""Turn retargeted mocap into a speed x phase gait library (``uv run gaitlib <robot>``).

Input: ``data/mocap/<robot>.npz`` from ``uv run mocap``. Output:
``data/mocap/<robot>_gaitlib.npz`` (+ ``media/mocap/<robot>_gaitlib.png``), a table the
policy tracks and the browser evaluates every control step:

  speeds[S]        forward speed of each row (m/s)
  period[S]        stride period (s), left touchdown to left touchdown
  duty[S]          stance fraction of each foot
  q[S, P, J]       joint angles over one stride, phase 0 = left touchdown, right
                   touchdown at exactly 0.5 (model joint order)
  grav[S, P, 2]    projected-gravity x/y of the base (lean and side sway)
  stand_q[J]       standing pose (the robot's default pose)

Strides are cut at left-foot touchdowns, kept if straight and forward, time-warped so
both feet's touchdown / lift-off land on common phases, then kernel-averaged per speed
and mirror-symmetrised (left at phase p == mirrored right at p + 0.5). The table is
therefore an averaged, symmetric human stride at each speed: no single actor's quirks,
but every actor's arm swing, knee flexion, pelvic list and lean.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from chris_robots.robots import ROBOTS_BY_NAME, RobotDef
from chris_robots.symmetry import _joint_mirror
from chris_robots.tools import mocap

N_PHASE = 40


def _yaw(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """R(q) v for wxyz quaternions q [N,4] and vectors v [N,3]."""
  w = q[:, :1]
  u = q[:, 1:]
  t = 2.0 * np.cross(u, v)
  return v + w * t + np.cross(u, t)


def _contacts(z: np.ndarray, v: np.ndarray, z_thr: float, v_thr: float) -> np.ndarray:
  c = (z < z_thr) & (v < v_thr)
  # Remove flicker: runs shorter than 3 frames flip to their neighbours' state.
  out = c.copy()
  n = len(c)
  i = 0
  while i < n:
    j = i
    while j < n and c[j] == c[i]:
      j += 1
    if j - i < 3 and i > 0 and j < n:
      out[i:j] = not c[i]
    i = j
  return out


def _events(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Indices of touchdowns (False->True) and lift-offs (True->False)."""
  d = np.diff(c.astype(np.int8))
  return np.nonzero(d == 1)[0] + 1, np.nonzero(d == -1)[0] + 1


def _warp(x: np.ndarray, src: list[float], dst: list[float]) -> np.ndarray:
  """Resample a stride [T, D] (time normalised to [0,1]) so event phases ``src``
  move to ``dst`` (both increasing, starting at 0 and ending at 1)."""
  t = np.linspace(0.0, 1.0, x.shape[0])
  phase = np.linspace(0.0, 1.0, N_PHASE, endpoint=False)
  src_at = np.interp(phase, dst, src)  # where in the original stride each phase comes from
  return np.stack([np.interp(src_at, t, x[:, i]) for i in range(x.shape[1])], axis=1)


def extract_strides(robot: RobotDef, data: dict) -> list[dict]:
  fps = float(data["fps"])
  root = data["root_pos"].astype(np.float64)
  quat = data["root_quat"].astype(np.float64)
  q = data["joint_pos"].astype(np.float64)
  grav = data["gravity_b"].astype(np.float64)
  feet_b = data["foot_pos_b"].astype(np.float64).reshape(len(root), 2, 3)
  clip = data["clip"]
  left = robot.foot_order.index("left")
  feet_w = np.stack([root + _rotate(quat, feet_b[:, i]) for i in range(2)], axis=1)
  h = robot.standing_height
  sole = np.percentile(feet_w[:, :, 2], 3.0)
  z_thr = sole + 0.035 * h
  # Retargeted stance feet slide a little (joint-space retargeting does not pin them),
  # so the speed gate is loose: it only rejects a fast swing foot skimming the floor.
  v_thr = 2.5 * h
  strides = []
  for c in np.unique(clip):
    idx = np.nonzero(clip == c)[0]
    fw = feet_w[idx]
    vel = np.linalg.norm(np.gradient(fw[:, :, :2], 1.0 / fps, axis=0), axis=2)
    con = [_contacts(fw[:, i, 2], vel[:, i], z_thr, v_thr) for i in range(2)]
    td_l, lo_l = _events(con[left])
    td_r, lo_r = _events(con[1 - left])
    for a, b in zip(td_l[:-1], td_l[1:], strict=True):
      T = (b - a) / fps
      if not (0.25 * math.sqrt(h) < T < 2.0 * math.sqrt(h)):
        continue
      r_td = td_r[(td_r > a) & (td_r < b)]
      l_lo = lo_l[(lo_l > a) & (lo_l < b)]
      r_lo = lo_r[(lo_r > a) & (lo_r < b)]
      if len(r_td) != 1 or len(l_lo) != 1 or len(r_lo) != 1:
        continue
      p_rtd, p_llo, p_rlo = ((float(k[0]) - a) / (b - a) for k in (r_td, l_lo, r_lo))
      duty_l = p_llo
      duty_r = (p_rlo - p_rtd) % 1.0
      sl = slice(idx[a], idx[b] + 1)
      yaw = np.unwrap(_yaw(quat[sl]))
      heading = yaw.mean()
      disp = root[idx[b], :2] - root[idx[a], :2]
      ch, sh = math.cos(heading), math.sin(heading)
      vx = (ch * disp[0] + sh * disp[1]) / T
      vy = (-sh * disp[0] + ch * disp[1]) / T
      turn = (yaw[-1] - yaw[0]) / T
      strides.append(
        {
          "T": T,
          "vx": vx,
          "vy": vy,
          "turn": turn,
          "duty": 0.5 * (duty_l + duty_r),
          "events": (p_rtd, p_llo, p_rlo),
          "q": q[sl],
          "grav": grav[sl, :2],
          # pelvis yaw about the stride's mean heading, turn trend removed
          "yaw": (
            yaw - np.linspace(yaw[0], yaw[-1], len(yaw)) - yaw.mean() + 0.5 * (yaw[0] + yaw[-1])
          )[:, None],
          "clip": int(c),
        }
      )
  return strides


def build_table(robot: RobotDef, strides: list[dict]) -> dict:
  keep = [
    s
    for s in strides
    if s["vx"] > 0.12 * math.sqrt(robot.standing_height)
    and abs(s["turn"]) < 0.4
    and abs(s["vy"]) < 0.15 * s["vx"] + 0.05
  ]
  warped = []
  for s in keep:
    d = float(np.clip(s["duty"], 0.2, 0.75))
    p_rtd, p_llo, p_rlo = s["events"]
    # Event phases -> targets: right touchdown at 0.5, lift-offs at duty after each
    # touchdown. Strides whose event order disagrees with the target order are odd
    # shuffles and are dropped.
    ev = sorted([(0.5, p_rtd), (d, p_llo), ((0.5 + d) % 1.0, p_rlo)])
    dst = [0.0] + [e[0] for e in ev] + [1.0]
    src = [0.0] + [e[1] for e in ev] + [1.0]
    if np.any(np.diff(src) <= 0):
      continue
    g = np.concatenate([s["grav"], s["yaw"] - s["yaw"].mean()], axis=1)
    warped.append((s["vx"], s["T"], d, _warp(s["q"], src, dst), _warp(g, src, dst)))
  vx = np.array([w[0] for w in warped])
  T = np.array([w[1] for w in warped])
  duty = np.array([w[2] for w in warped])
  Q = np.stack([w[3] for w in warped])
  G = np.stack([w[4] for w in warped])
  print(
    f"  {len(strides)} strides, {len(warped)} straight+forward, speed {vx.min():.2f}..{vx.max():.2f}"
  )

  # Walking (double support) and running strides are averaged separately: a blend of
  # the two is neither. Each gets its own speed grid; the table interpolates between
  # the fastest walk row and the slowest run row.
  rows = []
  v_cap = 1.1 * robot.max_lin_vel
  for is_run, n_rows in ((False, 4), (True, 5)):
    m = (duty < 0.47) == is_run
    lo, hi = np.percentile(vx[m], 8), min(np.percentile(vx[m], 97), v_cap)
    if hi <= lo:
      continue
    grid = np.linspace(lo, hi, n_rows)
    bw = 0.6 * (grid[1] - grid[0])
    for sp in grid:
      w = np.where(m, np.exp(-0.5 * ((vx - sp) / bw) ** 2), 0.0)
      if w.sum() < 3.0:
        continue
      w = w / w.sum()
      rows.append(
        {
          "speed": float((w * vx).sum()),
          "period": float((w * T).sum()),
          "duty": float((w * duty).sum()),
          "q": np.einsum("n,npj->pj", w, Q),
          "g": np.einsum("n,npk->pk", w, G),
          "n_eff": float(1.0 / (w**2).sum()),
        }
      )
  rows.sort(key=lambda r: r["speed"])
  return {
    "speeds": np.array([r["speed"] for r in rows]),
    "period": np.array([r["period"] for r in rows]),
    "duty": np.array([r["duty"] for r in rows]),
    "q": np.stack([r["q"] for r in rows]),
    "grav": np.stack([r["g"][:, :2] for r in rows]),
    "yaw": np.stack([r["g"][:, 2] for r in rows]),
    "n_eff": np.array([r["n_eff"] for r in rows]),
  }


def sample(table: dict, speed: float, phase: float) -> tuple[np.ndarray, np.ndarray, float, float]:
  """Reference (q, grav_xy, period, duty) at a forward speed and stride phase.

  The same lookup runs in the training command term and in the browser: linear in
  speed (clamped to the table), linear and periodic in phase, and blended towards the
  standing pose below the slowest row.
  """
  sp = table["speeds"]
  s = float(np.clip(speed, sp[0], sp[-1]))
  i = int(np.clip(np.searchsorted(sp, s) - 1, 0, len(sp) - 2))
  a = (s - sp[i]) / (sp[i + 1] - sp[i])
  x = (phase % 1.0) * N_PHASE
  k = int(x) % N_PHASE
  b = x - int(x)

  def at(key: str) -> np.ndarray:
    t = table[key]
    row = lambda r: (1 - b) * t[r, k] + b * t[r, (k + 1) % N_PHASE]  # noqa: E731
    return (1 - a) * row(i) + a * row(i + 1)

  q, g = at("q"), at("grav")
  period = (1 - a) * table["period"][i] + a * table["period"][i + 1]
  duty = (1 - a) * table["duty"][i] + a * table["duty"][i + 1]
  w = _stand_blend(speed, sp[0])
  q = table["stand_q"] + w * (q - table["stand_q"])
  return q, w * g, float(period), float(duty)


def _stand_blend(speed: float, slowest: float) -> float:
  u = float(np.clip((speed - 0.25 * slowest) / (0.75 * slowest), 0.0, 1.0))
  return u * u * (3 - 2 * u)


def symmetrize(table: dict, perm: np.ndarray, sign: np.ndarray) -> None:
  half = N_PHASE // 2
  q = table["q"]
  qm = np.roll(q, -half, axis=1)[..., perm] * sign
  table["q"] = 0.5 * (q + qm)
  g = table["grav"]
  gm = np.roll(g, -half, axis=1) * np.array([1.0, -1.0])
  table["grav"] = 0.5 * (g + gm)
  y = table["yaw"]
  table["yaw"] = 0.5 * (y - np.roll(y, -half, axis=1))
  # light circular smoothing over phase (averaging already removes most jitter)
  k = np.array([0.25, 0.5, 0.25])
  for key in ("q", "grav", "yaw"):
    x = table[key]
    table[key] = k[0] * np.roll(x, 1, axis=1) + k[1] * x + k[2] * np.roll(x, -1, axis=1)


def plot(robot: RobotDef, table: dict, names: list[str], out) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  pick = [
    i
    for i, n in enumerate(names)
    if "left" in n
    and any(k in n for k in ("hip_pitch", "knee", "shoulder_pitch", "elbow", "hip_roll"))
  ]
  if not any("ankle" in names[i] for i in pick):
    pick += [i for i, n in enumerate(names) if "left" in n and "ankle" in n][:1]
  fig, axes = plt.subplots(1, len(pick) + 2, figsize=(3.2 * (len(pick) + 2), 3.0))
  ph = np.linspace(0, 1, N_PHASE, endpoint=False)
  cmap = plt.get_cmap("viridis")
  S = len(table["speeds"])
  for ax, j in zip(axes[:-2], pick, strict=False):
    for s in range(S):
      ax.plot(ph, table["q"][s, :, j], color=cmap(s / max(1, S - 1)))
    ax.set_title(names[j], fontsize=8)
  for s in range(S):
    c = cmap(s / max(1, S - 1))
    axes[-2].plot(ph, np.degrees(np.arcsin(np.clip(table["grav"][s, :, 1], -1, 1))), color=c)
    axes[-1].plot(ph, np.degrees(table["yaw"][s]), color=c)
  axes[-2].set_title("base roll (deg)", fontsize=8)
  axes[-1].set_title("pelvis yaw (deg)", fontsize=8)
  fig.suptitle(
    f"{robot.display_name} gait library: "
    + ", ".join(f"{v:.2f}" for v in table["speeds"])
    + " m/s",
    fontsize=9,
  )
  fig.tight_layout()
  fig.savefig(out, dpi=90)
  plt.close(fig)


def render_ramp(robot: RobotDef, table: dict, names: list[str], seconds: float = 14.0) -> None:
  """Kinematic playback of the table while the speed ramps 0 -> max -> 0."""
  model = mujoco.MjModel.from_xml_path(str(robot.xml))
  data = mujoco.MjData(model)
  qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
  sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in robot.foot_sites]
  fps = mocap.OUT_FPS
  n = int(seconds * fps)
  v_max = table["speeds"][-1]
  phase, x = 0.0, 0.0
  out = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "root_lin_vel_b")}
  for k in range(n):
    u = k / (n - 1)
    v = v_max * (1.0 - abs(2.0 * u - 1.0)) ** 1.5
    q, g, period, _ = sample(table, v, phase)
    if v > 0.25 * table["speeds"][0]:
      phase = (phase + 1.0 / (fps * period)) % 1.0
    pitch = math.asin(float(np.clip(g[0], -1, 1)))
    roll = math.asin(float(np.clip(-g[1] / math.cos(pitch), -1, 1)))
    cr, sr, cp, sp_ = (
      math.cos(roll / 2),
      math.sin(roll / 2),
      math.cos(pitch / 2),
      math.sin(pitch / 2),
    )
    quat = np.array([cr * cp, sr * cp, cr * sp_, -sr * sp_])
    data.qpos[:3] = [0.0, 0.0, 0.0]
    data.qpos[3:7] = quat
    data.qpos[qadr] = q
    mujoco.mj_kinematics(model, data)
    z = table["sole"] - min(data.site_xpos[s][2] for s in sites)
    x += v / fps
    out["root_pos"].append([x, 0.0, z])
    out["root_quat"].append(quat)
    out["joint_pos"].append(q)
    out["root_lin_vel_b"].append([v, 0.0, 0.0])
  clip = {k: np.array(v) for k, v in out.items()}
  chain = mocap._Chain(model, robot.foot_sites)
  png = mocap.MEDIA_DIR / f"{robot.name}_gaitlib_ramp.png"
  mocap.render_playback(robot, chain, clip, png, start_s=0.0, mp4_s=seconds)
  print(f"rendered {png.with_suffix('.mp4')}")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("robot", choices=sorted(ROBOTS_BY_NAME))
  ap.add_argument("--render", action="store_true", help="kinematic speed-ramp playback mp4")
  args = ap.parse_args()
  robot = ROBOTS_BY_NAME[args.robot]
  data = dict(np.load(mocap.DATA_DIR / f"{robot.name}.npz"))
  names = [str(n) for n in data["joint_names"]]

  model = mujoco.MjModel.from_xml_path(str(robot.xml))
  chain = mocap._Chain(model, robot.foot_sites)
  g1 = mocap._Chain(mujoco.MjModel.from_xml_path(str(mocap._g1_xml())))
  mapped = {name for name, *_ in mocap.build_mapping(chain, g1)}
  init = robot.info["init_pose"]["joint_pos"]
  stand_q = np.array([float(init.get(n, 0.0)) for n in names])
  # Joints with no human counterpart follow the standing pose.
  for j, n in enumerate(names):
    if n not in mapped:
      data["joint_pos"][:, j] = stand_q[j]

  strides = extract_strides(robot, data)
  table = build_table(robot, strides)
  perm, sign = _joint_mirror(model, tuple(names))
  symmetrize(table, perm, sign)
  table["stand_q"] = stand_q
  feet_w = [
    data["root_pos"][:, 2]
    + _rotate(data["root_quat"], data["foot_pos_b"][:, 3 * i : 3 * i + 3])[:, 2]
    for i in range(2)
  ]
  table["sole"] = np.array(np.percentile(np.minimum(*feet_w), 3.0))
  table["joint_names"] = np.array(names)
  table["mapped"] = np.array([n in mapped for n in names])
  out = mocap.DATA_DIR / f"{robot.name}_gaitlib.npz"
  np.savez(out, **table)
  for s, T, d, n in zip(
    table["speeds"], table["period"], table["duty"], table["n_eff"], strict=True
  ):
    print(f"    {s:5.2f} m/s  period {T:.2f} s  duty {d:.2f}  ({n:.0f} strides)")
  mocap.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
  plot(robot, table, names, mocap.MEDIA_DIR / f"{robot.name}_gaitlib.png")
  print(f"wrote {out}")
  if args.render:
    render_ramp(robot, table, names)


if __name__ == "__main__":
  main()
