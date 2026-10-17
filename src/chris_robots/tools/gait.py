"""Gait analysis for bundle rollouts: is it *walking*, or exploiting the simulator?

A policy can score well on velocity tracking while doing something that is not a
gait at all (one foot planted, the other hovering, propelling itself by chattering
on a foot corner). Single frames do not reveal this; the contact timeline does.

``record()`` rolls a BundleRunner out under a fixed command and logs, per control
step, foot contact state (from MuJoCo's contact list), number of contact points per
foot, sole tilt, foot position/velocity and a few joints. ``analyze()`` turns that
into step counts, duty factors, alternation, stride length, swing height, stance
tilt, corner-contact fraction, left/right symmetry and a verdict. ``diagram()``
draws the classic gait diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chris_robots.tools.export_web import BundleRunner

FONT = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
MIN_SWING_S = 0.08  # shorter airborne phases are chatter, not steps
MIN_STANCE_S = 0.06


@dataclass
class GaitTrace:
  dt: float
  t: np.ndarray  # [T]
  contact: np.ndarray  # [T, 2] bool (left, right)
  ncon: np.ndarray  # [T, 2] contact points per foot
  tilt_deg: np.ndarray  # [T, 2] sole tilt from horizontal
  foot_pos: np.ndarray  # [T, 2, 3] sole site position (world)
  foot_vel_xy: np.ndarray  # [T, 2]
  base_pos: np.ndarray  # [T, 3]
  base_vel_b: np.ndarray  # [T, 3]
  hip_pitch: np.ndarray  # [T, 2] (left, right), sign-normalised so + = thigh forward
  knee: np.ndarray  # [T, 2] |flexion|
  action_rate: np.ndarray  # [T] mean |a_t - a_{t-1}|
  fell_at: float | None = None
  floor_z: float = 0.0


@dataclass
class GaitMetrics:
  seconds: float
  steps: tuple[int, int]  # proper steps per foot (swing >= MIN_SWING_S)
  chatter: tuple[int, int]  # contact toggles shorter than the minimum phase
  duty: tuple[float, float]  # fraction of time in contact
  double_support: float
  single_support: float
  flight: float
  alternation: float  # fraction of consecutive touchdowns that alternate L/R
  stride_m: tuple[float, float]  # mean touchdown-to-touchdown distance per foot
  swing_height_m: tuple[float, float]  # mean peak sole height per swing
  stance_tilt_deg: tuple[float, float]  # mean sole tilt while in contact
  corner_frac: tuple[float, float]  # stance time with <= 1 contact point
  slip_mps: tuple[float, float]  # mean horizontal sole speed while in contact
  period_s: float | None  # gait period from hip-pitch autocorrelation
  symmetry: float | None  # L/R hip-pitch correlation at half-period lag (1 = perfect)
  action_rate: float
  mean_speed: float
  verdict: str
  reasons: list[str] = field(default_factory=list)

  def line(self) -> str:
    s = self
    per = f"{s.period_s:.2f}s" if s.period_s else "-"
    sym = f"{s.symmetry:+.2f}" if s.symmetry is not None else "-"
    return (
      f"{s.verdict:8s} steps L/R {s.steps[0]}/{s.steps[1]}  chatter {s.chatter[0]}/{s.chatter[1]}"
      f"  duty {s.duty[0]:.2f}/{s.duty[1]:.2f}  dbl {s.double_support:.2f} flight {s.flight:.2f}"
      f"  alt {s.alternation:.2f}  stride {s.stride_m[0]:.2f}/{s.stride_m[1]:.2f} m"
      f"  swing_h {s.swing_height_m[0]:.3f}/{s.swing_height_m[1]:.3f}"
      f"  tilt {s.stance_tilt_deg[0]:.0f}/{s.stance_tilt_deg[1]:.0f} deg"
      f"  corner {s.corner_frac[0]:.2f}/{s.corner_frac[1]:.2f}  slip {s.slip_mps[0]:.2f}/{s.slip_mps[1]:.2f}"
      f"  T {per}  sym {sym}  act_rate {s.action_rate:.2f}"
    )


# ------------------------------------------------------------------------------ recording


def _find_joint(names: list[str], side: str, part: str) -> int | None:
  for i, n in enumerate(names):
    if side in n and part in n:
      return i
  return None


def record(
  runner: BundleRunner,
  command: np.ndarray,
  seconds: float,
  foot_order: tuple[str, ...],
  joint_axis_y: dict[str, float] | None = None,
  on_step=None,
) -> GaitTrace:
  """Roll out ``seconds`` under ``command`` (runner state is continued, not reset)."""
  m, d = runner.model, runner.data
  dt = runner.cfg["sim"]["control_dt"]
  n = int(round(seconds / dt))
  floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
  sides = ("left", "right")
  foot_geom = {
    s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_collision") for s in sides
  }
  foot_site = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot") for s in sides}
  names = runner.cfg["joints"]["names"]
  hip_idx = [_find_joint(names, s, "hip_pitch") for s in sides]
  knee_idx = [_find_joint(names, s, "knee") for s in sides]
  # + = thigh forward. A pitch joint about +y moves a hanging limb backwards for +q.
  hip_sign = [
    -np.sign(joint_axis_y.get(names[i], 1.0)) if (joint_axis_y and i is not None) else 1.0
    for i in hip_idx
  ]
  del foot_order  # analysis is always reported as (left, right)
  # Sole normal = the box axis that points up in the current (standing-ish) pose.
  up_axis = {}
  for s in sides:
    R = d.geom_xmat[foot_geom[s]].reshape(3, 3)
    up_axis[s] = int(np.argmax(np.abs(R[2, :])))

  T = np.zeros(n)
  contact = np.zeros((n, 2), bool)
  ncon = np.zeros((n, 2), int)
  tilt = np.zeros((n, 2))
  fpos = np.zeros((n, 2, 3))
  fvel = np.zeros((n, 2))
  bpos = np.zeros((n, 3))
  bvel = np.zeros((n, 3))
  hip = np.zeros((n, 2))
  knee = np.zeros((n, 2))
  arate = np.zeros(n)
  fell_at = None
  prev_action = runner.last_action.copy()
  t0 = d.time
  for k in range(n):
    runner.step(command.astype(np.float32))
    T[k] = d.time - t0
    arate[k] = float(np.mean(np.abs(runner.last_action - prev_action)))
    prev_action = runner.last_action.copy()
    for c in range(d.ncon):
      g1, g2 = d.contact[c].geom1, d.contact[c].geom2
      for j, s in enumerate(sides):
        if (g1 == floor and g2 == foot_geom[s]) or (g2 == floor and g1 == foot_geom[s]):
          ncon[k, j] += 1
    contact[k] = ncon[k] > 0
    for j, s in enumerate(sides):
      R = d.geom_xmat[foot_geom[s]].reshape(3, 3)
      tilt[k, j] = np.degrees(np.arccos(np.clip(abs(R[2, up_axis[s]]), -1, 1)))
      fpos[k, j] = d.site_xpos[foot_site[s]]
      # Sole-site horizontal speed from finite differences (next loop fills k-1).
    if k > 0:
      fvel[k] = np.linalg.norm((fpos[k, :, :2] - fpos[k - 1, :, :2]) / dt, axis=1)
    bpos[k] = d.xpos[runner.base_id]
    bvel[k] = d.sensordata[runner.velo]
    q = d.qpos[runner.qpos_adr] - runner.default
    for j in range(2):
      if hip_idx[j] is not None:
        hip[k, j] = hip_sign[j] * q[hip_idx[j]]
      if knee_idx[j] is not None:
        knee[k, j] = abs(d.qpos[runner.qpos_adr][knee_idx[j]])
    if fell_at is None and (
      runner.upright() < 0.5 or bpos[k, 2] < 0.4 * runner.cfg["sim"]["standing_base_height_m"]
    ):
      fell_at = T[k]
    if on_step is not None:
      on_step(k)
  return GaitTrace(
    dt=dt,
    t=T,
    contact=contact,
    ncon=ncon,
    tilt_deg=tilt,
    foot_pos=fpos,
    foot_vel_xy=fvel,
    base_pos=bpos,
    base_vel_b=bvel,
    hip_pitch=hip,
    knee=knee,
    action_rate=arate,
    fell_at=fell_at,
    floor_z=0.0,
  )


# ------------------------------------------------------------------------------ analysis


def _phases(contact_1d: np.ndarray, dt: float) -> list[tuple[bool, int, int]]:
  """Run-length encode a contact signal -> [(in_contact, start, end_exclusive)]."""
  out = []
  start = 0
  for i in range(1, len(contact_1d) + 1):
    if i == len(contact_1d) or contact_1d[i] != contact_1d[start]:
      out.append((bool(contact_1d[start]), start, i))
      start = i
  del dt
  return out


def _autocorr_period(x: np.ndarray, dt: float, tmin: float, tmax: float) -> float | None:
  x = x - x.mean()
  if np.std(x) < 1e-4 or len(x) < int(tmax / dt) + int(tmin / dt):
    return None
  # Unbiased autocorrelation (normalise by overlap length) so long lags aren't penalised.
  ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
  ac = ac / (np.arange(len(x), 0, -1) * (x.var() + 1e-9))
  lo, hi = int(tmin / dt), min(int(tmax / dt), len(ac) - 1)
  if hi <= lo:
    return None
  seg = ac[lo:hi]
  # First local maximum above threshold = fundamental period (argmax could pick 2T).
  for i in range(1, len(seg) - 1):
    if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.3:
      return (lo + i) * dt
  return None


def analyze(trace: GaitTrace, swing_target_m: float, speed_cmd: float) -> GaitMetrics:
  dt = trace.dt
  n = len(trace.t)
  seconds = n * dt
  steps, chatter, strides, swing_h, tilts, corner, slip = [], [], [], [], [], [], []
  touchdowns: list[tuple[float, int]] = []
  for j in range(2):
    c = trace.contact[:, j]
    ph = _phases(c, dt)
    n_steps = n_chat = 0
    st, sh = [], []
    last_td_pos = None
    for idx, (inc, a, b) in enumerate(ph):
      dur = (b - a) * dt
      if inc:
        if dur < MIN_STANCE_S and 0 < idx < len(ph) - 1:
          n_chat += 1
      else:
        if dur < MIN_SWING_S:
          if 0 < idx < len(ph) - 1:
            n_chat += 1
          continue
        if idx == len(ph) - 1:
          continue  # still airborne at the end: not a completed step
        n_steps += 1
        sh.append(trace.foot_pos[a:b, j, 2].max() - trace.floor_z)
        td = b  # touchdown index
        touchdowns.append((trace.t[td], j))
        pos = trace.foot_pos[td, j, :2]
        if last_td_pos is not None:
          st.append(float(np.linalg.norm(pos - last_td_pos)))
        last_td_pos = pos
      if inc and idx == 0:
        last_td_pos = trace.foot_pos[a, j, :2]
    steps.append(n_steps)
    chatter.append(n_chat)
    strides.append(float(np.mean(st)) if st else 0.0)
    swing_h.append(float(np.mean(sh)) if sh else 0.0)
    inc_mask = c
    tilts.append(float(trace.tilt_deg[inc_mask, j].mean()) if inc_mask.any() else 0.0)
    corner.append(float((trace.ncon[inc_mask, j] <= 1).mean()) if inc_mask.any() else 0.0)
    slip.append(float(trace.foot_vel_xy[inc_mask, j].mean()) if inc_mask.any() else 0.0)

  both = trace.contact.all(axis=1)
  none = ~trace.contact.any(axis=1)
  single = trace.contact.sum(axis=1) == 1
  touchdowns.sort()
  if len(touchdowns) >= 2:
    alt = float(
      np.mean([a[1] != b[1] for a, b in zip(touchdowns[:-1], touchdowns[1:], strict=False)])
    )
  else:
    alt = 0.0

  period = _autocorr_period(trace.hip_pitch[:, 0], dt, 0.25, 1.6)
  sym = None
  if period is not None:
    half = int(round(period / 2 / dt))
    L = trace.hip_pitch[:, 0] - trace.hip_pitch[:, 0].mean()
    R = trace.hip_pitch[:, 1] - trace.hip_pitch[:, 1].mean()
    if half < len(L) - 10 and np.std(L) > 1e-4 and np.std(R) > 1e-4:
      a, b = L[half:], R[: len(R) - half]
      sym = float(np.corrcoef(a, b)[0, 1])

  duty = (float(trace.contact[:, 0].mean()), float(trace.contact[:, 1].mean()))
  mean_speed = float(np.linalg.norm(trace.base_vel_b[:, :2].mean(0)))

  reasons = []
  min_steps = max(2, int(seconds * 0.8))  # a real gait makes >= ~1 step/s/foot
  if trace.fell_at is not None:
    reasons.append(f"fell at {trace.fell_at:.1f}s")
  if speed_cmd > 0.05:
    if min(steps) < min_steps:
      reasons.append(f"too few steps ({steps[0]}/{steps[1]} < {min_steps})")
    if alt < 0.75:
      reasons.append(f"feet not alternating (alt {alt:.2f})")
    if max(chatter) > max(2, 1.0 * seconds):
      reasons.append(f"contact chatter ({chatter[0]}/{chatter[1]})")
    if min(swing_h) < 0.3 * swing_target_m:
      reasons.append(f"swing too low ({min(swing_h):.3f} m)")
    if max(tilts) > 12.0:
      reasons.append(f"stance on tilted sole ({max(tilts):.0f} deg)")
    if max(corner) > 0.4:
      reasons.append(f"corner contacts ({max(corner):.2f})")
    if strides[0] > 0 and strides[1] > 0:
      ratio = strides[0] / strides[1]
      if ratio < 0.7 or ratio > 1.4:
        reasons.append(f"asymmetric strides ({ratio:.2f})")
    if sym is not None and sym < 0.5:
      reasons.append(f"L/R hip trajectories dissimilar (sym {sym:.2f})")
  if trace.fell_at is not None:
    verdict = "FELL"
  elif speed_cmd <= 0.05:
    verdict = "STAND" if min(steps) == 0 and max(trace.foot_vel_xy.mean(0)) < 0.05 else "FIDGET"
  else:
    verdict = "WALK" if not reasons else "GLITCH"
  return GaitMetrics(
    seconds=seconds,
    steps=(steps[0], steps[1]),
    chatter=(chatter[0], chatter[1]),
    duty=duty,
    double_support=float(both.mean()),
    single_support=float(single.mean()),
    flight=float(none.mean()),
    alternation=alt,
    stride_m=(strides[0], strides[1]),
    swing_height_m=(swing_h[0], swing_h[1]),
    stance_tilt_deg=(tilts[0], tilts[1]),
    corner_frac=(corner[0], corner[1]),
    slip_mps=(slip[0], slip[1]),
    period_s=period,
    symmetry=sym,
    action_rate=float(trace.action_rate.mean()),
    mean_speed=mean_speed,
    verdict=verdict,
    reasons=reasons,
  )


# ------------------------------------------------------------------------------ diagram


def diagram(
  traces: list[tuple[str, GaitTrace, GaitMetrics]], title: str, width: int = 1200
) -> Image.Image:
  """Gait diagram per segment: contact bars (L/R), sole height, hip pitch."""
  font = ImageFont.truetype(FONT, 15)
  font_small = ImageFont.truetype(FONT, 12)
  row_h = 150
  pad_l = 70
  img = Image.new("RGB", (width, 30 + row_h * len(traces)), (18, 18, 22))
  dr = ImageDraw.Draw(img)
  dr.text((10, 7), title, font=font, fill=(255, 255, 255))
  for r, (label, tr, met) in enumerate(traces):
    y0 = 30 + r * row_h
    n = len(tr.t)

    def x_of(k: int, n: int = n) -> int:
      return pad_l + int((width - pad_l - 10) * k / max(1, n - 1))

    color = (90, 220, 120) if met.verdict == "WALK" else (240, 90, 90)
    if met.verdict in ("STAND",):
      color = (160, 160, 160)
    dr.text((10, y0 + 4), label, font=font_small, fill=(230, 230, 230))
    dr.text((pad_l, y0 + 4), met.verdict, font=font_small, fill=color)
    dr.text(
      (pad_l + 70, y0 + 4), "; ".join(met.reasons)[:150], font=font_small, fill=(200, 200, 200)
    )
    # Contact bars.
    for j, side in enumerate(("L", "R")):
      yb = y0 + 26 + j * 16
      dr.text((pad_l - 18, yb), side, font=font_small, fill=(200, 200, 200))
      for k in range(n):
        if tr.contact[k, j]:
          bad = tr.ncon[k, j] <= 1 or tr.tilt_deg[k, j] > 12
          dr.line(
            [(x_of(k), yb + 2), (x_of(k), yb + 12)], fill=(230, 80, 80) if bad else (120, 200, 255)
          )
    # Sole height curves.
    yh0, hh = y0 + 62, 36
    hmax = max(0.02, float(tr.foot_pos[:, :, 2].max() - tr.floor_z))
    dr.text((pad_l - 60, yh0 + 10), f"h {hmax:.2f}", font=font_small, fill=(200, 200, 200))
    for j, col in enumerate(((120, 200, 255), (255, 170, 90))):
      pts = [
        (x_of(k), yh0 + hh - int(hh * (tr.foot_pos[k, j, 2] - tr.floor_z) / hmax)) for k in range(n)
      ]
      dr.line(pts, fill=col, width=1)
    # Hip pitch curves (L blue, R orange): symmetric gait = same shape, half-period shift.
    yq0, hq = y0 + 104, 36
    q = tr.hip_pitch
    qmax = max(0.05, float(np.abs(q - q.mean(0)).max()))
    dr.text((pad_l - 60, yq0 + 10), "hip", font=font_small, fill=(200, 200, 200))
    dr.line([(pad_l, yq0 + hq // 2), (width - 10, yq0 + hq // 2)], fill=(60, 60, 70))
    for j, col in enumerate(((120, 200, 255), (255, 170, 90))):
      pts = [
        (x_of(k), yq0 + hq // 2 - int((hq // 2) * (q[k, j] - q[:, j].mean()) / qmax))
        for k in range(n)
      ]
      dr.line(pts, fill=col, width=1)
  return img


def joint_axis_y_from_info(info: dict) -> dict[str, float]:
  return {n: v["world_axis_at_init"][1] for n, v in info["joint_info"].items()}


__all__ = [
  "GaitMetrics",
  "GaitTrace",
  "analyze",
  "diagram",
  "joint_axis_y_from_info",
  "record",
]
