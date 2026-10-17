"""Left/right mirror symmetry for rsl-rl's ``Symmetry`` extension.

A walking gait is (ideally) invariant under reflection through the robot's sagittal
plane: swap left and right limbs, flip lateral quantities. rsl-rl can exploit that
prior in two ways (Mittal et al. 2024): *data augmentation* (every mini-batch gets the
mirrored copy of each sample appended) and a *mirror loss* (the actor is penalised when
``pi(mirror(obs)) != mirror(pi(obs))``).

Both need one function, :func:`mirror`, that maps observations and actions to their
mirrored counterparts. Rather than hand-write per-robot joint tables we derive the
mapping from the MuJoCo model: left/right joints are paired by name, and the sign of
each pair is read off the joint axes (a rotation about axis ``a`` reflects to a rotation
about ``M a`` with negated angle, ``M = diag(1, -1, 1)``). Observation terms are mirrored
according to their physical meaning (looked up by term name in the observation manager).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

_MIRROR = np.diag([1.0, -1.0, 1.0])

# Per-term mirror rule for fixed-size vector terms: (sign pattern). Lateral (y) components
# and rotations about x/z flip under a sagittal reflection.
_VECTOR_RULES: dict[str, tuple[float, ...]] = {
  "base_lin_vel": (1.0, -1.0, 1.0),
  "base_ang_vel": (-1.0, 1.0, -1.0),
  "projected_gravity": (1.0, -1.0, 1.0),
  "command": (1.0, -1.0, -1.0),  # (vx, vy, wz)
  "gait_clock": (-1.0, -1.0),  # phase + 0.5 -> (sin, cos) negate
}
# Per-foot terms (2 feet, k values each): swap feet, optional per-value signs.
_FOOT_RULES: dict[str, tuple[float, ...]] = {
  "foot_height": (1.0,),
  "foot_air_time": (1.0,),
  "foot_contact": (1.0,),
  "foot_contact_forces": (1.0, -1.0, 1.0),
}


def _swap_side(name: str) -> str:
  for a, b in (("left", "right"), ("Left", "Right"), ("_l_", "_r_"), ("L_", "R_")):
    if a in name:
      return name.replace(a, b)
    if b in name:
      return name.replace(b, a)
  return name


@dataclass
class _Affine:
  perm: torch.Tensor
  sign: torch.Tensor
  offset: torch.Tensor

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return x[..., self.perm] * self.sign + self.offset


@dataclass
class MirrorSpec:
  groups: dict[str, _Affine]
  actions: _Affine
  joint_pairs: list[tuple[str, str, float]]


def _joint_mirror(
  model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
  """Return (perm, sign) such that mirrored_q[i] = sign[i] * q[perm[i]]."""
  data = mujoco.MjData(model)
  if model.nkey > 0:
    data.qpos[:] = model.key_qpos[0]
  mujoco.mj_forward(model, data)

  def jid(name: str) -> int:
    for cand in (name, f"robot/{name}"):
      i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cand)
      if i >= 0:
        return i
    raise KeyError(name)

  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
  R_base = data.xmat[model.jnt_bodyid[free[0]]].reshape(3, 3) if free else np.eye(3)

  def axis(name: str) -> np.ndarray:
    return R_base.T @ data.xaxis[jid(name)]

  n = len(joint_names)
  perm = np.zeros(n, dtype=np.int64)
  sign = np.zeros(n)
  index = {name: i for i, name in enumerate(joint_names)}
  for i, name in enumerate(joint_names):
    partner = _swap_side(name)
    if partner not in index:
      raise KeyError(f"no mirror partner for joint {name!r} (expected {partner!r})")
    j = index[partner]
    s = -float(axis(name) @ (_MIRROR @ axis(partner)))
    if abs(abs(s) - 1.0) > 0.05:
      raise ValueError(f"joint axes of {name}/{partner} are not mirror images (s={s:.3f})")
    perm[i] = j
    sign[i] = float(np.sign(s))
  return perm, sign


def build_spec(env) -> MirrorSpec:
  env = getattr(env, "unwrapped", env)
  device = env.device
  robot = env.scene["robot"]
  joint_names = tuple(robot.joint_names)
  jperm, jsign = _joint_mirror(env.sim.mj_model, joint_names)
  q0 = robot.data.default_joint_pos[0].detach().cpu().numpy()
  # q_rel mirrored: s * (q_rel[p] + q0[p]) - q0[i]
  joint_offset = jsign * q0[jperm] - q0
  if np.abs(joint_offset).max() > 1e-3:
    print(
      f"[symmetry] default pose is not mirror-symmetric (max {np.abs(joint_offset).max():.3f} rad)"
    )

  # Actions: processed = a * scale + offset. Mirrored target for slot i is
  # s * (a[p] * scale[p] + off[p]); solve for a_i.
  term = env.action_manager.get_term(env.action_manager.active_terms[0])
  target_names = tuple(term._joint_names) if hasattr(term, "_joint_names") else joint_names
  if tuple(target_names) != joint_names:
    raise ValueError("symmetry assumes the action term controls all joints in model order")
  scale = term.scale
  scale = (
    scale[0].detach().cpu().numpy() if torch.is_tensor(scale) else np.full(len(joint_names), scale)
  )
  off = term.offset
  off = off[0].detach().cpu().numpy() if torch.is_tensor(off) else np.full(len(joint_names), off)
  a_sign = jsign * scale[jperm] / scale
  a_off = (jsign * off[jperm] - off) / scale

  def affine(perm, sign, offset=None) -> _Affine:
    perm = np.asarray(perm, dtype=np.int64)
    sign = np.asarray(sign, dtype=np.float32)
    offset = np.zeros_like(sign) if offset is None else np.asarray(offset, dtype=np.float32)
    return _Affine(
      torch.as_tensor(perm, device=device),
      torch.as_tensor(sign, device=device),
      torch.as_tensor(offset, device=device),
    )

  om = env.observation_manager
  groups: dict[str, _Affine] = {}
  for group, names in om.active_terms.items():
    dims = om.group_obs_term_dim[group]
    perm_parts, sign_parts, off_parts = [], [], []
    start = 0
    for name, shape in zip(names, dims, strict=True):
      d = int(np.prod(shape))
      idx = np.arange(start, start + d)
      off = np.zeros(d)
      if name in _VECTOR_RULES:
        sgn = np.array(_VECTOR_RULES[name])
        if len(sgn) != d:
          raise ValueError(f"term {name} has dim {d}, rule expects {len(sgn)}")
        p = idx
      elif name in _FOOT_RULES:
        per = np.array(_FOOT_RULES[name])
        if d != 2 * len(per):
          raise ValueError(f"term {name} has dim {d}, expected 2 feet x {len(per)}")
        p = np.concatenate([idx[len(per) :], idx[: len(per)]])
        sgn = np.tile(per, 2)
      elif name in ("joint_pos", "joint_vel"):
        if d != len(joint_names):
          raise ValueError(f"term {name} has dim {d}, expected {len(joint_names)} joints")
        p = idx[jperm]
        sgn = jsign
        if name == "joint_pos":
          off = joint_offset
      elif name == "actions":
        if d != len(joint_names):
          raise ValueError(f"term {name} has dim {d}, expected {len(joint_names)}")
        p = idx[jperm]
        sgn = a_sign
        off = a_off
      else:
        raise KeyError(f"no mirror rule for observation term {name!r} in group {group!r}")
      perm_parts.append(p)
      sign_parts.append(sgn)
      off_parts.append(off)
      start += d
    groups[group] = affine(
      np.concatenate(perm_parts), np.concatenate(sign_parts), np.concatenate(off_parts)
    )

  pairs = [
    (joint_names[i], joint_names[jperm[i]], float(jsign[i])) for i in range(len(joint_names))
  ]
  return MirrorSpec(groups=groups, actions=affine(jperm, a_sign, a_off), joint_pairs=pairs)


_SPECS: dict[int, MirrorSpec] = {}


def get_spec(env) -> MirrorSpec:
  key = id(getattr(env, "unwrapped", env))
  spec = _SPECS.get(key)
  if spec is None:
    spec = _SPECS[key] = build_spec(env)
    print(
      "[symmetry] mirror spec built; joint pairs: "
      + ", ".join(f"{a}->{b}({'+' if s > 0 else '-'})" for a, b, s in spec.joint_pairs)
    )
  return spec


def mirror(env, obs: TensorDict | None, actions: torch.Tensor | None):
  """rsl-rl ``data_augmentation_func``: append mirrored copies along the batch dim."""
  spec = get_spec(env)
  obs_out = None
  if obs is not None:
    mirrored = TensorDict(
      {g: spec.groups[g](obs[g]) for g in obs.keys()}, batch_size=obs.batch_size
    )
    obs_out = torch.cat([obs, mirrored], dim=0)
  act_out = None
  if actions is not None:
    act_out = torch.cat([actions, spec.actions(actions)], dim=0)
  return obs_out, act_out
