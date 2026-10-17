"""Adversarial Motion Priors (Peng et al. 2021) as a self-training mjlab reward term.

The discriminator ``D`` sees state transitions ``(f_t, f_{t+1})`` of hand-picked
features (joint positions/velocities, root velocity, gravity direction, root height,
foot positions in the base frame) and is trained to output +1 on retargeted mocap
transitions (``data/mocap/<robot>.npz`` from ``uv run mocap``) and -1 on the policy's.
The style reward is ``max(0, 1 - 0.25 (D - 1)^2)``, added to the usual velocity-tracking
task rewards. Instead of hand-designed arm-swing / torso-sway / clock terms, the
policy is simply rewarded for moving like the mocap actor while doing what the
velocity command says.

Why a reward term and not an rsl-rl runner: the reward manager is called every env
step with the full env state, so collecting policy transitions and periodically
fitting ``D`` here keeps the standard ``train`` / ``play`` / export tooling untouched.
rsl-rl runs rollouts under ``torch.inference_mode``; the discriminator update
re-enables grad for its own tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


class Discriminator(torch.nn.Module):
  def __init__(self, in_dim: int, hidden: tuple[int, ...] = (1024, 512)):
    super().__init__()
    layers: list[torch.nn.Module] = []
    d = in_dim
    for h in hidden:
      layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
      d = h
    self.trunk = torch.nn.Sequential(*layers)
    self.head = torch.nn.Linear(d, 1)
    torch.nn.init.uniform_(self.head.weight, -0.01, 0.01)
    torch.nn.init.zeros_(self.head.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(x)).squeeze(-1)


class amp_style:
  """Reward term: AMP style reward, with the discriminator trained in place."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.device = env.device
    asset: Entity = env.scene["robot"]
    data = np.load(Path(p["motion_file"]))
    fps = float(data["fps"])
    assert abs(fps * env.step_dt - 1.0) < 1e-3, (fps, env.step_dt)

    # Joint ordering: npz is MuJoCo model order, the entity may differ.
    ref_names = [str(n) for n in data["joint_names"]]
    self.joint_ids = torch.tensor(
      [asset.joint_names.index(n) for n in ref_names], device=self.device
    )
    site_ids, _ = asset.find_sites(list(p["foot_sites"]), preserve_order=True)
    self.site_ids = torch.tensor(site_ids, device=self.device)
    # The npz stores feet in the robot's foot_sites order; the env term is given the
    # same tuple, so no permutation is needed.

    # Which features the discriminator sees. Sim joint/angular velocities carry
    # high-frequency contact noise that the smooth mocap never has, which lets D win on
    # noise alone; so by default velocities are left implicit in the transition pair,
    # taken ``transition_steps`` policy steps apart (0.1 s) rather than consecutive.
    self.feature_set = tuple(
      p.get(
        "feature_set", ("joint_pos", "root_lin_vel_b", "gravity_b", "root_height", "foot_pos_b")
      )
    )
    self.k = int(p.get("transition_steps", 5))
    ref = np.concatenate([data[name] for name in self.feature_set], axis=1).astype(np.float32)
    clip = data["clip"]
    h_stand = float(p["standing_height"])
    ok = data["root_height"][:, 0] > p.get("min_height_frac", 0.85) * h_stand
    k = self.k
    same_clip = clip[k:] == clip[:-k]
    valid = same_clip & ok[k:] & ok[:-k]
    ref_t = torch.tensor(ref, device=self.device)
    idx = torch.tensor(np.nonzero(valid)[0], device=self.device)
    self.expert = torch.cat([ref_t[idx], ref_t[idx + k]], dim=1)  # [M, 2F]
    self.feat_dim = ref.shape[1]
    mean = self.expert.mean(0)
    std = self.expert.std(0).clamp(min=1e-2)
    self.register_norm(mean, std)
    print(
      f"[AMP] {self.expert.shape[0]} expert transitions, {self.feat_dim} features, "
      f"{int(valid.sum())}/{len(valid)} frames kept"
    )

    self.disc = Discriminator(2 * self.feat_dim, tuple(p.get("hidden", (1024, 512)))).to(
      self.device
    )
    self.opt = torch.optim.Adam(self.disc.parameters(), lr=float(p.get("lr", 1e-4)))
    self.grad_penalty = float(p.get("grad_penalty", 10.0))
    self.logit_reg = float(p.get("logit_reg", 0.05))
    self.update_every = int(p.get("update_every", 24))
    self.minibatch = int(p.get("minibatch", 4096))
    self.epochs = int(p.get("epochs", 1))
    self.train = bool(p.get("train", True))

    n = env.num_envs
    cap = int(p.get("buffer_steps", 24)) * n
    self.buffer = torch.zeros(cap, 2 * self.feat_dim, device=self.device)
    self.buf_ptr = 0
    self.buf_full = False
    self.hist = torch.zeros(self.k, n, self.feat_dim, device=self.device)  # ring of past feats
    self.step = 0
    self.last_stats: dict[str, float] = {}

  def register_norm(self, mean: torch.Tensor, std: torch.Tensor) -> None:
    self.mean = mean
    self.std = std

  # ---- features ---------------------------------------------------------------------
  def features(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset: Entity = env.scene["robot"]
    d = asset.data
    quat = d.root_link_quat_w
    parts = []
    for name in self.feature_set:
      if name == "joint_pos":
        parts.append(d.joint_pos[:, self.joint_ids])
      elif name == "joint_vel":
        parts.append(d.joint_vel[:, self.joint_ids])
      elif name == "root_lin_vel_b":
        parts.append(quat_apply_inverse(quat, d.root_link_lin_vel_w))
      elif name == "root_ang_vel_b":
        parts.append(d.root_link_ang_vel_b)
      elif name == "gravity_b":
        parts.append(d.projected_gravity_b)
      elif name == "root_height":
        parts.append(d.root_link_pos_w[:, 2:3])
      elif name == "foot_pos_b":
        feet = d.site_pos_w[:, self.site_ids] - d.root_link_pos_w[:, None, :]
        parts.append(
          torch.cat([quat_apply_inverse(quat, feet[:, i]) for i in range(feet.shape[1])], dim=1)
        )
      else:
        raise KeyError(name)
    return torch.cat(parts, dim=1)

  def norm(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std

  # ---- reward -----------------------------------------------------------------------
  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del kwargs
    feat = self.features(env)
    slot = self.step % self.k
    first = env.episode_length_buf <= self.k  # history would span a reset
    prev = torch.where(first[:, None], feat, self.hist[slot])  # feature k steps ago
    trans = torch.cat([prev, feat], dim=1)
    with torch.no_grad():
      logits = self.disc(self.norm(trans))
    reward = torch.clamp(1.0 - 0.25 * (logits - 1.0) ** 2, min=0.0)

    with torch.inference_mode(False):
      self.hist[slot].copy_(feat.clone())
      self.step += 1
      if self.train:
        self._store(trans[~first].clone())
        if self.step % self.update_every == 0 and (self.buf_full or self.buf_ptr > self.minibatch):
          self._update()
    return reward

  def _store(self, x: torch.Tensor) -> None:
    n = x.shape[0]
    cap = self.buffer.shape[0]
    if n >= cap:
      self.buffer.copy_(x[-cap:])
      self.buf_ptr = 0
      self.buf_full = True
      return
    end = self.buf_ptr + n
    if end <= cap:
      self.buffer[self.buf_ptr : end] = x
    else:
      k = cap - self.buf_ptr
      self.buffer[self.buf_ptr :] = x[:k]
      self.buffer[: n - k] = x[k:]
      self.buf_full = True
    self.buf_ptr = end % cap
    if end >= cap:
      self.buf_full = True

  def _update(self) -> None:
    size = self.buffer.shape[0] if self.buf_full else self.buf_ptr
    n_batches = max(1, size // self.minibatch)
    stats = {"loss": 0.0, "acc_expert": 0.0, "acc_policy": 0.0, "gp": 0.0}
    count = 0
    with torch.enable_grad():
      self.disc.train()
      for _ in range(self.epochs):
        perm = torch.randperm(size, device=self.device)
        for b in range(n_batches):
          pol = self.buffer[perm[b * self.minibatch : (b + 1) * self.minibatch]]
          eidx = torch.randint(0, self.expert.shape[0], (pol.shape[0],), device=self.device)
          exp = self.norm(self.expert[eidx]).requires_grad_(True)
          pol_n = self.norm(pol)
          d_exp = self.disc(exp)
          d_pol = self.disc(pol_n)
          loss_exp = ((d_exp - 1.0) ** 2).mean()
          loss_pol = ((d_pol + 1.0) ** 2).mean()
          grad = torch.autograd.grad(d_exp.sum(), exp, create_graph=True)[0]
          gp = (grad.norm(dim=1) ** 2).mean()
          reg = (d_exp**2).mean() + (d_pol**2).mean()
          loss = 0.5 * (loss_exp + loss_pol) + self.grad_penalty * gp + self.logit_reg * reg
          self.opt.zero_grad(set_to_none=True)
          loss.backward()
          self.opt.step()
          stats["loss"] += float(loss.detach())
          stats["acc_expert"] += float((d_exp > 0).float().mean())
          stats["acc_policy"] += float((d_pol < 0).float().mean())
          stats["gp"] += float(gp.detach())
          count += 1
      self.disc.eval()
    self.last_stats = {k: v / max(count, 1) for k, v in stats.items()}
    if (self.step // self.update_every) % 50 == 1:
      s = self.last_stats
      print(
        f"[AMP] step {self.step}: loss {s['loss']:.3f}  acc expert {s['acc_expert']:.2f} "
        f"policy {s['acc_policy']:.2f}  gp {s['gp']:.3f}"
      )

  # ---- checkpointing ------------------------------------------------------------------
  def state_dict(self) -> dict:
    return {"disc": self.disc.state_dict(), "opt": self.opt.state_dict()}

  def load_state_dict(self, sd: dict) -> None:
    self.disc.load_state_dict(sd["disc"])
    self.opt.load_state_dict(sd["opt"])
