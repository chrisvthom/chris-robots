"""Import robots from the K-Scale asset dumps into clean, mjlab-ready MJCF models.

Source layout (per robot, as exported by kscale's urdf2mjcf):
  <src>/robot.mjcf      floating-base MJCF with visual meshes + box feet
  <src>/robot.urdf      original URDF
  <src>/metadata.json   per-joint kp/kd + actuator type
  <src>/meshes/*.stl

Output layout (per robot):
  robots/<name>/<name>.xml       cleaned MJCF (what mjlab loads)
  robots/<name>/scene.xml        standalone viewing scene (floor, light, skybox)
  robots/<name>/assets/*.stl     only the meshes the model references
  robots/<name>/source/          original urdf + metadata for provenance

What we change relative to the kscale MJCF:
  * drop the <motor> actuators (mjlab adds <position> actuators from kp/kd),
  * replace the sensor block with mjlab's convention
    (imu_ang_vel, imu_lin_vel, imu_lin_acc, imu_upvector, root_angmom),
  * rename root body -> "base", freejoint -> "root", IMU site -> "imu",
  * rename foot collision boxes -> "{left,right}_foot_collision",
  * add/move "{left,right}_foot" sites to the centre of each sole,
  * drop stale contact/site/camera clutter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from chris_robots import _REPO_ROOT, ROBOTS_DIR

RAVEN = Path("/run/media/christhomas/Raven/Projects/K-Scale/URDFs & Assets")
KSCALE_ASSETS = RAVEN / "kscale-assets"
STAGING = _REPO_ROOT / ".cache" / "staging"


# --------------------------------------------------------------------------- opus


def _rotate_body_contents(body: mujoco.MjsBody, q: np.ndarray) -> None:
  """Re-express a body's children, geoms, sites and inertia in a frame rotated by q."""
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, q)
  R = R.reshape(3, 3)

  def rot(el) -> None:
    el.pos = R @ np.asarray(el.pos)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, np.asarray(el.quat))
    el.quat = out

  for el in (*body.bodies, *body.geoms, *body.sites):
    rot(el)
  body.ipos = R @ np.asarray(body.ipos)
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, q, np.asarray(body.iquat))
  body.iquat = out


def prepare_opus(staging: Path) -> None:
  """Turn the raw opus URDF into a kscale-style ``robot.mjcf`` + ``meshes/``.

  opus.urdf has a massless ``base`` link with a ``floating`` joint to the torso
  (which MuJoCo rejects), mesh collision for every link, and faces -y. We make the
  torso the root with a freejoint, rotate it to face +x, drop the mesh collisions,
  add box collision for each sole, and fix the ``righ_elbow_yaw`` typo.
  """
  src_urdf = RAVEN / "robot" / "opus.urdf"
  src_meshes = RAVEN / "robot" / "meshes"
  staging.mkdir(parents=True, exist_ok=True)
  meshes = staging / "meshes"
  if meshes.is_symlink() or meshes.exists():
    meshes.unlink() if meshes.is_symlink() else shutil.rmtree(meshes)
  meshes.symlink_to(src_meshes)

  txt = src_urdf.read_text()
  txt = re.sub(
    r'<link name="base">.*?</link>\s*<joint name="floating_base".*?</joint>', "", txt, flags=re.S
  )
  txt = txt.replace("righ_elbow_yaw", "right_elbow_yaw")
  # The URDF's "hip_yaw" joints rotate about the forward axis (roll) and "hip_roll"
  # about the vertical axis (yaw). Swap the names so they describe the motion.
  txt = (
    txt.replace("_hip_yaw", "_hip_TMP")
    .replace("_hip_roll", "_hip_yaw")
    .replace("_hip_TMP", "_hip_roll")
  )
  txt = txt.replace(
    '<robot name="opus">',
    f'<robot name="opus"><mujoco><compiler discardvisual="false" fusestatic="false" '
    f'balanceinertia="true" meshdir="{staging}" strippath="false"/></mujoco>',
  )
  (staging / "robot.urdf").write_text(txt)

  spec = mujoco.MjSpec.from_file(str(staging / "robot.urdf"))
  spec.modelname = "opus"
  vis = spec.add_default("visual", spec.default)
  vis.geom.contype, vis.geom.conaffinity, vis.geom.group = 0, 0, 1
  col = spec.add_default("collision", spec.default)
  col.geom.contype, col.geom.conaffinity, col.geom.group = 1, 1, 3

  for g in list(spec.geoms):
    if g.name.endswith("_collision"):
      spec.delete(g)
    else:
      g.classname = vis

  torso = spec.body("Torso")
  torso.add_freejoint(name="root")
  torso.pos = [0, 0, 0.32]
  # Face +x. The freejoint state (keyframe / mjlab init_state) overrides the root
  # body's own quat, so bake the yaw into everything attached to the torso instead.
  _rotate_body_contents(torso, np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]))

  # Sole boxes from the world-frame AABB of each foot mesh (robot upright at qpos0).
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for side in ("left", "right"):
    body = spec.body(f"foot_{side}")
    bid = model.body(body.name).id
    gid = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == bid)
    mid = model.geom_dataid[gid]
    verts = model.mesh_vert[
      model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
    ]
    world = verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    lo, hi = world.min(0), world.max(0)
    hz = 0.01
    center_w = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + hz])
    half = np.array([(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, hz])
    Rb = data.xmat[bid].reshape(3, 3)
    pos_b = Rb.T @ (center_w - data.xpos[bid])
    quat_b = np.zeros(4)
    mujoco.mju_mat2Quat(quat_b, Rb.T.flatten())
    body.add_geom(
      col,
      name=f"{side}_foot_collision_box",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half,
      pos=pos_b,
      quat=quat_b,
    )

  spec.compile()
  (staging / "robot.mjcf").write_text(spec.to_xml())
  (staging / "metadata.json").write_text(json.dumps({"joint_name_to_metadata": {}}, indent=2))


@dataclass
class Source:
  name: str
  src: Path
  display_name: str
  notes: str = ""
  # Joint name substrings used to classify a foot as left/right while walking up
  # the kinematic tree from the foot geom.
  side_tokens: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"left": ("left", "_l_", "_l"), "right": ("right", "_r_", "_r")}
  )
  # Optional hook that (re)builds ``src`` in the kscale layout from some other
  # source format (see ``prepare_opus``).
  prepare: Callable[[Path], None] | None = None


SOURCES: dict[str, Source] = {
  "kbot": Source(
    name="kbot",
    src=KSCALE_ASSETS / "kbot-v2-feet",
    display_name="K-Bot V2",
    notes="K-Scale K-Bot V2 (no hands), Robstride actuators. 20 DoF, ~36 kg, 1.39 m.",
  ),
  "zbot_v3": Source(
    name="zbot_v3",
    src=KSCALE_ASSETS / "zbot-6dof-feet",
    display_name="Z-Bot V3",
    notes="K-Scale Z-Bot V3.1 with 6-DoF legs (ankle roll), Feetech STS3250/STS3215. 20 DoF, ~3.3 kg, 0.40 m.",
  ),
  "zbot_v2": Source(
    name="zbot_v2",
    src=KSCALE_ASSETS / "zbot-feet",
    display_name="Z-Bot V2",
    notes="K-Scale Z-Bot V2, 5-DoF legs, Feetech servos. 18 DoF, ~3.75 kg, 0.48 m.",
  ),
  "gpr": Source(
    name="gpr",
    src=KSCALE_ASSETS / "kbot-v1-feet",
    display_name="GPR / Stompy",
    notes=(
      "General Purpose Robot (Stompy). Physics model is the K-Bot V1 skeleton, which is the "
      "same mechanism; GPR's head and shells are a visual-only overlay (TODO)."
    ),
  ),
  "opus": Source(
    name="opus",
    src=STAGING / "opus",
    display_name="Opus",
    notes="3D-printed plastic mini humanoid (K-Scale 'opus'), hobby servos. 16 DoF, ~2.8 kg, 0.32 m.",
    prepare=prepare_opus,
  ),
}


def _find_parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
  return {c: p for p in root.iter() for c in p}


def _ancestor_joint_names(el: ET.Element, parents: dict[ET.Element, ET.Element]) -> list[str]:
  names = []
  cur = el
  while cur in parents:
    cur = parents[cur]
    if cur.tag == "body":
      for j in cur.findall("joint"):
        names.append(j.get("name", ""))
  return names


def _classify_side(joint_names: list[str], side_tokens) -> str | None:
  for name in joint_names:  # nearest ancestor first
    low = name.lower()
    for side, toks in side_tokens.items():
      if any(t in low for t in toks):
        return side
  return None


_LEG_PATTERNS = {
  "hip_pitch": r".*hip_pitch.*",
  "knee": r".*knee.*",
  "ankle_pitch": r".*ankle(_pitch)?(_\d+)?$",
}
_ELBOW_PATTERN = r".*elbow.*"
_SIDE_PATTERN = {"left": r".*left.*", "right": r".*right.*"}


def _world_axis(model: mujoco.MjModel, data: mujoco.MjData, jid: int) -> np.ndarray:
  return data.xmat[model.jnt_bodyid[jid]].reshape(3, 3) @ model.jnt_axis[jid]


def compute_init_pose(model: mujoco.MjModel, bend: float = 0.25, elbow_bend: float = 0.6) -> dict:
  """Derive a knees-slightly-bent standing pose with the correct sign per joint.

  Sign convention is derived from each joint's world axis at qpos0 (robot standing,
  facing +x, z up): a pitch joint whose axis points along +y rotates a hanging limb
  backwards (-x) for positive angles. Hence, for a crouch that keeps the torso and
  feet level: hip = -b*s, knee = +2b*s, ankle = -b*s, where s = sign(axis . y).
  One-sided joint ranges (knee, elbow) are used as a cross-check.
  """
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  joint_pos: dict[str, float] = {}
  warnings: list[str] = []
  hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  names = {j: model.joint(j).name for j in hinge}

  def one_sided_sign(j: int) -> float | None:
    lo, hi = model.jnt_range[j]
    if abs(lo) < 1e-3 and hi > 0.1:
      return 1.0
    if abs(hi) < 1e-3 and lo < -0.1:
      return -1.0
    return None

  for j in hinge:
    n = names[j]
    axis = _world_axis(model, data, j)
    s = float(np.sign(axis[1])) if abs(axis[1]) > 0.7 else 0.0
    if re.fullmatch(_LEG_PATTERNS["knee"], n):
      target = 2 * bend * s
      os_ = one_sided_sign(j)
      if os_ is not None and np.sign(target) != os_:
        warnings.append(
          f"{n}: axis says {np.sign(target):+.0f} but range is one-sided {os_:+.0f}; using range"
        )
        target = 2 * bend * os_
      joint_pos[n] = target
    elif re.fullmatch(_LEG_PATTERNS["hip_pitch"], n):
      joint_pos[n] = -bend * s
    elif re.fullmatch(_LEG_PATTERNS["ankle_pitch"], n):
      joint_pos[n] = -bend * s
    elif re.fullmatch(_ELBOW_PATTERN, n):
      os_ = one_sided_sign(j)
      if os_ is not None:
        joint_pos[n] = elbow_bend * os_

  # Apply pose and drop the base so the soles rest just above the floor.
  qpos = data.qpos.copy()
  for j in hinge:
    if names[j] in joint_pos:
      qpos[model.jnt_qposadr[j]] = joint_pos[names[j]]
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  sole_z = []
  tilt = []
  for side in ("left", "right"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    R = data.geom_xmat[gid].reshape(3, 3)
    half = model.geom_size[gid]
    sole_z.append(float(data.geom_xpos[gid][2] - np.abs(R[2]) @ half))
    # Foot flatness: the box axis closest to vertical should be ~vertical.
    tilt.append(float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2])), -1, 1)))))
  drop = min(sole_z) - 0.002
  base_z = float(model.body("base").pos[2] - drop)
  if max(tilt) > 3.0:
    warnings.append(f"feet not level in init pose: tilt={tilt}")
  if abs(sole_z[0] - sole_z[1]) > 0.005:
    warnings.append(f"feet at different heights in init pose: {sole_z}")
  return {
    "joint_pos": {k: round(v, 4) for k, v in joint_pos.items()},
    "base_pos": [0.0, 0.0, round(base_z, 4)],
    "foot_tilt_deg": [round(t, 2) for t in tilt],
    "warnings": warnings,
  }


def import_robot(src_cfg: Source, out_root: Path) -> dict:
  src = src_cfg.src
  if src_cfg.prepare is not None:
    src_cfg.prepare(src)
  mjcf_path = src / "robot.mjcf"
  assert mjcf_path.exists(), mjcf_path
  out_dir = out_root / src_cfg.name
  assets_dir = out_dir / "assets"
  source_dir = out_dir / "source"
  for d in (assets_dir, source_dir):
    d.mkdir(parents=True, exist_ok=True)

  tree = ET.parse(mjcf_path)
  root = tree.getroot()
  root.set("model", src_cfg.name)

  # --- compiler -------------------------------------------------------------
  compiler = root.find("compiler")
  if compiler is None:
    compiler = ET.SubElement(root, "compiler")
  compiler.set("angle", "radian")
  compiler.set("meshdir", "assets")
  compiler.set("autolimits", "true")

  # --- defaults: strip actuator-only attributes, fix undefined material -------
  for d in root.iter("default"):
    for motor in d.findall("motor"):
      d.remove(motor)
    for j in d.findall("joint"):
      j.attrib.pop("actuatorfrcrange", None)
    for g in d.findall("geom"):
      if g.get("material") == "visualgeom":
        g.attrib.pop("material")
    for eq in d.findall("equality"):
      d.remove(eq)

  # --- drop kscale actuators / sensors / contact --------------------------------
  for tag in ("actuator", "sensor", "contact", "keyframe"):
    for el in root.findall(tag):
      root.remove(el)

  worldbody = root.find("worldbody")
  assert worldbody is not None
  bodies = worldbody.findall("body")
  assert len(bodies) == 1, "expected a single root body"
  base = bodies[0]
  old_root_name = base.get("name")
  base.set("name", "base")
  fj = base.find("freejoint")
  if fj is None:
    fj = next((j for j in base.findall("joint") if j.get("type") == "free"), None)
  if fj is None:
    fj = ET.SubElement(base, "freejoint")
  fj.set("name", "root")

  # Remove kscale clutter directly under the root body.
  for site in list(base.findall("site")):
    if site.get("name", "").endswith("_site") and "imu" not in site.get("name", "").lower():
      base.remove(site)
  for cam in list(base.findall("camera")):
    base.remove(cam)

  parents = _find_parent_map(root)

  # --- IMU site -------------------------------------------------------------
  # kscale's IMU site sits on a sub-body with a rotated frame. We don't care about
  # sim2real here, so put the IMU at the base origin, aligned with the base frame:
  # observations (base lin/ang vel) then come out in the same frame as the commands.
  for s in list(root.iter("site")):
    if "imu" in s.get("name", "").lower():
      parents[s].remove(s)
  imu = ET.SubElement(base, "site")
  imu.set("name", "imu")
  imu.set("pos", "0 0 0")
  imu.set("size", "0.01")
  imu.set("group", "5")
  parents = _find_parent_map(root)

  # --- feet: find collision boxes, classify side, rename --------------------
  foot_geoms = [g for g in worldbody.iter("geom") if g.get("class") == "collision"]
  assert len(foot_geoms) == 2, [g.get("name") for g in foot_geoms]
  foot_bodies: dict[str, ET.Element] = {}
  for g in foot_geoms:
    side = _classify_side(_ancestor_joint_names(g, parents), src_cfg.side_tokens)
    assert side is not None, f"could not classify side of {g.get('name')}"
    assert side not in foot_bodies, "both feet classified as the same side"
    g.set("name", f"{side}_foot_collision")
    foot_bodies[side] = parents[g]
    # Remove any pre-existing foot sites; we re-add them at the sole below.
    for s in list(parents[g].findall("site")):
      if "foot" in s.get("name", "").lower():
        parents[g].remove(s)

  # Visual geoms: make sure every geom has a name and is in the visual class.
  for g in worldbody.iter("geom"):
    if g.get("class") not in ("collision",):
      g.set("class", "visual")

  # --- meshes: copy referenced files into assets/ --------------------------
  used = set()
  for m in root.iter("mesh"):
    f = m.get("file")
    if f is None:
      continue
    src_file = src / f
    assert src_file.exists(), src_file
    dst = assets_dir / Path(f).name
    if not dst.exists() or dst.stat().st_size != src_file.stat().st_size:
      shutil.copy2(src_file, dst)
    m.set("file", Path(f).name)
    used.add(dst.name)
  for stale in assets_dir.iterdir():
    if stale.name not in used:
      stale.unlink()

  # --- sensors (mjlab convention) ------------------------------------------
  sensor = ET.SubElement(root, "sensor")
  ET.SubElement(sensor, "gyro", name="imu_ang_vel", site="imu")
  ET.SubElement(sensor, "velocimeter", name="imu_lin_vel", site="imu")
  ET.SubElement(sensor, "accelerometer", name="imu_lin_acc", site="imu")
  ET.SubElement(
    sensor,
    "framezaxis",
    name="imu_upvector",
    objtype="body",
    objname="world",
    reftype="site",
    refname="imu",
  )
  ET.SubElement(sensor, "subtreeangmom", name="root_angmom", body="base")

  # Write once so we can compile and place foot sites at the soles.
  xml_path = out_dir / f"{src_cfg.name}.xml"
  ET.indent(tree, space="  ")
  tree.write(xml_path, encoding="unicode", xml_declaration=False)

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  foot_info = {}
  for side, body_el in foot_bodies.items():
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    bid = model.geom_bodyid[gid]
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
    # Lowest point of the box in world: centre - extent along world z.
    R = data.geom_xmat[gid].reshape(3, 3)
    half = model.geom_size[gid]
    zmin = data.geom_xpos[gid][2] - float(np.abs(R[2]) @ half)
    sole_world = np.array([data.geom_xpos[gid][0], data.geom_xpos[gid][1], zmin])
    Rb = data.xmat[bid].reshape(3, 3)
    sole_local = Rb.T @ (sole_world - data.xpos[bid])
    site = ET.SubElement(body_el, "site")
    site.set("name", f"{side}_foot")
    site.set("pos", " ".join(f"{v:.6f}" for v in sole_local))
    site.set("size", "0.01")
    site.set("group", "5")
    site.set("rgba", "1 0 0 1")
    foot_info[side] = {
      "body": model.body(bid).name,
      "sole_height_at_init": float(zmin),
      "box_half_size": half.tolist(),
    }

  ET.indent(tree, space="  ")
  tree.write(xml_path, encoding="unicode", xml_declaration=False)

  # Recompile to validate and gather stats.
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == 3]
  # Model order of the feet. mjlab resolves regex patterns (contact sensors, site
  # lists) in model order, so every per-foot tensor must use this order.
  foot_order = sorted(
    foot_info, key=lambda side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
  )
  base_z = float(model.body("base").pos[2])
  sole_z = min(v["sole_height_at_init"] for v in foot_info.values())
  init_pose = compute_init_pose(model)
  for w in init_pose["warnings"]:
    print(f"  [warn] {src_cfg.name}: {w}")

  # Also record the joint ranges and armature so the Python constants can be
  # written against the data instead of re-reading the XML.
  joint_info = {}
  for i in range(model.njnt):
    if model.jnt_type[i] != 3:
      continue
    n = model.joint(i).name
    joint_info[n] = {
      "range": [round(float(x), 4) for x in model.jnt_range[i]],
      "armature": round(float(model.dof_armature[model.jnt_dofadr[i]]), 5),
      "frictionloss": round(float(model.dof_frictionloss[model.jnt_dofadr[i]]), 4),
      "world_axis_at_init": [round(float(x), 3) for x in _world_axis(model, data, i)],
    }

  # --- scene.xml for standalone viewing ------------------------------------
  # Keyframe lives in scene.xml (not the robot xml) because mjlab adds its own
  # "init_state" key to the robot spec.
  key_qpos = list(init_pose["base_pos"]) + [1, 0, 0, 0]
  for i in range(model.njnt):
    if model.jnt_type[i] == 3:
      key_qpos.append(init_pose["joint_pos"].get(model.joint(i).name, 0.0))
  key_str = " ".join(f"{v:.4f}" for v in key_qpos)
  scene = f"""<mujoco model="{src_cfg.name} scene">
  <include file="{src_cfg.name}.xml"/>

  <keyframe>
    <key name="init_state" qpos="{key_str}"/>
  </keyframe>

  <statistic center="0 0 {base_z / 2:.2f}" extent="{max(1.0, base_z * 1.6):.2f}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20" offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 {base_z + 1.5:.2f}" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="0"/>
  </worldbody>
</mujoco>
"""
  (out_dir / "scene.xml").write_text(scene)

  # --- provenance -----------------------------------------------------------
  for fname in ("robot.urdf", "metadata.json", "robot.mjcf"):
    if (src / fname).exists():
      shutil.copy2(src / fname, source_dir / fname)

  metadata = json.loads((src / "metadata.json").read_text())["joint_name_to_metadata"]
  info = {
    "name": src_cfg.name,
    "display_name": src_cfg.display_name,
    "notes": src_cfg.notes,
    "source": str(src),
    "source_root_body": old_root_name,
    "dof": int(model.nv - 6),
    "mass_kg": round(float(model.body_subtreemass[model.body("base").id]), 3),
    "base_height_at_init_m": base_z,
    "sole_height_at_init_m": sole_z,
    "standing_base_height_m": round(base_z - sole_z, 4),
    "joints": joint_names,
    "joint_info": joint_info,
    "init_pose": init_pose,
    "foot_order": foot_order,
    "feet": foot_info,
    "joint_metadata": {
      j: {
        k: metadata[j][k]
        for k in ("kp", "kd", "actuator_type", "soft_torque_limit")
        if k in metadata[j]
      }
      for j in joint_names
      if j in metadata
    },
    "nmesh": int(model.nmesh),
  }
  (out_dir / "robot.json").write_text(json.dumps(info, indent=2) + "\n")
  return info


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "robots", nargs="*", default=list(SOURCES), help="robots to import (default: all)"
  )
  parser.add_argument("--out", type=Path, default=ROBOTS_DIR)
  args = parser.parse_args()
  for name in args.robots:
    info = import_robot(SOURCES[name], args.out)
    print(
      f"OK  {name:8s} dof={info['dof']:2d} mass={info['mass_kg']:6.2f}kg "
      f"base_h={info['standing_base_height_m']:.3f}m feet={ {k: v['body'] for k, v in info['feet'].items()} }"
    )


if __name__ == "__main__":
  main()
