"""K-Bot V2 (K-Scale), no hands. 20 DoF, ~36 kg, 1.39 m.

Joint names look like ``dof_left_hip_pitch_04``; the numeric suffix is the Robstride
motor size (00 / 02 / 03 / 04). Armature values come from K-Scale's MJCF export.
Torque limits are deliberately *above* the real motors' (see below): the policies
only ever run in a browser and are meant to look athletic, not to transfer.
"""

from chris_robots.robots.base import ActuatorGroup, RobotDef

KBOT = RobotDef(
  name="kbot",
  actuators=(
    # Real limits: Robstride 04 = 84 Nm, 03 = 42 Nm, 02 = 12 Nm, 00 = 10 Nm. Legs get
    # ~1.5x so a 36 kg body can actually run and lean; arms stay realistic.
    ActuatorGroup("legs_04", (r".*(hip_pitch|knee)_04$",), effort_limit=130.0, armature=0.04),
    ActuatorGroup("hip_roll", (r".*hip_roll_03$",), effort_limit=140.0, armature=0.04),
    ActuatorGroup("hip_yaw", (r".*hip_yaw_03$",), effort_limit=60.0, armature=0.02),
    ActuatorGroup("ankle", (r".*ankle_02$",), effort_limit=80.0, armature=0.03),
    ActuatorGroup(
      "arms_03", (r".*(shoulder_pitch|shoulder_roll)_03$",), effort_limit=42.0, armature=0.02
    ),
    ActuatorGroup("arms_02", (r".*(shoulder_yaw|elbow)_02$",), effort_limit=11.9, armature=0.0042),
    ActuatorGroup("wrist_00", (r".*_00$",), effort_limit=9.8, armature=0.001),
  ),
  # Legs get a G1-like scale (~0.5 rad per unit action) so a full stride is cheap in
  # action-rate terms; arms smaller so they don't flail.
  action_scale={
    r".*(hip|knee|ankle).*": 0.5,
    r".*(shoulder|elbow|wrist).*": 0.25,
  },
  walk_vel=0.8,
  max_lin_vel=2.5,
  max_ang_vel=1.5,
  foot_clearance=0.12,
  push_vel=0.3,
  com_offset=0.025,
  hip_adduction_min=0.5,
  # Looks: the importer's default pose (knee 29 deg) reads as a crouch once the policy
  # adds its own flexion on top; start from straighter legs.
  init_bend=0.15,
)
