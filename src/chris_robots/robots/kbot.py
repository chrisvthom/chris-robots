"""K-Bot V2 (K-Scale), no hands. 20 DoF, ~36 kg, 1.39 m.

Joint names look like ``dof_left_hip_pitch_04``; the numeric suffix is the Robstride
motor size (00 / 02 / 03 / 04). Armature values come from K-Scale's MJCF export,
torque limits from their ``metadata.json`` soft limits.
"""

from chris_robots.robots.base import ActuatorGroup, PostureStd, RobotDef

KBOT = RobotDef(
  name="kbot",
  actuators=(
    ActuatorGroup("robstride_04", (r".*_04$",), effort_limit=84.0, armature=0.04),
    ActuatorGroup(
      "robstride_03",
      (r".*(shoulder_pitch|shoulder_roll|hip_yaw)_03$",),
      effort_limit=42.0,
      armature=0.02,
    ),
    ActuatorGroup(
      "robstride_02", (r".*(shoulder_yaw|elbow)_02$",), effort_limit=11.9, armature=0.0042
    ),
    ActuatorGroup("robstride_00", (r".*_00$",), effort_limit=9.8, armature=0.001),
    # Sim-only upgrades (we're not doing sim2real). On the real robot hip roll is a
    # Robstride 03 (42 Nm) and the ankle a Robstride 02 (12 Nm). Shifting 36 kg over
    # one foot needs ~40 Nm of hip roll, so with the real limits the policy can only
    # shuffle; the real ankle can't even hold the body upright. G1 puts its strongest
    # motor (139 Nm) on hip roll for the same reason.
    ActuatorGroup("hip_roll", (r".*hip_roll_03$",), effort_limit=120.0, armature=0.04),
    ActuatorGroup("ankle", (r".*ankle_02$",), effort_limit=60.0, armature=0.03),
  ),
  posture=PostureStd(
    standing={".*": 0.05},
    walking={
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle.*": 0.25,
      r".*shoulder_pitch.*": 0.4,  # loose: arm swing is rewarded separately
      r".*shoulder_roll.*": 0.15,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    },
    running={
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.6,
      r".*ankle.*": 0.35,
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.2,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.35,
      r".*wrist.*": 0.3,
    },
  ),
  # Legs get a G1-like scale (~0.5 rad per unit action) so a full stride is cheap in
  # action-rate terms; arms stay small so they don't flail.
  action_scale={
    r".*(hip|knee|ankle).*": 0.5,
    r".*(shoulder|elbow|wrist).*": 0.25,
  },
  max_lin_vel=1.0,
  max_ang_vel=1.0,
  foot_clearance=0.08,
  push_vel=0.5,
  com_offset=0.025,
  hip_adduction_min=0.5,
  # Looks: the importer's default pose (knee 29 deg) reads as a crouch once the policy
  # adds its own flexion on top; start from straighter legs. Arms follow the opposite
  # leg so the walk has a visible arm swing.
  init_bend=0.15,
  arm_swing_gain=0.8,
)
