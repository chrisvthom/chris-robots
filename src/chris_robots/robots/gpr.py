"""GPR / Stompy. Physics model: K-Bot V1 skeleton (same mechanism). 20 DoF, ~29 kg, 1.16 m.

Joint names look like ``left_hip_pitch_04``; the suffix is the Robstride motor size.
GPR's head and body shells are a visual-only overlay (TODO).
"""

from chris_robots.robots.base import ActuatorGroup, PostureStd, RobotDef

GPR = RobotDef(
  name="gpr",
  actuators=(
    ActuatorGroup("robstride_04", (r".*_04$",), effort_limit=84.0, armature=0.04),
    ActuatorGroup(
      "robstride_03",
      (r".*(shoulder_pitch|shoulder_roll|hip_yaw)_03$",),
      effort_limit=42.0,
      armature=0.02,
    ),
    ActuatorGroup(
      "robstride_02", (r".*(shoulder_yaw|elbow|wrist)_02$",), effort_limit=11.9, armature=0.0042
    ),
    # Sim-only stronger hip roll + ankle, see kbot.py.
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
      r".*shoulder_pitch.*": 0.15,
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
  action_scale=0.25,
  max_lin_vel=1.0,
  max_ang_vel=1.0,
  foot_clearance=0.08,
  push_vel=0.5,
  com_offset=0.025,
  hip_adduction_min=0.5,
)
