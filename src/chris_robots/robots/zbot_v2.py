"""Z-Bot V2 (K-Scale "zbot"): 5-DoF legs (no ankle roll), 4-DoF arms. 18 DoF, ~3.8 kg, 0.48 m.

Feetech STS3215 (12 V) servos throughout. Servo-like gains, see zbot_v3.
"""

from chris_robots.robots.base import ActuatorGroup, PostureStd, RobotDef

_LEGS = (r".*hip.*", r".*knee.*", r".*ankle.*")
_ARMS = (r".*shoulder.*", r".*elbow.*", r".*gripper.*")

ZBOT_V2 = RobotDef(
  name="zbot_v2",
  actuators=(
    ActuatorGroup("sts3215_leg", _LEGS, effort_limit=2.5, armature=0.004),
    ActuatorGroup("sts3215_arm", _ARMS, effort_limit=2.0, armature=0.002),
  ),
  posture=PostureStd(
    standing={".*": 0.05},
    walking={
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle.*": 0.25,
      r".*shoulder_pitch.*": 0.2,
      r".*shoulder_yaw.*": 0.1,
      r".*elbow.*": 0.15,
      r".*gripper.*": 0.05,
    },
    running={
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.6,
      r".*ankle.*": 0.35,
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.3,
      r".*gripper.*": 0.05,
    },
  ),
  action_scale={r".*(hip|knee|ankle).*": 0.5, r".*(shoulder|elbow|wrist|gripper).*": 0.25},
  max_lin_vel=0.35,
  max_ang_vel=1.0,
  foot_clearance=0.03,
  push_vel=0.2,
  com_offset=0.01,
  hip_adduction_min=0.5,
)
