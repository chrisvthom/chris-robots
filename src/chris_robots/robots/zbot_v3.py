"""Z-Bot V3 (K-Scale "zbot-6dof"): 6-DoF legs with ankle roll, 4-DoF arms. 20 DoF, ~3.3 kg, 0.40 m.

Feetech STS3250 on legs/shoulders, STS3215 on elbows/grippers. These are hobby
position servos, so the gains are chosen to be servo-like (stiff, well damped)
rather than derived from real motor data.
"""

from chris_robots.robots.base import ActuatorGroup, PostureStd, RobotDef

_LEGS = (r".*hip.*", r".*knee.*", r".*ankle.*")
_SHOULDERS = (r".*shoulder.*",)
_FOREARMS = (r".*elbow.*", r".*gripper.*")

ZBOT_V3 = RobotDef(
  name="zbot_v3",
  actuators=(
    ActuatorGroup("sts3250_leg", _LEGS, effort_limit=3.0, armature=0.005),
    ActuatorGroup("sts3250_arm", _SHOULDERS, effort_limit=3.0, armature=0.003),
    ActuatorGroup("sts3215", _FOREARMS, effort_limit=2.0, armature=0.002),
  ),
  posture=PostureStd(
    standing={".*": 0.05},
    walking={
      r".*hip_pitch.*": 0.3,
      r".*hip_roll.*": 0.15,
      r".*hip_yaw.*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      r".*shoulder_pitch.*": 0.2,
      r".*shoulder_roll.*": 0.1,
      r".*elbow.*": 0.15,
      r".*gripper.*": 0.05,
    },
    running={
      r".*hip_pitch.*": 0.5,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.6,
      r".*ankle_pitch.*": 0.35,
      r".*ankle_roll.*": 0.15,
      r".*shoulder_pitch.*": 0.5,
      r".*shoulder_roll.*": 0.15,
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
)
