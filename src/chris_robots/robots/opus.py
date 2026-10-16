"""Opus: 3D-printed plastic mini humanoid. 16 DoF (5-DoF legs, 3-DoF arms), ~2.8 kg, 0.32 m.

Hobby servos; the URDF has no actuator data so we use Z-Bot-class servo gains.
"""

from chris_robots.robots.base import ActuatorGroup, PostureStd, RobotDef

_LEGS = (r".*hip.*", r".*knee.*", r".*ankle.*")
_ARMS = (r".*shoulder.*", r".*elbow.*")

OPUS = RobotDef(
  name="opus",
  actuators=(
    ActuatorGroup("servo_leg", _LEGS, effort_limit=2.5, armature=0.004),
    ActuatorGroup("servo_arm", _ARMS, effort_limit=2.0, armature=0.002),
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
    },
  ),
  action_scale=0.25,
  max_lin_vel=0.3,
  max_ang_vel=1.0,
  foot_clearance=0.03,
  push_vel=0.2,
  com_offset=0.01,
  hip_adduction_min=0.5,
)
