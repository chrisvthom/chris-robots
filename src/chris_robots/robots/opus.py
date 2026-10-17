"""Opus: 3D-printed plastic mini humanoid. 16 DoF (5-DoF legs, 3-DoF arms), ~2.8 kg, 0.32 m.

Hobby servos; the URDF has no actuator data so we use Z-Bot-class servo gains with
the same sim-only leg torque boost.
"""

from chris_robots.robots.base import SOFT_SERVO, ActuatorGroup, RobotDef

_LEGS = (r".*hip.*", r".*knee.*", r".*ankle.*")
_ARMS = (r".*shoulder.*", r".*elbow.*")

OPUS = RobotDef(
  name="opus",
  actuators=(
    ActuatorGroup(
      "servo_leg", _LEGS, effort_limit=3.0, armature=0.004, **{**SOFT_SERVO, "natural_freq_hz": 8.0}
    ),
    ActuatorGroup("servo_arm", _ARMS, effort_limit=1.0, armature=0.002, **SOFT_SERVO),
  ),
  action_scale={r".*(hip|knee|ankle).*": 0.5, r".*(shoulder|elbow|wrist|gripper).*": 0.25},
  walk_vel=0.25,
  max_lin_vel=0.8,
  max_ang_vel=2.0,
  foot_clearance=0.04,
  push_vel=0.15,
  com_offset=0.01,
  hip_adduction_min=0.5,
  arm_swing_gain=1.5,  # light arms: ask for a bigger swing so it reads on camera
  action_rate_weight=-0.15,  # hobby servos: smooth, not twitchy
  sway_roll_deg=6.0,
  sway_yaw_deg=8.0,
)
