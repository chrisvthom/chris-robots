"""Z-Bot V2 (K-Scale "zbot"): 5-DoF legs (no ankle roll), 4-DoF arms. 18 DoF, ~3.8 kg, 0.48 m.

Feetech STS3215 (12 V) servos throughout. Servo-like gains, see zbot_v3. Leg torque
boosted ~1.8x (sim-only) so it can run.
"""

from chris_robots.robots.base import SOFT_SERVO, ActuatorGroup, RobotDef

_LEGS = (r".*hip.*", r".*knee.*", r".*ankle.*")
_ARMS = (r".*shoulder.*", r".*elbow.*", r".*gripper.*")

ZBOT_V2 = RobotDef(
  name="zbot_v2",
  actuators=(
    ActuatorGroup("sts3215_leg", _LEGS, effort_limit=3.0, armature=0.004, **SOFT_SERVO),
    ActuatorGroup("sts3215_arm", _ARMS, effort_limit=1.0, armature=0.002, **SOFT_SERVO),
  ),
  action_scale={r".*(hip|knee|ankle).*": 0.5, r".*(shoulder|elbow|wrist|gripper).*": 0.25},
  walk_vel=0.3,
  max_lin_vel=1.0,
  max_ang_vel=2.0,
  foot_clearance=0.05,
  push_vel=0.15,
  com_offset=0.01,
  hip_adduction_min=0.5,
  arm_swing_gain=1.5,  # light arms: ask for a bigger swing so it reads on camera
  action_rate_weight=-0.15,  # hobby servos: smooth, not twitchy
  sway_roll_deg=6.0,
  sway_yaw_deg=8.0,
)
