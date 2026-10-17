"""Z-Bot V3 (K-Scale "zbot-6dof"): 6-DoF legs with ankle roll, 4-DoF arms. 20 DoF, ~3.3 kg, 0.40 m.

Feetech STS3250 on legs/shoulders, STS3215 on elbows/grippers. These are hobby
position servos, so the gains are chosen to be servo-like (stiff, well damped)
rather than derived from real motor data. Leg torque is ~1.7x the real servo so the
little robot can run (sim-only).
"""

from chris_robots.robots.base import SOFT_SERVO, ActuatorGroup, RobotDef

_LEGS = (r".*hip.*", r".*knee.*", r".*ankle.*")
_SHOULDERS = (r".*shoulder.*",)
_FOREARMS = (r".*elbow.*", r".*gripper.*")

ZBOT_V3 = RobotDef(
  name="zbot_v3",
  actuators=(
    ActuatorGroup("sts3250_leg", _LEGS, effort_limit=3.0, armature=0.005, **SOFT_SERVO),
    ActuatorGroup("sts3250_arm", _SHOULDERS, effort_limit=1.5, armature=0.003, **SOFT_SERVO),
    ActuatorGroup("sts3215", _FOREARMS, effort_limit=1.0, armature=0.002, **SOFT_SERVO),
  ),
  action_scale={r".*(hip|knee|ankle).*": 0.5, r".*(shoulder|elbow|wrist|gripper).*": 0.25},
  walk_vel=0.3,
  max_lin_vel=1.0,
  max_ang_vel=2.0,
  foot_clearance=0.05,
  push_vel=0.15,
  com_offset=0.01,
  arm_swing_gain=1.5,  # light arms: ask for a bigger swing so it reads on camera
  action_rate_weight=-0.15,  # hobby servos: smooth, not twitchy
  sway_roll_deg=10.0,
  sway_yaw_deg=12.0,
)
