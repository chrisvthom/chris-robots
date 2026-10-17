"""GPR / Stompy. Physics model: K-Bot V1 skeleton (same mechanism). 20 DoF, ~29 kg, 1.16 m.

Joint names look like ``left_hip_pitch_04``; the suffix is the Robstride motor size.
GPR's head and body shells are a visual-only overlay (TODO). Same sim-only leg torque
boost as the K-Bot.
"""

from chris_robots.robots.base import ActuatorGroup, RobotDef

GPR = RobotDef(
  name="gpr",
  actuators=(
    ActuatorGroup("legs_04", (r".*(hip_pitch|knee)_04$",), effort_limit=130.0, armature=0.04),
    ActuatorGroup("hip_roll", (r".*hip_roll_03$",), effort_limit=140.0, armature=0.04),
    ActuatorGroup("hip_yaw", (r".*hip_yaw_03$",), effort_limit=60.0, armature=0.02),
    ActuatorGroup("ankle", (r".*ankle_02$",), effort_limit=80.0, armature=0.03),
    ActuatorGroup(
      "arms_03", (r".*(shoulder_pitch|shoulder_roll)_03$",), effort_limit=42.0, armature=0.02
    ),
    ActuatorGroup(
      "arms_02", (r".*(shoulder_yaw|elbow|wrist)_02$",), effort_limit=11.9, armature=0.0042
    ),
  ),
  action_scale={r".*(hip|knee|ankle).*": 0.5, r".*(shoulder|elbow|wrist|gripper).*": 0.25},
  walk_vel=0.8,
  max_lin_vel=2.5,
  max_ang_vel=1.5,
  foot_clearance=0.12,
  push_vel=0.3,
  com_offset=0.025,
  hip_adduction_min=0.5,
  init_bend=0.15,
)
