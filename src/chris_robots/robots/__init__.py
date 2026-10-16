"""All robots, in training priority order."""

from chris_robots.robots.base import ActuatorGroup, PostureStd, RobotDef
from chris_robots.robots.gpr import GPR
from chris_robots.robots.kbot import KBOT
from chris_robots.robots.opus import OPUS
from chris_robots.robots.zbot_v2 import ZBOT_V2
from chris_robots.robots.zbot_v3 import ZBOT_V3

ALL_ROBOTS: tuple[RobotDef, ...] = (KBOT, ZBOT_V3, ZBOT_V2, OPUS, GPR)
ROBOTS_BY_NAME: dict[str, RobotDef] = {r.name: r for r in ALL_ROBOTS}

__all__ = [
  "ActuatorGroup",
  "PostureStd",
  "RobotDef",
  "ALL_ROBOTS",
  "ROBOTS_BY_NAME",
  "KBOT",
  "ZBOT_V3",
  "ZBOT_V2",
  "OPUS",
  "GPR",
]
