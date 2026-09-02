from chase_the_dot.env import ChaseTheDotEnv, Environment, normalize
from chase_the_dot.pid import PID
from chase_the_dot.vpg import VPG

VanillaPolicyGradient = VPG

__all__ = [
    "ChaseTheDotEnv", "Environment", "normalize",
    "PID", "VPG", "main"
]
