from chase_the_dot.env import ChaseTheDotEnv, Environment, normalize
from chase_the_dot.pid import PID
from chase_the_dot.vpg import VPG
from chase_the_dot.a2c import A2C

VanillaPolicyGradient = VPG
AdvantageActorCritic = A2C

__all__ = [
    "ChaseTheDotEnv", "Environment", "normalize",
    "PID", "VPG", "A2C", "main"
]
