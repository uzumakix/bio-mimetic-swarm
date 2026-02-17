"""
bio-mimetic-swarm
=================
Decentralized multi-agent swarm intelligence using Lennard-Jones
artificial potential fields.

Example::

    from swarm import SwarmConfig, SwarmEngine

    cfg = SwarmConfig(num_agents=50, epsilon=80.0)
    engine = SwarmEngine(cfg)

    for _ in range(600):
        engine.step()
        print(engine.spread, engine.mean_speed)
"""

__version__ = "2.0.0"
__author__ = "Bio-Mimetic Swarm Project"

from .config import SwarmConfig, build_config, list_scenarios, load_scenario
from .engine import SwarmEngine

__all__ = [
    "SwarmConfig",
    "SwarmEngine",
    "build_config",
    "list_scenarios",
    "load_scenario",
]
