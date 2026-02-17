"""Shared test fixtures for bio-mimetic-swarm."""

from __future__ import annotations

import pytest

from swarm.config import SwarmConfig
from swarm.engine import SwarmEngine


@pytest.fixture
def default_config() -> SwarmConfig:
    """Standard 10-agent test configuration."""
    return SwarmConfig(num_agents=10, steps=100, init_seed=42)


@pytest.fixture
def two_agent_config() -> SwarmConfig:
    """Minimal config for pairwise force tests."""
    return SwarmConfig(
        num_agents=2,
        epsilon=80.0,
        sigma=12.0,
        damping=0.0,
        target_gain=0.0,
        max_force=1e6,
        max_velocity=1e6,
        world_size=1000.0,
        init_spread=0.01,
        steps=100,
        dt=0.01,
        init_seed=42,
    )


@pytest.fixture
def undamped_config() -> SwarmConfig:
    """No damping, no navigation — for energy conservation tests."""
    return SwarmConfig(
        num_agents=6,
        epsilon=40.0,
        sigma=10.0,
        damping=0.0,
        target_gain=0.0,
        max_force=1e6,
        max_velocity=1e6,
        world_size=1000.0,
        init_spread=15.0,
        steps=500,
        dt=0.005,
        init_seed=123,
    )


@pytest.fixture
def default_engine(default_config: SwarmConfig) -> SwarmEngine:
    """Ready-to-use engine with default test config."""
    return SwarmEngine(default_config)


@pytest.fixture
def two_agent_engine(two_agent_config: SwarmConfig) -> SwarmEngine:
    """Ready-to-use 2-agent engine."""
    return SwarmEngine(two_agent_config)


@pytest.fixture
def undamped_engine(undamped_config: SwarmConfig) -> SwarmEngine:
    """Ready-to-use engine with no damping for energy tests."""
    return SwarmEngine(undamped_config)
