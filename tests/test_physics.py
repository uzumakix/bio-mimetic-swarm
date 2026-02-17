"""
Physics invariant tests for the swarm engine.

Verifies fundamental physical laws and numerical properties:
  - Newton's third law (F_ij = -F_ji)
  - Energy conservation under symplectic integration
  - Equilibrium spacing matches LJ prediction
  - Force/velocity saturation bounds
  - Boundary reflection
  - Robustness under noise and dropout
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.config import SwarmConfig
from swarm.engine import SwarmEngine


class TestNewtonThirdLaw:
    """F_ij = -F_ji for all agent pairs."""

    def test_force_symmetry_two_agents(self, two_agent_config: SwarmConfig) -> None:
        """Two agents: forces should be equal and opposite."""
        engine = SwarmEngine(two_agent_config)
        engine.pos = np.array([[0.0, 0.0], [20.0, 0.0]])
        engine.vel = np.zeros((2, 2))

        forces = engine._lj_forces(engine.pos)

        np.testing.assert_allclose(
            forces[0], -forces[1], atol=1e-10,
            err_msg="LJ forces violate Newton's third law",
        )

    def test_force_symmetry_multiple_agents(self, default_config: SwarmConfig) -> None:
        """N agents: sum of all internal forces should be zero."""
        engine = SwarmEngine(default_config)
        total = np.sum(engine._lj_forces(engine.pos), axis=0)

        np.testing.assert_allclose(
            total, [0.0, 0.0], atol=1e-6,
            err_msg="Sum of internal LJ forces is non-zero",
        )


class TestEquilibriumSpacing:
    """Two agents should settle at r_eq = 2^(1/6) · σ."""

    @pytest.mark.slow
    def test_two_agents_reach_equilibrium(self) -> None:
        cfg = SwarmConfig(
            num_agents=2, epsilon=80.0, sigma=12.0,
            damping=3.0, target_gain=0.0,
            max_force=1e6, max_velocity=50.0,
            world_size=1000.0, init_spread=0.01,
            steps=3000, dt=0.02, init_seed=42,
        )
        engine = SwarmEngine(cfg)
        engine.pos = np.array([[0.0, 0.0], [25.0, 0.0]])
        engine.vel = np.zeros((2, 2))

        for _ in range(cfg.steps):
            engine.step()

        actual = np.linalg.norm(engine.pos[0] - engine.pos[1])
        expected = cfg.lj_r_eq

        assert abs(actual - expected) / expected < 0.05, (
            f"Equilibrium distance {actual:.2f}m differs from "
            f"predicted {expected:.2f}m by more than 5%"
        )


class TestEnergyConservation:
    """Symplectic Euler should approximately conserve energy without damping."""

    @pytest.mark.slow
    def test_energy_bounded(
        self,
        undamped_engine: SwarmEngine,
        undamped_config: SwarmConfig,
    ) -> None:
        """Total energy should not drift more than 5% over 500 steps."""
        engine = undamped_engine

        # Let transients settle
        for _ in range(10):
            engine.step()
        E0 = engine.total_energy

        energies = []
        for _ in range(undamped_config.steps - 10):
            engine.step()
            energies.append(engine.total_energy)

        max_dev = max(abs(E - E0) for E in energies)
        rel_dev = max_dev / (abs(E0) + 1e-10)

        assert rel_dev < 0.05, (
            f"Energy drifted by {rel_dev:.1%} (E0={E0:.1f}, max_dev={max_dev:.1f})"
        )


class TestForceSaturation:
    """No force vector should exceed MAX_FORCE."""

    def test_forces_clipped(self, default_config: SwarmConfig) -> None:
        engine = SwarmEngine(default_config)

        # Place two agents dangerously close → huge LJ force
        engine.pos[0] = [0.0, 0.0]
        engine.pos[1] = [0.5, 0.0]

        forces = engine.compute_forces(engine.pos)
        max_norm = float(np.max(np.linalg.norm(forces, axis=1)))

        assert max_norm <= default_config.max_force + 1e-6, (
            f"Force {max_norm:.1f} exceeds MAX_FORCE {default_config.max_force}"
        )


class TestVelocitySaturation:
    """No velocity should exceed MAX_VELOCITY after a step."""

    def test_velocities_clipped(self) -> None:
        cfg = SwarmConfig(
            num_agents=2, max_velocity=10.0, max_force=1e6,
            init_spread=0.01, init_seed=42,
        )
        engine = SwarmEngine(cfg)
        # Place agents very close to generate massive acceleration
        engine.pos = np.array([[0.0, 0.0], [0.5, 0.0]])
        engine.vel = np.zeros((2, 2))

        engine.step()

        max_speed = float(np.max(np.linalg.norm(engine.vel, axis=1)))
        assert max_speed <= cfg.max_velocity + 1e-6, (
            f"Speed {max_speed:.1f} exceeds MAX_VELOCITY {cfg.max_velocity}"
        )


class TestBoundaryReflection:
    """Agents at world edges should reflect with damped velocity."""

    def test_agent_stays_in_bounds(self, default_config: SwarmConfig) -> None:
        engine = SwarmEngine(default_config)
        half = default_config.world_size / 2.0

        engine.pos[0] = [half + 10.0, 0.0]
        engine.vel[0] = [5.0, 0.0]

        engine.step()

        assert engine.pos[0, 0] <= half, (
            f"Agent escaped boundary: x={engine.pos[0, 0]:.1f}"
        )

    def test_agent_reflects_both_axes(self, default_config: SwarmConfig) -> None:
        engine = SwarmEngine(default_config)
        half = default_config.world_size / 2.0

        engine.pos[0] = [half + 5.0, -(half + 5.0)]
        engine.vel[0] = [3.0, -3.0]

        engine.step()

        assert engine.pos[0, 0] <= half
        assert engine.pos[0, 1] >= -half


class TestSensorNoise:
    """Sensor noise should perturb distances but not break the simulation."""

    def test_noise_runs_without_error(self) -> None:
        cfg = SwarmConfig(num_agents=10, sensor_noise=3.0, steps=50, init_seed=42)
        engine = SwarmEngine(cfg)

        for _ in range(cfg.steps):
            engine.step()

        assert np.all(np.isfinite(engine.pos)), "Positions contain NaN/Inf"
        assert np.all(np.isfinite(engine.vel)), "Velocities contain NaN/Inf"


class TestCommunicationDropout:
    """Dropout should reduce active interactions but not crash."""

    def test_dropout_runs_without_error(self) -> None:
        cfg = SwarmConfig(num_agents=10, dropout_rate=0.3, steps=50, init_seed=42)
        engine = SwarmEngine(cfg)

        for _ in range(cfg.steps):
            engine.step()

        assert np.all(np.isfinite(engine.pos)), "Positions contain NaN/Inf"
        assert np.all(np.isfinite(engine.vel)), "Velocities contain NaN/Inf"


class TestDiagnostics:
    """Engine diagnostic properties should return sensible values."""

    def test_mean_speed_non_negative(self, default_engine: SwarmEngine) -> None:
        default_engine.step()
        assert default_engine.mean_speed >= 0.0

    def test_spread_non_negative(self, default_engine: SwarmEngine) -> None:
        default_engine.step()
        assert default_engine.spread >= 0.0

    def test_centroid_shape(self, default_engine: SwarmEngine) -> None:
        assert default_engine.centroid.shape == (2,)

    def test_convergence_status_is_valid(self, default_engine: SwarmEngine) -> None:
        assert default_engine.convergence_status() in ("DEPLOYING", "TRACKING", "CONVERGED")

    def test_snapshot_keys(self, default_engine: SwarmEngine) -> None:
        default_engine.step()
        snap = default_engine.snapshot()
        expected = {"pos", "target", "history", "t", "spread", "speed", "status", "energy"}
        assert set(snap.keys()) == expected

    def test_total_energy_equals_sum(self, default_engine: SwarmEngine) -> None:
        default_engine.step()
        np.testing.assert_allclose(
            default_engine.total_energy,
            default_engine.kinetic_energy + default_engine.lj_potential_energy,
            atol=1e-6,
        )

    def test_kinetic_energy_non_negative(self, default_engine: SwarmEngine) -> None:
        default_engine.step()
        assert default_engine.kinetic_energy >= 0.0
