"""
simulation.py -- Decentralized maritime search simulation engine
=================================================================

Runs a fleet of autonomous surface vessels across an open water
coordinate plane. Each vessel uses only local vector math (modified
Boids + sweep heuristic) to decide its heading. There is no central
controller.

The simulation uses a two-pass update to avoid order-dependent
behavior: all agents compute their steering vectors from the
current state, then all agents apply their updates simultaneously.
"""

import math

import numpy as np

from agents import SearchAgent


class SearchSimulation:
    """
    Kinematic simulation of decentralized maritime search.

    Parameters
    ----------
    num_agents : int
        Number of autonomous surface vessels.
    grid_size : float
        Side length of the square search area (meters).
    grid_resolution : float
        Cell size for coverage tracking (meters).
    max_speed : float
        Maximum agent speed (meters/timestep).
    perception_radius : float
        Agent sensing range (meters).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, num_agents=50, grid_size=400.0,
                 grid_resolution=5.0, max_speed=1.8,
                 perception_radius=25.0, seed=42):
        self.grid_size = grid_size
        self.grid_res = grid_resolution
        self.rng = np.random.default_rng(seed)

        # coverage grid: 0 = unvisited, 1 = visited
        grid_cells = int(grid_size / grid_resolution)
        self.coverage = np.zeros((grid_cells, grid_cells), dtype=int)

        # spawn agents in a cluster near the center (launch point)
        center = grid_size / 2.0
        self.agents = []
        for i in range(num_agents):
            offset = self.rng.normal(0, grid_size * 0.03, size=2)
            pos = np.clip(
                np.array([center, center]) + offset,
                0.0, grid_size - 1e-6,
            )
            heading = self.rng.uniform(0, 2 * np.pi)
            agent = SearchAgent(
                agent_id=i,
                position=pos,
                heading=heading,
                max_speed=max_speed,
                perception_radius=perception_radius,
            )
            self.agents.append(agent)

        # mark initial cells as visited
        for agent in self.agents:
            gx = int(math.floor(agent.pos[0] / grid_resolution))
            gy = int(math.floor(agent.pos[1] / grid_resolution))
            if 0 <= gx < grid_cells and 0 <= gy < grid_cells:
                self.coverage[gy, gx] = 1

        self.coverage_history = []

    def step(self):
        """
        Advance the simulation by one timestep.

        Two-pass approach to eliminate order-dependent updates:
        1. All agents compute steering from the current frozen state.
        2. All agents apply their updates simultaneously.
        """
        # pass 1: compute all steering vectors
        accels = []
        for agent in self.agents:
            accel = agent.compute_steering(
                self.agents, self.coverage, self.grid_res, self.rng,
            )
            accels.append(accel)

        # pass 2: apply all updates
        for agent, accel in zip(self.agents, accels):
            agent.apply_steering(
                accel, self.grid_size, self.coverage, self.grid_res,
            )

    def run(self, num_steps=1000):
        """
        Run the full simulation.

        Returns
        -------
        coverage_history : list of float
            Fraction of grid covered at each timestep.
        """
        total_cells = self.coverage.size

        for t in range(num_steps):
            self.step()
            covered = np.sum(self.coverage) / total_cells
            self.coverage_history.append(covered)

            if (t + 1) % 200 == 0:
                print(f"  Step {t + 1:>5d}  coverage={covered:.1%}")

        return self.coverage_history
