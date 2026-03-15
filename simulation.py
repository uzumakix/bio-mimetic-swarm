"""
simulation.py -- Decentralized maritime search simulation engine
=================================================================

Runs a fleet of autonomous surface vessels across an open water
coordinate plane. Each vessel uses only local vector math (modified
Boids + sweep heuristic) to decide its heading. There is no central
controller.

The simulation tracks a coverage grid to measure how efficiently
the fleet sweeps the search area over time.
"""

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
        Side length of the square search area.
    grid_resolution : float
        Cell size for coverage tracking.
    max_speed : float
        Maximum agent speed (units/timestep).
    perception_radius : float
        Agent sensing range.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, num_agents=50, grid_size=200.0,
                 grid_resolution=5.0, max_speed=2.0,
                 perception_radius=30.0, seed=42):
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
            offset = self.rng.normal(0, grid_size * 0.04, size=2)
            pos = np.array([center, center]) + offset
            heading = self.rng.uniform(0, 2 * np.pi)
            agent = SearchAgent(
                agent_id=i,
                position=pos,
                heading=heading,
                max_speed=max_speed,
                perception_radius=perception_radius,
            )
            self.agents.append(agent)

        self.coverage_history = []

    def step(self):
        """Advance the simulation by one timestep."""
        for agent in self.agents:
            agent.update(self.agents, self.coverage, self.grid_res)

            # boundary wrapping: keep agents in the search area
            agent.pos[0] = np.clip(agent.pos[0], 0, self.grid_size)
            agent.pos[1] = np.clip(agent.pos[1], 0, self.grid_size)

    def run(self, num_steps=1000):
        """
        Run the simulation for num_steps timesteps.

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
