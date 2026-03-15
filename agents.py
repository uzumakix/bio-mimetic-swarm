"""
agents.py -- Vector math for decentralized maritime search agents
==================================================================

Each agent (autonomous surface vessel) computes its next velocity
from local neighborhood information only. No central controller
exists. The steering rules are:

    1. Separation: repel from nearby agents to avoid collision
    2. Alignment:  match heading with neighbors
    3. Cohesion:   steer toward local center of mass
    4. Sweep:      bias toward unvisited grid cells (search heuristic)

The resulting velocity is a weighted sum of these four vectors,
clamped to a maximum speed. This is a modified Boids model
(Reynolds 1987) extended with a spatial coverage objective.
"""

import numpy as np


class SearchAgent:
    """
    Single autonomous surface vessel with local-only sensing.

    Parameters
    ----------
    agent_id : int
    position : ndarray of shape (2,)
        Initial (x, y) in the coordinate plane.
    heading : float
        Initial heading in radians.
    max_speed : float
        Maximum velocity magnitude (units/timestep).
    perception_radius : float
        Range within which other agents are visible.
    """

    def __init__(self, agent_id, position, heading=0.0,
                 max_speed=2.0, perception_radius=30.0):
        self.id = agent_id
        self.pos = np.array(position, dtype=float)
        self.vel = np.array([np.cos(heading), np.sin(heading)],
                            dtype=float) * max_speed * 0.5
        self.max_speed = max_speed
        self.radius = perception_radius
        self.trail = [self.pos.copy()]

    def neighbors(self, all_agents):
        """Return list of agents within perception radius."""
        nbrs = []
        for other in all_agents:
            if other.id == self.id:
                continue
            dist = np.linalg.norm(other.pos - self.pos)
            if dist < self.radius and dist > 0:
                nbrs.append(other)
        return nbrs

    def separation(self, nbrs, min_dist=8.0):
        """Steer away from agents that are too close."""
        steer = np.zeros(2)
        for n in nbrs:
            diff = self.pos - n.pos
            d = np.linalg.norm(diff)
            if d < min_dist and d > 0:
                steer += diff / (d * d)
        return steer

    def alignment(self, nbrs):
        """Match velocity direction with neighbors."""
        if not nbrs:
            return np.zeros(2)
        avg_vel = np.mean([n.vel for n in nbrs], axis=0)
        return avg_vel - self.vel

    def cohesion(self, nbrs):
        """Steer toward the center of mass of neighbors."""
        if not nbrs:
            return np.zeros(2)
        center = np.mean([n.pos for n in nbrs], axis=0)
        return center - self.pos

    def sweep_bias(self, coverage_grid, grid_res):
        """
        Steer toward the nearest unvisited grid cell.

        The coverage grid tracks which cells have been visited.
        The agent biases its heading toward the closest cell
        that has not yet been marked as covered.
        """
        gx = int(self.pos[0] / grid_res)
        gy = int(self.pos[1] / grid_res)
        h, w = coverage_grid.shape

        best_dir = np.zeros(2)
        best_dist = float("inf")

        # search in a local window around the agent
        search_r = 6
        for dx in range(-search_r, search_r + 1):
            for dy in range(-search_r, search_r + 1):
                cx, cy = gx + dx, gy + dy
                if 0 <= cx < w and 0 <= cy < h:
                    if coverage_grid[cy, cx] == 0:
                        cell_center = np.array(
                            [(cx + 0.5) * grid_res,
                             (cy + 0.5) * grid_res]
                        )
                        dist = np.linalg.norm(cell_center - self.pos)
                        if dist < best_dist:
                            best_dist = dist
                            best_dir = cell_center - self.pos

        norm = np.linalg.norm(best_dir)
        if norm > 0:
            best_dir = best_dir / norm
        return best_dir

    def update(self, all_agents, coverage_grid, grid_res, dt=1.0,
               w_sep=1.5, w_ali=0.3, w_coh=0.4, w_sweep=1.8):
        """
        Compute the weighted steering vector and update position.

        Parameters
        ----------
        w_sep, w_ali, w_coh, w_sweep : float
            Weights for separation, alignment, cohesion, and sweep.
        dt : float
            Timestep.
        """
        nbrs = self.neighbors(all_agents)

        accel = (
            w_sep * self.separation(nbrs)
            + w_ali * self.alignment(nbrs)
            + w_coh * self.cohesion(nbrs)
            + w_sweep * self.sweep_bias(coverage_grid, grid_res)
        )

        self.vel = self.vel + accel * dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed

        self.pos = self.pos + self.vel * dt
        self.trail.append(self.pos.copy())

        # mark current cell as visited
        gx = int(self.pos[0] / grid_res)
        gy = int(self.pos[1] / grid_res)
        h, w = coverage_grid.shape
        if 0 <= gx < w and 0 <= gy < h:
            coverage_grid[gy, gx] = 1
