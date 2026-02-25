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

The resulting velocity is a weighted sum of these four normalized
steering vectors, clamped to a maximum speed. This is a modified
Boids model (Reynolds 1987) extended with a spatial coverage
objective for maritime search applications.
"""

import math
import numpy as np

_EPS = 1e-10


def _normalize(v, max_mag=1.0):
    """Normalize vector to at most max_mag. Returns zero vector if input is near-zero."""
    mag = np.linalg.norm(v)
    if mag < _EPS:
        return np.zeros(2)
    return v / mag * min(mag, max_mag)


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
        self.agent_id = agent_id
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
            if other.agent_id == self.agent_id:
                continue
            dist = np.linalg.norm(other.pos - self.pos)
            if 0 < dist < self.radius:
                nbrs.append(other)
        return nbrs

    def separation(self, nbrs, min_dist=8.0):
        """
        Steer away from agents that are too close.
        Uses inverse-square repulsion: force ~ 1/d^2.
        """
        steer = np.zeros(2)
        for n in nbrs:
            diff = self.pos - n.pos
            d = np.linalg.norm(diff)
            if d < min_dist and d > _EPS:
                # unit direction * 1/d^2 gives inverse-square repulsion
                steer += (diff / d) * (1.0 / (d * d))
            elif d <= _EPS:
                # agents at same point: repel in random direction
                angle = np.random.uniform(0, 2 * np.pi)
                steer += np.array([np.cos(angle), np.sin(angle)]) * (1.0 / (_EPS * _EPS))
        return _normalize(steer)

    def alignment(self, nbrs):
        """Match velocity direction with neighbors. Returns normalized steering vector."""
        if not nbrs:
            return np.zeros(2)
        avg_vel = np.mean([n.vel for n in nbrs], axis=0)
        return _normalize(avg_vel - self.vel)

    def cohesion(self, nbrs):
        """Steer toward the center of mass of neighbors. Returns normalized steering vector."""
        if not nbrs:
            return np.zeros(2)
        center = np.mean([n.pos for n in nbrs], axis=0)
        return _normalize(center - self.pos)

    def sweep_bias(self, coverage_grid, grid_res, rng):
        """
        Steer toward a nearby unvisited grid cell.

        Uses a local search window around the agent's current cell.
        When multiple unvisited cells are equidistant, a small random
        offset breaks ties to prevent all agents converging on the
        same target cell.
        """
        gx = int(math.floor(self.pos[0] / grid_res))
        gy = int(math.floor(self.pos[1] / grid_res))
        h, w = coverage_grid.shape

        best_dir = np.zeros(2)
        best_dist = float("inf")

        # search in a local window around the agent
        search_r = 8
        for dx in range(-search_r, search_r + 1):
            for dy in range(-search_r, search_r + 1):
                cx, cy = gx + dx, gy + dy
                if 0 <= cx < w and 0 <= cy < h:
                    if coverage_grid[cy, cx] == 0:
                        cell_center = np.array(
                            [(cx + 0.5) * grid_res,
                             (cy + 0.5) * grid_res]
                        )
                        # add small random offset for tie-breaking
                        jitter = rng.uniform(-0.5, 0.5, size=2) * grid_res * 0.3
                        target = cell_center + jitter
                        dist = np.linalg.norm(target - self.pos)
                        if dist < best_dist:
                            best_dist = dist
                            best_dir = target - self.pos

        return _normalize(best_dir)

    def compute_steering(self, all_agents, coverage_grid, grid_res, rng,
                         w_sep=2.0, w_ali=0.3, w_coh=0.5, w_sweep=1.8):
        """
        Compute the weighted steering acceleration vector.

        All component vectors are normalized before weighting,
        so the weights directly control relative influence.

        Returns the acceleration vector (not yet applied).
        """
        nbrs = self.neighbors(all_agents)

        accel = (
            w_sep * self.separation(nbrs)
            + w_ali * self.alignment(nbrs)
            + w_coh * self.cohesion(nbrs)
            + w_sweep * self.sweep_bias(coverage_grid, grid_res, rng)
        )
        return accel

    def apply_steering(self, accel, grid_size, coverage_grid, grid_res,
                       dt=1.0, min_speed=0.3):
        """
        Apply precomputed steering to update velocity and position.

        Enforces boundary clamping and minimum speed to prevent
        stalling.
        """
        self.vel = self.vel + accel * dt

        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        elif speed < min_speed and speed > _EPS:
            self.vel = self.vel / speed * min_speed

        self.pos = self.pos + self.vel * dt

        # clamp to grid bounds (use epsilon to keep grid index valid)
        self.pos[0] = np.clip(self.pos[0], 0.0, grid_size - 1e-6)
        self.pos[1] = np.clip(self.pos[1], 0.0, grid_size - 1e-6)

        self.trail.append(self.pos.copy())

        # mark current cell as visited
        gx = int(math.floor(self.pos[0] / grid_res))
        gy = int(math.floor(self.pos[1] / grid_res))
        h, w = coverage_grid.shape
        if 0 <= gx < w and 0 <= gy < h:
            coverage_grid[gy, gx] = 1
