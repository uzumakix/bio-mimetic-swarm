"""
solve.py -- Run the maritime search simulation and generate plots
==================================================================

Spawns 50 autonomous surface vessels at a central launch point,
runs 1000 timesteps of decentralized search, and saves:
    results/trajectory_sweep.png  -- 2D search trajectories
    results/coverage_curve.png    -- grid coverage over time
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from simulation import SearchSimulation

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NUM_AGENTS = 50
NUM_STEPS = 1000
GRID_SIZE = 200.0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Simulating {NUM_AGENTS} agents for {NUM_STEPS} steps...")
    sim = SearchSimulation(
        num_agents=NUM_AGENTS,
        grid_size=GRID_SIZE,
        grid_resolution=5.0,
        max_speed=2.0,
        perception_radius=30.0,
        seed=42,
    )
    coverage_history = sim.run(NUM_STEPS)

    final_coverage = coverage_history[-1]
    print(f"\nFinal grid coverage: {final_coverage:.1%}")
    print(f"Agents deployed: {NUM_AGENTS}")

    plot_trajectories(sim)
    plot_coverage_curve(coverage_history)

    print(f"\nPlots saved to {RESULTS_DIR}/")


def plot_trajectories(sim):
    """2D plot of all agent search trajectories over the grid."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # draw coverage heatmap as background
    extent = [0, sim.grid_size, 0, sim.grid_size]
    ax.imshow(
        sim.coverage, origin="lower", extent=extent,
        cmap="Blues", alpha=0.3, aspect="equal",
    )

    # color palette for agent trails
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for agent in sim.agents:
        trail = np.array(agent.trail)
        c = colors[agent.id % 20]
        ax.plot(trail[:, 0], trail[:, 1], color=c,
                linewidth=0.5, alpha=0.7)

    # mark launch point
    center = sim.grid_size / 2.0
    ax.plot(center, center, "r*", markersize=12, label="Launch point")

    # mark final positions
    final_x = [a.pos[0] for a in sim.agents]
    final_y = [a.pos[1] for a in sim.agents]
    ax.scatter(final_x, final_y, c="black", s=8, zorder=5,
               label="Final positions")

    ax.set_xlim(0, sim.grid_size)
    ax.set_ylim(0, sim.grid_size)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(
        f"Search Trajectories: {NUM_AGENTS} Agents, {NUM_STEPS} Steps",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "trajectory_sweep.png"), dpi=150)
    plt.close(fig)
    print("Saved trajectory_sweep.png")


def plot_coverage_curve(history):
    """Grid coverage fraction over time."""
    fig, ax = plt.subplots(figsize=(9, 5))

    steps = list(range(1, len(history) + 1))
    pct = [h * 100 for h in history]

    ax.plot(steps, pct, color="#2563eb", linewidth=2.0)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Grid Coverage (%)", fontsize=12)
    ax.set_title("Cumulative Search Coverage", fontsize=14)
    ax.grid(True, alpha=0.3)

    # annotate final coverage
    final = pct[-1]
    ax.axhline(y=final, color="#dc2626", linestyle="--",
               alpha=0.5, linewidth=0.8)
    ax.annotate(f"{final:.1f}%",
                xy=(steps[-1], final),
                xytext=(-60, 10), textcoords="offset points",
                fontsize=11, color="#dc2626", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "coverage_curve.png"), dpi=150)
    plt.close(fig)
    print("Saved coverage_curve.png")


if __name__ == "__main__":
    main()
