"""
solve.py -- Run the maritime search simulation and generate plots
==================================================================

Spawns 50 autonomous surface vessels at a central launch point,
runs 1500 timesteps of decentralized search, and saves:
    results/trajectory_sweep.png  -- 2D search trajectories with coverage
    results/coverage_curve.png    -- grid coverage fraction over time
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation import SearchSimulation

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NUM_AGENTS = 50
NUM_STEPS = 1500
GRID_SIZE = 400.0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Simulating {NUM_AGENTS} agents for {NUM_STEPS} steps...")
    sim = SearchSimulation(
        num_agents=NUM_AGENTS,
        grid_size=GRID_SIZE,
        grid_resolution=5.0,
        max_speed=1.8,
        perception_radius=25.0,
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
    """2D plot of all agent search trajectories overlaid on coverage grid."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # coverage heatmap background
    extent = [0, sim.grid_size, 0, sim.grid_size]
    ax.imshow(
        sim.coverage, origin="lower", extent=extent,
        cmap="Blues", alpha=0.25, aspect="equal",
    )

    # agent trails with per-agent color
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    for agent in sim.agents:
        trail = np.array(agent.trail)
        c = palette[agent.agent_id % 20]
        ax.plot(trail[:, 0], trail[:, 1], color=c,
                linewidth=0.35, alpha=0.6)

    # launch point
    center = sim.grid_size / 2.0
    ax.plot(center, center, "r*", markersize=14, label="Launch point",
            zorder=10, markeredgecolor="darkred", markeredgewidth=0.5)

    # final positions
    fx = [a.pos[0] for a in sim.agents]
    fy = [a.pos[1] for a in sim.agents]
    ax.scatter(fx, fy, c="black", s=10, zorder=6,
               label="Final positions", edgecolors="white", linewidths=0.3)

    ax.set_xlim(0, sim.grid_size)
    ax.set_ylim(0, sim.grid_size)
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(
        f"Decentralized Search: {NUM_AGENTS} ASVs, "
        f"{NUM_STEPS} Timesteps, {GRID_SIZE:.0f}m Grid",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "trajectory_sweep.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved trajectory_sweep.png")


def plot_coverage_curve(history):
    """Grid coverage fraction over time with milestone annotations."""
    fig, ax = plt.subplots(figsize=(9, 5))

    steps = np.arange(1, len(history) + 1)
    pct = np.array(history) * 100

    ax.plot(steps, pct, color="#1d4ed8", linewidth=2.0)
    ax.fill_between(steps, 0, pct, color="#1d4ed8", alpha=0.08)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Grid Coverage (%)", fontsize=12)
    ax.set_title("Cumulative Search Coverage", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlim(0, len(history))
    ax.set_ylim(0, 105)

    # final coverage annotation
    final = pct[-1]
    ax.axhline(y=final, color="#b91c1c", linestyle="--",
               alpha=0.5, linewidth=0.8)
    ax.annotate(f"{final:.1f}%",
                xy=(steps[-1], final),
                xytext=(-70, 12), textcoords="offset points",
                fontsize=11, color="#b91c1c", fontweight="bold")

    # milestone markers
    for target in [25, 50, 75]:
        idx = np.searchsorted(pct, target)
        if idx < len(pct):
            ax.plot(steps[idx], pct[idx], "o", color="#1d4ed8",
                    markersize=5, zorder=5)
            ax.annotate(f"{target}% at t={steps[idx]}",
                        xy=(steps[idx], pct[idx]),
                        xytext=(10, -15), textcoords="offset points",
                        fontsize=8, color="#374151")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "coverage_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved coverage_curve.png")


if __name__ == "__main__":
    main()
