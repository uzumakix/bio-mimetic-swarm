<p align="center">
  <img src="results/maritime_banner.jpg" width="720" />
</p>

# Decentralized Maritime Search Heuristics Engine

A spatial simulation engine for programming a fleet of autonomous surface vessels (ASVs) to sweep an open ocean grid using only local vector math and no central command. Each vessel computes its heading from neighbor positions and a coverage heuristic. The fleet self-organizes into an efficient search pattern through decentralized coordination alone.

The core question: given N autonomous aquatic drones with limited sensing range, how do you achieve near-total spatial coverage of a search area without any centralized controller?

## What the project finds

50 autonomous agents launched from a single point achieve **99.9% grid coverage** of a 400x400 meter search area in 1500 timesteps using only local sensing (25-meter radius). No agent has access to global state. The emergent search pattern fans outward from the launch point, with agents naturally partitioning space through separation pressure and sweep bias toward unvisited cells.

Coverage grows sublinearly: the fleet covers 25% by step 96, 50% by step 180, and 75% by step 292. Beyond 90% coverage, marginal returns diminish sharply as agents must traverse longer paths to reach isolated unvisited cells in corners and boundaries.

### Search Trajectories

<p align="center">
  <img src="results/trajectory_sweep.png" width="600" />
</p>

### Coverage Over Time

<p align="center">
  <img src="results/coverage_curve.png" width="600" />
</p>

## How it works

**1. The Agent Math (`agents.py`)**

Each vessel computes a steering vector from four components using only local information:

| Component | Weight | Purpose |
|-----------|--------|---------|
| Separation | 2.0 | Inverse-square repulsion within 8m to prevent collision |
| Alignment | 0.3 | Match heading with visible neighbors |
| Cohesion | 0.5 | Steer toward local center of mass |
| Sweep | 1.8 | Bias toward nearest unvisited grid cell (local window) |

All four vectors are normalized before weighting, so the weights directly control relative influence. The final velocity is clamped to a maximum speed of 1.8 m/timestep with a minimum of 0.3 m/timestep to prevent stalling. This extends the Boids model (Reynolds 1987) with a spatial coverage objective.

The sweep heuristic searches a local window of 8 grid cells around the agent and adds a small random offset to break ties. This prevents all agents from converging on the same target cell.

**2. The Simulation Engine (`simulation.py`)**

A kinematic loop spawns 50 agents at a central launch point with random initial headings. Each timestep uses a **two-pass update** to eliminate order-dependent behavior:
1. All agents compute their steering vectors from the current frozen state.
2. All agents apply their updates simultaneously.

A discrete coverage grid (80x80 cells over 400x400 meters) tracks which cells have been visited. Agents are bounded to the search area via position clamping.

**3. The Spatial Visualizer (`solve.py`)**

Runs the simulation for 1500 timesteps and generates two plots. The trajectory plot overlays all 50 agent paths on the coverage grid, showing the fan-out pattern from the launch point. The coverage curve tracks the fraction of cells visited over time, with milestone annotations at 25%, 50%, and 75%.

## Project structure

```
bio-mimetic-swarm/
    agents.py           # vector math: separation, alignment, cohesion, sweep
    simulation.py       # kinematic loop, coverage tracking, two-pass update
    solve.py            # runner, trajectory plot, coverage curve
    requirements.txt    # numpy, matplotlib
    results/
        trajectory_sweep.png   # 2D agent search paths
        coverage_curve.png     # grid coverage vs timestep
        maritime_banner.jpg    # header image (Unsplash, free license)
```

## Assumptions and limitations

This simulation makes several simplifying assumptions:

- **No ocean currents or drift.** Agents move in a static fluid. Real ASVs must compensate for tides, wind, and wave drift.
- **No communication latency.** Neighbor sensing is instantaneous within the perception radius. Real radio links introduce delay and packet loss.
- **Infinite energy.** Agents never run out of fuel or battery. Real missions must account for return-to-base constraints.
- **Flat 2D plane.** The search area has no obstacles, coastlines, or bathymetry. Real maritime environments are rarely featureless.
- **Fixed fleet size.** No agents are added or lost during the mission. Real operations must handle agent failure and dynamic redeployment.

Compared to structured algorithms (parallel sweep, Voronoi partitioning), the decentralized approach trades optimality for fault tolerance: no single agent failure can collapse the search plan. A formal comparison to these baselines is left as future work.

## Running

```bash
pip install -r requirements.txt
python solve.py
```

Results are saved to `results/`.

## References

1. Reynolds, C. W. (1987). "Flocks, Herds, and Schools: A Distributed Behavioral Model." *Computer Graphics (SIGGRAPH '87 Proceedings)*, 21(4), 25-34.

2. Koopman, B. O. (1980). *Search and Screening: General Principles with Historical Applications*. Pergamon Press. ISBN 978-0080231358.

3. Breivik, O. & Allen, A. A. (2008). "An Operational Search and Rescue Model for the Norwegian Sea and the North Sea." *Journal of Marine Systems*, 69(1-2), 99-113.

## License

MIT License. See [LICENSE](LICENSE).
