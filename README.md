<p align="center">
  <img src="assets/banner.svg" alt="Bio-Mimetic Swarm" width="800"/>
</p>

<p align="center">
  <strong>Decentralized swarm intelligence for autonomous ocean monitoring</strong><br/>
  <em>50 vessels. Zero central controllers. Pure physics.</em>
</p>

<p align="center">
  <a href="#quickstart"><img src="https://img.shields.io/badge/python-3.10+-3776ab?logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22d3ee" alt="MIT License"/></a>
  <a href="#testing"><img src="https://img.shields.io/badge/tests-pytest-06b6d4?logo=pytest" alt="Tests"/></a>
  <a href="#"><img src="https://img.shields.io/badge/no_GPU-required-64748b" alt="No GPU required"/></a>
</p>

---

## The Problem

Every year, oil spills contaminate thousands of square kilometres of ocean before response teams can even map their boundaries. Traditional monitoring relies on satellites (slow, weather-dependent) or manned vessels (expensive, dangerous, can't be everywhere at once). By the time a response fleet arrives, the slick has drifted, fragmented, and the damage is done.

**What if 50 small, cheap autonomous boats could deploy in minutes — and figure out the formation themselves?**

No central command server. No GPS coordination network. No radio communication infrastructure. Each vessel responds only to what it can sense locally: how close are my nearest neighbours? Where's the target? That's it.

The result? Self-organizing flocking, collision avoidance, and coordinated target tracking — all emerging from the same physics that governs how atoms arrange themselves in a crystal. Nature solved this problem billions of years ago. We're borrowing the solution.

<p align="center">
  <img src="swarm_behavior.gif" alt="Swarm simulation — 50 ASVs tracking a drifting target" width="750"/>
</p>

---

## How It Works

The control law is borrowed directly from molecular physics. Each vessel is governed by two forces:

**1. Lennard-Jones Potential** — the same equation that describes how noble gas atoms attract at a distance and repel when too close. This creates a natural equilibrium spacing (~13.5 m) without any explicit formation logic:

$$U_{LJ}(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

**2. Navigation Potential** — a simple spring force pulling every vessel toward the target (the drifting oil spill, modelled as a Lissajous trajectory):

$$\mathbf{F}_{nav} = k_{att} \left( \mathbf{x}_{target} - \mathbf{x}_{i} \right)$$

No vessel communicates with any other vessel. No vessel knows the global swarm state. The collective behaviour — spacing, cohesion, pursuit — emerges from physics alone.

> **Why not just use a central controller?**
> A central system is a single point of failure. If the command server goes down, every vessel stops. If a radio link drops, a vessel is lost. A decentralised swarm degrades gracefully — lose 10 vessels, and the remaining 40 reorganise automatically. That's how fish schools survive predators. That's how we build resilient ocean infrastructure.

---

## Quickstart

### Prerequisites

- Python 3.10+
- No GPU required — runs on any laptop in under 3 minutes

### Install

```bash
git clone https://github.com/your-handle/bio-mimetic-swarm.git
cd bio-mimetic-swarm
pip install -e ".[dev]"
```

### Run

```bash
# Default simulation — 50 vessels, oil spill tracking
python -m swarm

# Customize from the command line
python -m swarm --agents 100 --epsilon 40 --sigma 8

# Use a named scenario
python -m swarm --scenario search-rescue
python -m swarm --scenario tight-crystal
python -m swarm --scenario loose-flock

# Add real-world noise
python -m swarm --sensor-noise 2.0 --dropout 0.15

# Export as MP4 instead of GIF
python -m swarm --format mp4
```

### Output

```
╔═══════════════════════════════════════════════════════════╗
║  bio-mimetic-swarm v2.0  ·  APF Simulation Engine         ║
║  Scenario: oil-spill  ·  50 ASVs  ·  ε=80  σ=12m         ║
╚═══════════════════════════════════════════════════════════╝

[1/3] Deploying swarm from staging area ...
        0.0%  |  t=  0.0s  |  spread=56.3m  |  v̄=0.29 m/s  |  DEPLOYING
       50.0%  |  t= 24.0s  |  spread=18.1m  |  v̄=3.47 m/s  |  TRACKING
      100.0%  ✓  Mission complete.

[2/3] Rendering animation ...
[3/3] Saved → swarm_oil-spill.gif  (600 frames, 30 fps)

╔═══════════════════════════════════════════════════════════╗
║  ✓ Complete — open swarm_oil-spill.gif                     ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Scenarios

The `--scenario` flag loads curated parameter sets that demonstrate different emergent behaviours:

| Scenario | Description | What to watch for |
|---|---|---|
| `oil-spill` | Default. 50 ASVs tracking a drifting Lissajous target | Formation crystallises, then pursues as a rigid body |
| `search-rescue` | Wide-area grid sweep with high target gain | Swarm stretches into a search line, then collapses on target |
| `tight-crystal` | High ε, low σ — strong bonds, tight spacing | Beautiful hexagonal lattice, very stable |
| `loose-flock` | Low ε, high σ — weak bonds, wide spacing | Organic, bird-like flocking with stragglers |
| `stress-test` | 200 agents, sensor noise, 15% dropout | Tests robustness under realistic conditions |

You can also create your own scenario files in `scenarios/` (YAML format).

---

## Real-World Realism

This isn't just a pretty animation. The simulation includes features that matter for actual deployment:

### Sensor Noise (`--sensor-noise <std_dev>`)
Real vessels don't have perfect position knowledge. Enable Gaussian noise on neighbour distance measurements to see how the formation degrades (spoiler: it's surprisingly robust up to ~3 m noise).

### Communication Dropout (`--dropout <fraction>`)
In real ocean environments, radio links fail. Setting `--dropout 0.15` randomly masks 15% of pairwise interactions each timestep. The LJ lattice holds together far better than a centralized system would.

### Heterogeneous Fleet
Mix vessel types by editing the scenario YAML. Give some agents different `sigma` values — larger boats need bigger exclusion zones. The swarm accommodates automatically.

---

## What Emerges

The simulation produces three distinct behavioural phases — none of which are explicitly programmed:

### Phase 1 — Deployment (t = 0–8 s)
Agents start as a Gaussian cloud with random separations. Vessels too close experience violent repulsion (the r⁻¹² term). Vessels too far feel gentle attraction. The result: rapid disordering followed by structured self-organisation.

### Phase 2 — Crystallisation (t = 8–20 s)
The LJ potential drives every vessel toward the equilibrium spacing r_eq ≈ 13.5 m. A loose 2D hexagonal lattice forms — the same geometry that governs noble gas solids. This happens because it's the minimum-energy configuration of the potential field.

### Phase 3 — Target Tracking (t > 20 s)
The navigation force overwhelms LJ cohesion for distant targets. The crystallised formation translates as a near-rigid body, maintaining internal spacing while pursuing the Lissajous trajectory. This mirrors how fish schools follow a food gradient.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  __main__.py        CLI + ORCHESTRATION                          │
│  ─────────────────────────────────────────────────────────────── │
│  argparse CLI  ·  scenario loading  ·  engine → renderer pipe    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  renderer.py        VISUALISATION LAYER                          │
│  ─────────────────────────────────────────────────────────────── │
│  Dark-mode matplotlib  ·  HUD overlay  ·  GIF/MP4 export        │
│  Reads: engine snapshots (pos, vel, target, metrics)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  engine.py          PHYSICS SIMULATION                           │
│  ─────────────────────────────────────────────────────────────── │
│  SwarmEngine class  ·  vectorised LJ forces  ·  symplectic Euler │
│  Sensor noise injection  ·  communication dropout                │
│  Reads: SwarmConfig                                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  config.py          CONFIGURATION                                │
│  ─────────────────────────────────────────────────────────────── │
│  SwarmConfig frozen dataclass  ·  YAML scenario loader           │
│  All physics, numerics, and rendering parameters                 │
└─────────────────────────────────────────────────────────────────┘
```

Each layer depends only on the layer below it. Changing `SIGMA` from 12 m to 8 m requires touching exactly one value — the engine and renderer are untouched.

---

## Mathematical Foundation

<details>
<summary><strong>Click to expand full derivation</strong></summary>

### Agent Dynamics

Each ASV is modelled as a double-integrator point mass with linear damping:

$$\ddot{\mathbf{x}}_i = \mathbf{u}_i - k_d \dot{\mathbf{x}}_i$$

where **x**_i ∈ ℝ² is the position of agent i, k_d is the damping coefficient, and **u**_i is the control input defined as the negative gradient of the total potential field:

$$\mathbf{u}_i = -\nabla_{\mathbf{x}_i} U_{\text{total}}(\mathbf{x}_i)$$

### Lennard-Jones Inter-Agent Potential

$$U_{LJ}(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

The r⁻¹² term dominates at short range (hard-body repulsion, preventing collisions). The r⁻⁶ term dominates at long range (cohesive attraction, preventing dispersion). The potential minimum — the natural equilibrium spacing — occurs at:

$$r_{eq} = 2^{1/6} \cdot \sigma \approx 1.122 \, \sigma$$

The pairwise force is:

$$\mathbf{F}_{ij} = \frac{24\epsilon}{r} \left( 2\frac{\sigma^{12}}{r^{12}} - \frac{\sigma^{6}}{r^{6}} \right) \hat{\mathbf{r}}_{ij}$$

### Navigation Potential

$$U_{att}(\mathbf{x}_i) = \frac{1}{2} k_{att} \left\| \mathbf{x}_i - \mathbf{x}_{target}(t) \right\|^2$$

$$\mathbf{F}_{att,i} = k_{att} \left( \mathbf{x}_{target}(t) - \mathbf{x}_i \right)$$

### Total Control Law

$$\mathbf{u}_i = \underbrace{-\sum_{j \neq i} \nabla U_{LJ}(r_{ij})}_{\text{swarm forces}} + \underbrace{k_{att}(\mathbf{x}_{target} - \mathbf{x}_i)}_{\text{navigation}} - \underbrace{k_d \dot{\mathbf{x}}_i}_{\text{damping}}$$

### Semi-Implicit Euler Integration

$$\mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \cdot \Delta t$$
$$\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_{n+1} \cdot \Delta t$$

Using the updated velocity v_{n+1} in the position update preserves the symplectic structure of the Hamiltonian, preventing artificial energy drift.

</details>

---

## Tuning Guide

| Parameter | Effect of ↑ | Effect of ↓ | Default |
|---|---|---|---|
| `epsilon` | Stiffer lattice, stronger bonds | Looser swarm, dispersion risk | 80 |
| `sigma` | Wider equilibrium spacing | Tighter packed formation | 12 m |
| `target-gain` | Faster tracking, less stable lattice | Sluggish tracking, stable crystal | 1.2 |
| `damping` | Overdamped — slow, no oscillation | Underdamped — oscillatory | 2.0 |
| `agents` | Richer collective behaviour | Faster simulation | 50 |
| `sensor-noise` | Realistic degradation | Perfect conditions | 0.0 m |
| `dropout` | Tests robustness | Full connectivity | 0.0 |

> **Stability note:** The LJ r⁻¹² term diverges as r → 0. Force saturation (MAX_FORCE = 120 N) and the symplectic integrator prevent blow-up. If you increase epsilon substantially, decrease dt proportionally.

---

## Testing

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=swarm --cov-report=term-missing

# Run specific test categories
pytest tests/test_physics.py -v    # Physics invariant tests
pytest tests/test_config.py -v     # Configuration & scenario tests
```

### What the tests verify

- **Energy conservation** — with no damping or target, total energy stays within 1% over 1000 steps
- **Equilibrium spacing** — two agents settle at r_eq = 2^(1/6) · σ within 5% tolerance
- **Newton's third law** — F_ij = −F_ji for all agent pairs
- **Boundary reflection** — agents at world edges reflect with damped velocity
- **Force saturation** — no force vector exceeds MAX_FORCE
- **Scenario loading** — all YAML scenarios parse and produce valid configs

---

## Project Structure

```
bio-mimetic-swarm/
├── src/swarm/
│   ├── __init__.py          # Package metadata & version
│   ├── __main__.py          # CLI entry point (python -m swarm)
│   ├── config.py            # SwarmConfig dataclass + scenario loader
│   ├── engine.py            # Physics: LJ forces, symplectic Euler, noise
│   └── renderer.py          # Dark-mode matplotlib animation + export
├── tests/
│   ├── test_physics.py      # Energy conservation, force symmetry, equilibrium
│   ├── test_config.py       # Config validation, scenario loading
│   └── conftest.py          # Shared fixtures
├── scenarios/
│   ├── oil_spill.yaml       # Default: drifting target tracking
│   ├── search_rescue.yaml   # Wide-area grid sweep
│   ├── tight_crystal.yaml   # High ε demo
│   ├── loose_flock.yaml     # Low ε organic flocking
│   └── stress_test.yaml     # 200 agents + noise + dropout
├── docs/
│   └── math.md              # Extended mathematical derivations
├── assets/
│   └── banner.svg           # Project banner
├── .github/workflows/
│   └── ci.yml               # GitHub Actions: lint + test
├── pyproject.toml            # Modern Python packaging
├── LICENSE                   # MIT
├── .gitignore
└── README.md                 # This file
```

---

## Future Work

- **3D ocean current model** — stochastic drift field as exogenous disturbance
- **Graph Laplacian consensus** — limited-range radio as an additional coordination force
- **Obstacle avoidance** — repulsive Gaussian potentials at reefs and shipping lanes
- **ROS 2 bridge** — export agent states as `nav_msgs/Odometry` for hardware-in-the-loop testing
- **Streamlit dashboard** — interactive web UI for parameter exploration

---

## References

- Reynolds, C. W. (1987). *Flocks, herds and schools: A distributed behavioral model.* ACM SIGGRAPH.
- Olfati-Saber, R. (2006). *Flocking for multi-agent dynamic systems.* IEEE Trans. Automatic Control.
- Lennard-Jones, J. E. (1924). *On the determination of molecular fields.* Proc. Royal Society.
- Koditschek, D. E. & Rimon, E. (1990). *Robot navigation functions on manifolds with boundary.* Advances in Applied Mathematics.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built with the conviction that the best swarm algorithms were invented by nature,<br/>and the best engineering is knowing when to borrow.</em>
</p>
