# System Architecture

## Overview

The swarm simulator runs N autonomous agents on a bounded 2D plane. There is no central controller. Each agent computes its own motion from a sum of local interaction forces (Lennard-Jones repulsion/attraction between neighbors), a global navigation potential (spring toward a moving target), and boundary reflections. A time-stepping engine integrates these forces forward, and a renderer writes the resulting trajectories to MP4 or GIF.

## Block Diagram

```
                          scenario.yaml
                               |
                               v
                         +-----------+
                         | SwarmConfig|
                         +-----+-----+
                               |
                               v
  +----------+          +------+-------+         +----------+
  |  Forces  | <------> | SwarmEngine  | ------> | Renderer |
  +----------+          +------+-------+         +-----+----+
  | LJ pairs |                 |                       |
  | Nav pull  |          position/velocity         MP4 or GIF
  | Boundary  |          arrays per step
  | Damping   |
  +----------+
```

Configuration flows in from YAML. `SwarmConfig` validates parameters and passes them to `SwarmEngine`, which owns the integration loop. At each timestep the engine calls into the force module, updates state, and hands frames to the renderer.

## Agent State Machine

```
  INIT ----> EXPLORING ----> NAVIGATING ----> SETTLED
                 |                |
                 +--- AVOIDING ---+
```

**INIT**: Agent placed at random position with zero velocity. Transitions to EXPLORING on first timestep.

**EXPLORING**: Agent moves under LJ forces only, no navigation target yet. Transitions to NAVIGATING once the target path begins (configurable delay).

**NAVIGATING**: Navigation spring force pulls the agent toward the current target point on the Lissajous curve. Most agents spend the majority of the simulation here.

**AVOIDING**: Triggered when an agent's nearest neighbor is closer than the avoidance threshold. Overrides the navigation force with a pure repulsive escape vector. Returns to EXPLORING or NAVIGATING when spacing recovers.

**SETTLED**: Agent velocity drops below a threshold and stays there for a configurable number of steps. Terminal state. The agent still participates in LJ interactions but applies no self-propulsion.

## Force Model

Three forces act on each agent at each timestep:

**Lennard-Jones interaction.** Standard 12-6 potential between every pair:

```
U(r) = 4 * eps * ((sig/r)^12 - (sig/r)^6)
F(r) = 24 * eps * (2*(sig/r)^13 - (sig/r)^7) / r
```

The `sig` parameter sets equilibrium spacing and `eps` controls interaction strength. Forces are clamped to a saturation limit to prevent numerical blowup at small separations. A distance floor of 0.1 units avoids the singularity at r=0.

**Navigation force.** A spring pulling toward the current target position on the Lissajous curve:

```
F_nav = -k_nav * (pos - target)
```

Spring constant `k_nav` is small relative to LJ forces at equilibrium distance, so agents track the target collectively without breaking formation.

**Damping.** Linear drag proportional to velocity:

```
F_damp = -gamma * velocity
```

Prevents indefinite acceleration and controls the settling timescale.

## Integration

We use semi-implicit Euler (symplectic Euler):

```
v(t+dt) = v(t) + (F/m) * dt
x(t+dt) = x(t) + v(t+dt) * dt
```

Note that position uses the *updated* velocity. This matters because standard forward Euler (using v(t) for the position update) does not conserve energy in oscillatory systems and the swarm tends to heat up over long runs.

RK4 would give better accuracy per step, but at our typical dt (0.01 to 0.05) the error is already small enough that you cannot see it in the animation. The extra function evaluations (4x the force calls per step) are not worth it given that the force computation dominates runtime at O(N^2).

## Design Tradeoffs

**O(N^2) force computation vs spatial hashing.** The pairwise loop is simple and correct. For N < 200 agents it finishes in under a millisecond per step with the C kernel. Spatial hashing (grid or k-d tree) would bring this to O(N log N) or O(N) with a cutoff radius, but adds code complexity and only matters at larger N. The C extension already gives us the headroom we need. If someone wants to push past 500 agents, spatial partitioning is the next step.

**Synchronous vs asynchronous update.** All agents are updated simultaneously from the same force snapshot. The alternative (updating agents one at a time within a timestep) introduces order-dependent artifacts where early-updated agents "see" a different world than late-updated ones. Synchronous update costs one extra copy of the state arrays but eliminates that asymmetry.

**Fixed dt vs adaptive timestepping.** We use a fixed timestep. Adaptive methods (like RKF45) adjust dt based on local error estimates, which is great for stiff ODEs but causes headaches for animation: variable frame spacing means the renderer has to interpolate, and the output framerate becomes decoupled from the physics rate. Fixed dt keeps things simple. One physics step per rendered frame.
