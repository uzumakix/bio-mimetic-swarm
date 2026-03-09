# Requirements Specification

## Functional Requirements

**FR-1: Agent population.**
The system shall simulate N agents on a 2D bounded rectangular domain, where N is configurable between 2 and 500 inclusive.

**FR-2: Lennard-Jones interaction.**
Agents shall interact through pairwise Lennard-Jones 12-6 potential fields. Configurable parameters: epsilon (well depth), sigma (equilibrium distance), and a force saturation limit.

**FR-3: Navigation tracking.**
Agents shall follow a time-varying navigation target that traces a Lissajous curve. The curve parameters (amplitudes, frequencies, phase offset) are set in the scenario configuration.

**FR-4: State output.**
The simulation shall produce position and velocity time series for all agents at every timestep, stored in memory for post-processing or export.

**FR-5: Animation rendering.**
The system shall render the simulation as a video animation. Supported output formats: MP4 (H.264) and GIF. Frame rate, resolution, trail length, and color scheme are configurable.

**FR-6: YAML configuration.**
All simulation parameters shall be specified via YAML scenario files. The system loads a named scenario, validates required fields, and applies defaults for optional fields.

**FR-7: Boundary handling.**
Agents reaching the domain boundary shall be reflected elastically. Velocity component normal to the boundary is negated, position is clamped.

**FR-8: Damping.**
A linear damping force proportional to agent velocity shall be applied to prevent unbounded kinetic energy growth. Damping coefficient is configurable.

## Performance Requirements

**PR-1: Simulation throughput.**
A 100-agent, 2000-step simulation shall complete in under 30 seconds wall time on a machine with 4 cores at 3.0 GHz or equivalent. This covers force computation and integration only, not rendering.

**PR-2: Playback frame rate.**
Rendered animations shall play back at 30 FPS or higher. The renderer must produce frames at this rate or batch them for encoding.

**PR-3: C kernel speedup.**
The C force computation kernel (forces_c.c) shall achieve at least 5x wall-time speedup over the pure Python fallback for simulations with N > 50 agents. Measured as time spent in the force function only.

**PR-4: Memory.**
Peak memory usage shall remain under 500 MB for a 500-agent, 5000-step simulation (positions + velocities + force arrays + render buffer).

## Interface Requirements

**IR-1: Command-line interface.**
The CLI shall accept a scenario name as positional argument and optional parameter overrides as key=value flags. Example: `python -m swarm run flock --n_agents=80`

**IR-2: Python API.**
The public API exposes:
- `SwarmConfig`: dataclass holding all simulation parameters, constructed from YAML or keyword arguments.
- `SwarmEngine`: accepts a `SwarmConfig`, runs the simulation via `.run()`, exposes position history via `.positions`.
- `render()`: function taking engine output and config, producing the animation file.

**IR-3: Output format selection.**
Output format is selectable via configuration (`output.format: mp4` or `output.format: gif`) or CLI flag (`--format gif`).

**IR-4: C library interface.**
The C force kernel is callable via ctypes. The Python wrapper (`forces_native.py`) handles loading, type conversion, and fallback transparently. No user action required beyond compiling the shared library.

## Traceability Matrix

| Requirement | Source Files |
|-------------|-------------|
| FR-1 | `src/swarm/engine.py`, `src/swarm/config.py` |
| FR-2 | `src/swarm/engine.py`, `src/swarm/forces_c.c`, `src/swarm/forces_native.py` |
| FR-3 | `src/swarm/engine.py` (navigation force, Lissajous target) |
| FR-4 | `src/swarm/engine.py` (position/velocity storage) |
| FR-5 | `src/swarm/renderer.py` |
| FR-6 | `src/swarm/config.py`, `scenarios/*.yaml` |
| FR-7 | `src/swarm/engine.py` (boundary reflection) |
| FR-8 | `src/swarm/engine.py` (damping term) |
| PR-1 | `src/swarm/engine.py`, `src/swarm/forces_c.c` |
| PR-3 | `src/swarm/forces_c.c`, `src/swarm/forces_native.py` |
| IR-1 | `src/swarm/__main__.py` |
| IR-2 | `src/swarm/config.py`, `src/swarm/engine.py`, `src/swarm/renderer.py` |
| IR-4 | `src/swarm/forces_c.c`, `src/swarm/forces_native.py` |
