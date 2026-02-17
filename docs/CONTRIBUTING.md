# Contributing to Bio-Mimetic Swarm

Thanks for your interest in contributing. Here's how to get started.

## Setup

```bash
git clone https://github.com/your-handle/bio-mimetic-swarm.git
cd bio-mimetic-swarm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development workflow

1. **Create a branch** from `main`
2. **Make your changes** — keep commits focused and atomic
3. **Run tests**: `pytest`
4. **Run lint**: `ruff check src/ tests/`
5. **Open a PR** against `main`

## Adding a scenario

1. Create a new YAML file in `scenarios/`
2. Use any `SwarmConfig` field name as a key
3. Include a descriptive `scenario_name` and `scenario_description`
4. Test it: `python -m swarm --scenario your-scenario`

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The CI will check this automatically. Run `ruff format src/ tests/` before committing.

## Testing

All physics changes should include corresponding tests. Key areas:

- **Force symmetry** — Newton's third law must hold
- **Energy conservation** — symplectic integrator must not drift (no-damping case)
- **Equilibrium spacing** — two agents must settle at r_eq = 2^(1/6) · σ
- **Boundary behaviour** — reflection must keep agents in bounds

## Reporting issues

Please include the output of `python -m swarm --version` and the full traceback if applicable.
