# Contributing

Thank you for helping improve ParSEL. Please open an issue before starting a
large behavioral change so that the intended scope and evaluation can be
agreed upon.

## Development setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python scripts/run_tests.py --portable
```

Install `.[viewer]` when changing Open3D code and run:

```bash
python scripts/run_tests.py --visualization
```

## Design expectations

- Keep the symbolic core independent of datasets, LLM clients, and Open3D.
- Prefer small, explicit APIs over compatibility aliases or implicit writes.
- Treat source datasets and released artifacts as read-only.
- Add a focused regression test for every bug fix.
- Keep optional dependencies behind their subsystem import boundaries.

The detailed module boundaries are documented in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Pull requests

Describe the behavior being changed, the commands used to test it, and any
data or provider requirements that a reviewer cannot reproduce offline. Do not
commit API keys, local dataset paths, generated outputs, or pickle artifacts.
