# Installation and configuration

ParSEL separates its symbolic core from optional data, prompting, and Open3D
dependencies. Install only the extras required by your workflow.

## Requirements

- Python 3.10 or newer for the core package
- Python 3.10 recommended for the tested Open3D 0.18 stack
- macOS and Linux have been used; see the [verified release check](EVALUATION.md#verified-release-check)
  for tested scopes and the current macOS GUI limitation
- no GPU required for the symbolic edit system or cached-shape viewer

Create an isolated environment from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate the environment with `.venv\Scripts\activate`. The
interactive batch reviewer currently requires the Unix `fcntl` module.

## Install matrix

### Core symbolic system

This installs NumPy, SciPy, and SymPy. It is sufficient for constructing shape
relations, edit programs, and running Analytical Edit Propagation on data that
is already in memory.

```bash
python -m pip install -e .
```

### Cached-shape viewer

This adds Open3D and PyTorch and is the smallest install that can load compatible
cached inputs, render shapes, and replay saved programs.

```bash
python -m pip install -e ".[viewer]"
```

### Natural-language prompting

This adds the OpenAI and Anthropic clients. Combine it with `viewer` for the
end-to-end cached-shape workflow.

```bash
python -m pip install -e ".[viewer,prompting]"
```

### Raw mesh I/O and complete visualization

The `visualization` extra adds mesh-processing packages used by the full app.
The `io` extra additionally includes raw PartNet preprocessing dependencies,
including PyTorch3D. These packages can require platform-specific wheels.

```bash
python -m pip install -e ".[prompting,visualization]"
python -m pip install -e ".[io]"  # only for raw-data regeneration
```

Existing compatible cached shapes do not require raw-data regeneration or the
`io` extra. The output-only Hugging Face release does not include shape caches.

### Development

```bash
python -m pip install -e ".[test]"
python scripts/run_tests.py --portable
```

The portable suite excludes local datasets and Open3D. To test the viewer:

```bash
python -m pip install -e ".[test,viewer]"
python scripts/run_tests.py --visualization
```

## Release data layout

The output archive will be hosted in a Hugging Face dataset repository. Follow
[the download and checksum instructions](HUGGING_FACE.md#download-and-verify)
before extracting it. The private draft is
[`bardofcodes/parsel-edit-programs`](https://huggingface.co/datasets/bardofcodes/parsel-edit-programs);
sign in with an account that has access.

Extract `parsel_edit_programs.zip` next to the repository. It creates
`parsel_clean_outputs/`. To use the viewer, obtain compatible inputs separately
following [INPUTS.md](INPUTS.md); the resulting local layout is:

```text
workspace/
├── parsel/
├── parsel_clean_outputs/
│   ├── manifest.csv
│   ├── partnet/
│   └── compat3d/
└── parsel_experiment_data/
    ├── manifest.json
    ├── partnet/<shape-id>/<shape-id>.pkl
    └── compat3d/<shape-id>/<shape-id>.pkl
```

Validate the saved programs without input geometry:

```bash
python scripts/check_replay_outputs.py ../parsel_clean_outputs --show partnet:0
```

With compatible local inputs, print the viewer command:

```bash
python scripts/check_replay_outputs.py \
    ../parsel_clean_outputs \
    --show partnet:0 \
    --data-root ../parsel_experiment_data
```

If your local input directory has a bundle manifest, validate it with
`python scripts/validate_data_bundle.py ../parsel_experiment_data` as well.
Programs and caches contain pickle files; load only trusted artifacts.

## LLM configuration

The prompting workflow reads the following environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARSEL_LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `PARSEL_LLM_API_KEY` | unset | provider API key |
| `PARSEL_LLM_MODEL` | `gpt-4o` | provider model name |
| `PARSEL_LLM_TEMPERATURE` | `0` | sampling temperature |
| `PARSEL_LLM_SEED` | `42` | sampling seed when supported |
| `PARSEL_API_LOG` | `parsel_api_calls.db` | local API-usage database |

For example:

```bash
export PARSEL_LLM_PROVIDER=openai
export PARSEL_LLM_API_KEY="your-api-key"
export PARSEL_LLM_MODEL=gpt-4o
```

Do not place API keys in source files, command history, or committed `.env`
files. `scripts/solve_edit.py --api-key-file PATH` can read a key from a
separately protected file.

To exercise propagation without an API call, provide a seed edit:

```bash
python scripts/solve_edit.py \
    compat3d 1 ../parsel_experiment_data/compat3d \
    --seed-edit 'Translate(shape.get("leg_front_left"), RightUnitVector())'
```

The part label must exist on the selected shape. Repeat `--keep-fixed PART`
for parts that should not move during propagation.

## Headless rendering

Render a cached shape to PNG:

```bash
python scripts/render_shape.py \
    partnet \
    ../parsel_experiment_data/partnet/13749/13749.pkl \
    ../parsel_experiment_data/partnet
```

On a headless Linux host, select EGL before Python imports Open3D:

```bash
EGL_PLATFORM=surfaceless python scripts/render_shape.py \
    partnet \
    ../parsel_experiment_data/partnet/13749/13749.pkl \
    ../parsel_experiment_data/partnet
```

## Common problems

### `ModuleNotFoundError: open3d`

Install the viewer extra in the active environment:

```bash
python -m pip install -e ".[viewer]"
```

### The viewer opens without edit controls

Pass a valid saved program with `--program-file`. Without one,
`visualize_shape.py` intentionally runs as a shape inspector.

### An existing output is not overwritten

Scripts refuse to overwrite results by default. Choose another output path or,
where supported, pass `--force` explicitly.

### PyTorch3D installation fails

PyTorch3D is only needed for raw-data regeneration. Use separately obtained
compatible caches instead, or install a wheel compatible with Python, PyTorch, CUDA, and
platform versions before installing the `io` extra.
