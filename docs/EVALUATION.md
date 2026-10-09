# Evaluation and output replay

The release workflow separates code, experiment inputs, and generated outputs:

- the repository contains implementation, readable metadata, and scripts;
- `parsel_experiment_data` contains the processed shape inputs; and
- `parsel_clean_outputs` contains saved edit programs and provenance records.

The output set has 102 rows: 52 PartNet requests and 50 3DCoMPaT++ requests.
Only the output-only `parsel_edit_programs.zip` is staged for Hugging Face.
Source meshes and processed caches are excluded; see [INPUTS.md](INPUTS.md)
for separately obtained compatible replay inputs.

See [the Hugging Face guide](HUGGING_FACE.md) for artifact hosting, downloads,
and archive checksums, and [LICENSES.md](../LICENSES.md) for source-data terms.

> [!WARNING]
> Shape caches and saved programs are Python pickle files. Validate checksums
> and load only artifacts from sources you trust.

## Validate the release

Validate the outputs without needing shape inputs:

```bash
python scripts/check_replay_outputs.py ../parsel_clean_outputs --show partnet:0
```

If you have a local input bundle with a manifest, validate it separately:

```bash
python scripts/validate_data_bundle.py ../parsel_experiment_data
```

With compatible inputs available, print one result and its viewer command:

```bash
python scripts/check_replay_outputs.py \
    ../parsel_clean_outputs \
    --show partnet:0 \
    --data-root ../parsel_experiment_data
```

The checker verifies the manifest status, JSON record, program checksum, and
serialized edit structure for every row. Select any `partnet:0..51` or
`compat3d:0..49` with `--show`.

## Inspect one result

```bash
python scripts/visualize_shape.py \
    partnet \
    ../parsel_experiment_data/partnet/13749/13749.pkl \
    ../parsel_experiment_data/partnet \
    --program-file ../parsel_clean_outputs/partnet/000_13749/program.pkl
```

The viewer copies the cache into an isolated output directory before loading
it and verifies that neither source artifact changed during the session.

## Review the complete set

The interactive reviewer skips rows that already have a saved verdict and
writes the CSV after every decision:

```bash
python scripts/review_replay_outputs.py \
    ../parsel_clean_outputs \
    ../parsel_experiment_data
```

Close each Open3D window after exploring the edit slider, then enter one of:

- `g`: good;
- `m`: mid;
- `b`: bad;
- `s`: skipped; or
- `q`: save the existing work and quit.

The default review file is `.test-output/replay_review.csv`. Use
`--start compat3d:12` to start at a specific row. A file lock prevents two
reviewers from writing the same review file simultaneously.

## Rebuild outputs from archived decisions

Replay the paper-era prompting decisions through the cleaned implementation
without making an LLM request:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/replay_archived_outputs.py \
    --partnet-data /path/to/partnet/shapes \
    --compat3d-data /path/to/compat/processed \
    --partnet-decisions /path/to/partnet/ours_pred \
    --compat3d-decisions /path/to/compat3d/ours_pred \
    --output /path/to/parsel_clean_outputs \
    --workers 8
```

The replay is resumable with `--resume`, uses deterministic numeric-check
seeds, and records a checksum and readable JSON representation beside every
saved program.

If bounded replay cannot regenerate a row, trusted paper program archives can
fill only the failed rows. These rows are explicitly marked
`archived_fallback` in the manifest:

```bash
python scripts/import_archived_fallbacks.py \
    /path/to/parsel_clean_outputs \
    --partnet-archive /path/to/PreLoadV3_5_final.zip \
    --compat3d-archive /path/to/UpfrontParallelLLMPrompterV3_5_final.zip
```

## Build a portable data bundle

This is a local packaging workflow, **not** the public Hugging Face release.
The resulting archive includes source meshes; do not redistribute it without
confirming the upstream permissions. For output-only packaging, use
`scripts/prepare_output_release.py` as described in [HUGGING_FACE.md](HUGGING_FACE.md).

```bash
python scripts/build_data_bundle.py \
    --partnet-data /path/to/curated/partnet/shapes \
    --partnet-acceptance-data /path/to/partnet/data_v0 \
    --compat3d-data /path/to/compat/processed \
    --output /path/to/parsel_experiment_data.zip
```

The builder follows cache dependencies, includes only required cache and mesh
files, records checksums and repository state, and never writes into the source
dataset trees.

## Test scopes

Portable tests require no datasets or Open3D:

```bash
python scripts/run_tests.py --portable
```

Viewer tests require the `viewer` extra:

```bash
python scripts/run_tests.py --visualization
```

Real-data acceptance tests require the extracted bundle:

```bash
python scripts/run_tests.py --real-data \
    --partnet-data ../parsel_experiment_data/partnet \
    --compat3d-data ../parsel_experiment_data/compat3d
```

The tests cover the domain model, edit operations, contact and symmetry
relations, symbolic solving, edit propagation, safe prompt parsing, cage
deformation, serialization, viewer state, output replay, and an offline
end-to-end workflow. Raw-data regeneration, live provider behavior, and
subjective mesh quality remain manual acceptance concerns.

## Verified release check

On 2026-10-09, a fresh Python 3.10.15 virtual environment on macOS ARM64,
created from `coref` without inheriting its installed packages, passed:

- installation of the core, `test`, and `viewer` extras, with `pip check` clean;
- a wheel build and non-editable install, importing from `site-packages` outside
  the repository;
- all **96 tests**: 73 portable, 16 visualization, and 7 local-data acceptance;
- authenticated browser downloads of the output ZIP and checksums pinned to
  Hugging Face revision `3475ad6c55480b10f1132b525aaae70ad2515385`, with matching
  SHA-256 and **102 valid programs, zero errors**; and
- local input-bundle validation: 2,025 payload files, 48 PartNet shapes, and
  50 3DCoMPaT++ shapes, unchanged against their manifest.

The interactive-window smoke check **failed** on macOS 26.6.2: the fresh
Open3D 0.18.0 wheel crashed in its native macOS OpenGL rendering path. GLFW also
reported `Cocoa: Failed to find service port for display`, but that warning
alone does not establish the cause; [other Open3D reports](https://github.com/isl-org/Open3D/issues/5945)
describe it with working windows. A working interactive window in this fresh
stack has not been verified, and a desktop-terminal check is still required
before claiming GUI compatibility. Headless Open3D checks passed. These checks
do not validate raw-input reconstruction, live LLM calls, or the visual quality
of all 102 edits.
