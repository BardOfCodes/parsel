# Verification boundary

The suite began as a characterization harness against the current `edit_sys`
working tree at commit `5ff278741bd1a5d31b1641103d94d85a6012e596`.
That legacy baseline produced `28 passed, 1 xfailed`.

After the arrangement pass, the same tests produced the same result from the
new `parsel` package. The correction pass then converted the expected failure
and added focused regression checks. With the visualization test extra
installed, the portable suite result is:

```text
80 passed
```

Six additional real-data acceptance tests are opt-in because the datasets
are not stored in Git. They load one acceptance shape from PartNet and one
metadata-backed shape from Compat3D, verify both complete part/relation
graphs, build their visualizer geometry and visibility state, propagate a
symmetric leg-edit scenario on each, and fingerprint both source directories
to prove they were not modified. With both local datasets configured, the
combined result is:

```text
86 passed
```

The graph signatures and symmetric-edit outcomes were also checked against
the cached-data path in the legacy `edit_sys` implementation.

## Covered behavior

- geometry conventions and cuboid feature naming;
- part hierarchy lookup, traversal, activation, and cleanup;
- translation, rotation, scale, shear, restricted edits, and serialization;
- contact, reflection, translation symmetry, and rotation symmetry;
- symmetry count and spacing changes, including fixed-extent edits;
- proposal selection, typed candidate generation, symbolic solving, and
  proxy-to-mesh deformation;
- prompt assembly, edit/type/relation parsing, majority voting, and rejection
  of Python outside the edit DSL;
- OpenAI and Anthropic client construction, provider-specific request
  adaptation, isolated usage logging, and bounded workflow termination;
- import boundaries that keep LLM, dataset, and GUI dependencies optional;
- normalized PartNet and Compat3D evaluation metadata, including deterministic
  regeneration and preservation of documented source irregularities;
- visual geometry maps, per-part visibility, relation glyph edge cases, saved
  edit replay into displayed vertices, and GUI callback ordering;
- an offline natural-language workflow using recorded prompt decisions.

## Real-data suite

```bash
PARSEL_PARTNET_DATA_DIR=/path/to/partnet/data_v0 \
PARSEL_COMPAT3D_DATA_DIR=/path/to/compat/processed \
pytest -m real_data
```

The loader copies each cached shape into its own directory beneath
`.test-output/real_data/` before opening it. Edit programs are saved there as
well; both configured dataset trees are read-only from the test's perspective.
The separate data bundle includes both acceptance shapes and can be passed to
`scripts/run_tests.py` with `--partnet-data` and `--compat3d-data`.

## Visualization suite

```bash
python -m pip install -e ".[test,visualization-test]"
python scripts/run_tests.py --visualization
```

The automated suite is display-independent. Separate acceptance smokes
rendered PartNet shape `37807` and Compat3D shape `685` through Open3D 0.18
using EGL 1.5 / OpenGL 4.1, and constructed/closed the interactive PartNet
window under Xvfb through Open3D's event loop. The scripts copied the cache
inputs under `.test-output/` and verified that the source cache digests did
not change.

## Not yet automated

- live LLM-provider calls;
- regeneration from raw PartNet/Compat3D geometry with the full heavy
  dependency stack;
- user-driven Open3D widget interaction and mesh-export quality;
- real multiprocessing and API-accounting integration.

Prompt aggregation is tested with synchronous workers, and visualization's
headless deformation and scene-state core are covered. The local cached-dataset
loader is covered; redistributing a small shape fixture would make real
rendering portable to CI.
