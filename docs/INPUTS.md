# Source inputs and replay compatibility

The Hugging Face release contains saved programs, not source meshes or
processed shape caches. You can read and validate its programs without inputs:

```bash
python scripts/check_replay_outputs.py ../parsel_clean_outputs --show partnet:0
```

## Obtain source assets

- **PartNet:** request access and accept the terms at the
  [official archive](https://huggingface.co/datasets/ShapeNet/PartNet-archive).
  `metadata/partnet_edits.csv` provides both the PartNet annotation ID
  (`shape_id`) and the corresponding ShapeNet model ID (`shapenet_model_id`).
- **3DCoMPaT++:** follow the
  [official download and license process](https://3dcompat-dataset.org/doc/dl-dataset.html).
  `metadata/compat3d_edits.csv` uses the original ParSEL processed shape IDs;
  these should not be assumed to be upstream asset filenames.

Keep upstream datasets outside the code repository. Their licenses and access
requirements remain applicable; the MIT software license does not replace them.
The existing local `parsel_experiment_data.zip` is not a public download.

## Compatible processed-input layout

For replay, you need the matching processed shape graph and its referenced
part meshes, arranged as:

```text
parsel_experiment_data/
├── partnet/<shape-id>/
│   ├── <shape-id>.pkl
│   └── objs/<referenced-part-mesh>.obj
└── compat3d/<shape-id>/
    ├── <shape-id>.pkl
    └── objs/<referenced-part-mesh>.glb
```

Each trusted cache stores the part representation, detected symmetry and
contact information, and category. Its mesh references must resolve within
the corresponding `objs/` folder. The dataset root and cache are separate
arguments to ParSEL's loaders and viewer.

**Matching filenames are not sufficient.** Saved programs refer to numeric
part and relation indices. Labels, ordering, hierarchy, proxy geometry, and
relation ordering must match the graph used to generate the program. Fresh
preprocessing can change these properties; do not treat a successful load as
proof of semantic compatibility.

The code includes raw PartNet preprocessing in
[`_partnet_preprocess.py`](../src/parsel/data/_partnet_preprocess.py), available
through explicit `load_shape(..., regenerate=True)`. That path needs the `io`
extra and is not a verified reconstruction of every released example. The
current release does not include a verified raw 3DCoMPaT++-to-matching-cache
pipeline or its original processed-ID mapping. For exact replay, obtain
compatible processed inputs through an authorized arrangement with the
[authors](mailto:adityaganeshan@gmail.com). This is a current reproducibility
limitation, not something the output-only download resolves.

## Replay with compatible inputs

Install the viewer extra and point the checker at your input root:

```bash
python -m pip install -e ".[viewer]"
python scripts/check_replay_outputs.py \
    ../parsel_clean_outputs --show partnet:0 \
    --data-root /path/to/parsel_experiment_data
```

The checker prints a viewer command. For example:

```bash
python scripts/visualize_shape.py \
    partnet \
    /path/to/parsel_experiment_data/partnet/13749/13749.pkl \
    /path/to/parsel_experiment_data/partnet \
    --program-file ../parsel_clean_outputs/partnet/000_13749/program.pkl
```

If you also have a local input-bundle manifest, check its payload with
`python scripts/validate_data_bundle.py /path/to/parsel_experiment_data`.
Checksum validation detects changed files; it does not establish rights or
compatibility with an independently regenerated shape graph.
