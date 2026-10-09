---
pretty_name: ParSEL Edit Programs
language:
  - en
license: other
license_name: partnet-and-3dcompat-source-terms
license_link: https://huggingface.co/datasets/bardofcodes/parsel-edit-programs/blob/main/LICENSES.md
size_categories:
  - n<1K
tags:
  - 3d
  - shape-editing
  - natural-language
  - symbolic-computation
  - siggraph-asia-2024
---

# ParSEL: Parameterized Shape Editing with Language

**SIGGRAPH Asia 2024 · ACM Transactions on Graphics**

[Code](https://github.com/BardOfCodes/parsel) ·
[Project](https://bardofcodes.github.io/parsel/) ·
[Paper](https://doi.org/10.1145/3687922) ·
[arXiv](https://arxiv.org/abs/2405.20319)

This is an **output-only release** of parameterized edit programs for 102
curated language requests. Source meshes, textures, annotations, and processed
shape caches are not redistributed. You can inspect and validate the programs
without those inputs; Open3D replay requires compatible inputs obtained
separately under their source-dataset terms.

## Contents

| File | Contents |
| --- | --- |
| [parsel_edit_programs.zip](parsel_edit_programs.zip) | 102 saved programs, readable results, sanitized run logs, request CSVs, manifests, generation settings, and release provenance |
| [SHA256SUMS](SHA256SUMS) | SHA-256 checksum for the archive |
| [LICENSES.md](LICENSES.md) | License scope and upstream source-data terms |
| [CITATION.cff](CITATION.cff) | Machine-readable paper citation |

The requests comprise 52 PartNet edits and 50 3DCoMPaT++ edits, referring to
47 distinct PartNet shapes and 50 distinct 3DCoMPaT++ shapes. A program is an
adjustable symbolic edit, not an exported edited mesh.

**99 programs were regenerated with the cleaned implementation.** Three
are explicitly marked `archived_fallback`: `partnet:42`, `partnet:51`, and
`compat3d:38`. Per-example records and the manifest preserve this distinction.
Program bytes and program checksums are unchanged from the original output
set. Machine-specific paths in run logs are replaced with `<local-path>`;
`release.json` records the affected files and the source-manifest checksum.

The archive extracts into:

```text
parsel_clean_outputs/
├── metadata/                 # both request CSVs and field documentation
├── manifest.csv              # program index, checksums, and provenance
├── summary.json
├── generation.json
├── release.json
├── partnet/<row>_<shape-id>/
│   ├── program.pkl
│   ├── result.json
│   └── run.log
└── compat3d/<row>_<shape-id>/
    ├── program.pkl
    ├── result.json
    └── run.log
```

## Download and inspect

Install the [ParSEL code](https://github.com/BardOfCodes/parsel). Run from its
repository root:

```bash
python -m pip install -e .
python -m pip install --upgrade huggingface_hub

hf download bardofcodes/parsel-edit-programs \
    parsel_edit_programs.zip SHA256SUMS \
    --repo-type dataset --local-dir ../parsel_artifacts

cd ../parsel_artifacts
shasum -a 256 -c SHA256SUMS
unzip parsel_edit_programs.zip -d ..
cd ../parsel

python scripts/check_replay_outputs.py ../parsel_clean_outputs --show partnet:0
```

For a private or gated release, log in and complete its access process first.
On Linux, `sha256sum -c SHA256SUMS` can replace `shasum`. To pin a release,
add `--revision COMMIT_HASH` to `hf download`.

The checker validates all 102 programs and prints the selected readable
result. Select any `partnet:0..51` or `compat3d:0..49`. The JSON and CSV files
can also be read without unpickling. Only unpickle trusted artifacts; pickle
loading can execute Python code. This is an artifact archive, not a tabular
dataset intended for `datasets.load_dataset`.

## Open3D replay: separate inputs required

Obtain source assets through the official
[PartNet access process](https://huggingface.co/datasets/ShapeNet/PartNet-archive)
and [3DCoMPaT++ access process](https://3dcompat-dataset.org/doc/dl-dataset.html).
Follow [ParSEL's input guide](https://github.com/BardOfCodes/parsel/blob/main/docs/INPUTS.md)
for the expected processed-input layout and compatibility requirements.

Raw mesh downloads alone are insufficient: saved programs reference part and
relation indices from the original processed shape graph. The current release
does not provide a verified end-to-end reconstruction of all matching caches
from raw upstream downloads.

Once you have compatible processed inputs under `../parsel_experiment_data`:

```bash
python -m pip install -e ".[viewer]"
python scripts/check_replay_outputs.py \
    ../parsel_clean_outputs --show partnet:0 \
    --data-root ../parsel_experiment_data
```

The checker prints the viewer command. No API key or new LLM call is needed
to replay a saved program.

## Scope and licenses

This release excludes source meshes, textures, annotations, processed shape
caches, model checkpoints, provider credentials, and local visual-review
scores. Structural validation does not establish perceptual edit quality.
The curated requests are not a representative benchmark for arbitrary 3D
assets or language requests.

The data retains its source-specific terms:

- **PartNet subset (52 requests):** [PartNet / ShapeNet terms](https://huggingface.co/datasets/ShapeNet/PartNet-archive).
- **3DCoMPaT++ subset (50 requests):** [3DCoMPaT++ dataset agreement](https://docs.google.com/forms/d/e/1FAIpQLSfZMIZPnseH4WNkAnM-uPtd6S3xxOosHE9H8SgWf8RSFQ_BWg/viewform).

Both agreements limit use to non-commercial research and education and
require research associates and colleagues receiving data to accept the
corresponding terms. Per-example artifacts and request metadata follow their
source subset; combined manifests do not override either agreement.

The card uses `license: other` for these custom source-dataset terms; see
[LICENSES.md](LICENSES.md). No blanket MIT or Creative Commons data license is
added. ParSEL code and software documentation remain MIT licensed. Removing
meshes or gating access does not establish additional redistribution rights;
applicable permissions still need review before public publication.

## Citation

```bibtex
@article{ganeshan2024parsel,
  author    = {Ganeshan, Aditya and Huang, Ryan and Xu, Xianghao and
               Jones, R. Kenny and Ritchie, Daniel},
  title     = {{ParSEL}: Parameterized Shape Editing with Language},
  journal   = {ACM Transactions on Graphics},
  volume    = {43},
  number    = {6},
  articleno = {197},
  year      = {2024},
  doi       = {10.1145/3687922}
}
```

Also cite the source datasets when using their assets. Their official release
pages provide the corresponding citations.
