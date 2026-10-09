# License scope and source datasets

## ParSEL code

The ParSEL implementation and its accompanying software documentation are
licensed under the [MIT License](LICENSE). Third-party dependencies retain
their own licenses.

## Experiment inputs

The MIT software license does not replace the terms governing source meshes,
textures, annotations, or their processed caches in
`parsel_experiment_data.zip`.

- **PartNet / ShapeNet:** consult the
  [official PartNet archive and terms](https://huggingface.co/datasets/ShapeNet/PartNet-archive).
  The published terms limit use to non-commercial research and education and
  restrict redistribution to research associates and colleagues who first
  agree to those terms. Acceptance of the upstream terms is required.
- **3DCoMPaT++:** consult the
  [official dataset agreement](https://docs.google.com/forms/d/e/1FAIpQLSfZMIZPnseH4WNkAnM-uPtd6S3xxOosHE9H8SgWf8RSFQ_BWg/viewform)
  and [download process](https://3dcompat-dataset.org/doc/dl-dataset.html).
  The agreement limits use to non-commercial research and education and
  permits access for research associates and colleagues only after they
  agree to those terms. A software license in the dataset's code repository
  is not a substitute for asset permissions.

Processing these datasets into ParSEL caches does not grant additional rights
to the source assets. Keep these distinctions when sharing the input archive.

## Evaluation outputs

The output-only `parsel_edit_programs.zip` contains ParSEL edit programs,
readable result records, request metadata, and sanitized replay diagnostics.
It excludes source meshes, textures, annotations, and processed shape caches.
The existing `parsel_experiment_data.zip` is kept separately for local use and
is not part of the Hugging Face upload.

The release preserves the upstream terms separately for each source subset:

| Output subset | Requests | Applicable source-data terms |
| --- | --- | --- |
| `partnet/` and `metadata/partnet_edits.csv` | 52 | PartNet / ShapeNet terms linked above |
| `compat3d/` and `metadata/compat3d_edits.csv` | 50 | 3DCoMPaT++ dataset agreement linked above |

Per-example programs, results, and logs follow their corresponding subset.
Combined manifests and summaries cover both subsets and do not override either
agreement. We do not relicense the data under MIT or a blanket Creative Commons
license. ParSEL code and software documentation remain MIT licensed.

Recipients must comply with the applicable upstream agreement, including its
use and sharing restrictions. Removing source geometry does not by itself
establish additional redistribution rights for derived artifacts. Review the
applicable permissions before public publication; this notice does not grant
rights beyond the upstream terms. Source-data terms also apply when obtaining
inputs, replaying programs with source meshes, or distributing derived geometry.

## Hugging Face metadata

The output-only card uses `license: other` and links to this notice because
the two source subsets have their own custom dataset agreements. Access gating
is not a substitute for permission. The code repository and Python package
use MIT metadata.
