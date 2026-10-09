# Hugging Face output-only release

The output-only release is uploaded to
[`bardofcodes/parsel-edit-programs`](https://huggingface.co/datasets/bardofcodes/parsel-edit-programs)
as a **private draft**. Do not make it public until its scope and upstream
redistribution permissions have been reviewed.

Verified artifact upload commit:
[`171ee750e646ae2194fc7120170f9f573b394272`](https://huggingface.co/datasets/bardofcodes/parsel-edit-programs/commit/171ee750e646ae2194fc7120170f9f573b394272).
The archive downloaded back from Hugging Face matches the local SHA-256:
`8b9539167702dc2bfc32412771d3cfd4cbee03cfebb095c52ef6db6b502f82d5`.

Source-specific license notices were updated at
[`3475ad6c55480b10f1132b525aaae70ad2515385`](https://huggingface.co/datasets/bardofcodes/parsel-edit-programs/commit/3475ad6c55480b10f1132b525aaae70ad2515385).
The archive is unchanged at this revision; the download command below pins it.

The release contains edit programs and request metadata only. The local
`parsel_experiment_data.zip` includes source meshes and caches and must **not**
be uploaded as part of this release. Do not upload the older mixed staging
folder `.test-output/huggingface-release/` either.

## Prepare the upload folder

Run from the ParSEL code repository root using trusted saved outputs:

```bash
python scripts/prepare_output_release.py \
    /path/to/parsel_clean_outputs \
    .test-output/huggingface-outputs
```

The command refuses an existing destination. Choose a new directory when
rebuilding. It validates the saved programs, packages an explicit allowlist,
sanitizes machine-specific paths in run logs, and writes the archive checksum.
It does not modify the source outputs or input bundle.

The resulting upload directory contains exactly:

```text
huggingface-outputs/
├── README.md
├── LICENSE
├── LICENSES.md
├── CITATION.cff
├── SHA256SUMS
└── parsel_edit_programs.zip
```

The archive extracts into `parsel_clean_outputs/`. Program bytes and their
manifest checksums are unchanged. `release.json` records log sanitization.
Both request CSVs and their field documentation are included under `metadata/`.
No shape caches, source geometry, credentials, or local review scores are
included. Check the file list before uploading; never upload the whole local
experiment directory.

Verify the archive:

```bash
cd .test-output/huggingface-outputs
shasum -a 256 -c SHA256SUMS
cd ../..
```

The checked-in [SHA256SUMS](huggingface/SHA256SUMS) identifies the prepared
output-only archive. If the source outputs change, rebuild and update this
checksum before release. The source ZIPs and their previous checksums remain
untouched.

## Upload a private draft

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf repos create bardofcodes/parsel-edit-programs --repo-type dataset --private
hf upload bardofcodes/parsel-edit-programs .test-output/huggingface-outputs . \
    --repo-type dataset
```

See the [official CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
Review the rendered card and file list, and record the upload commit for pinned
downloads. If the repository already exists, inspect it rather than changing
its visibility or overwriting unrelated files.

ParSEL's code license remains MIT. The dataset metadata is `other`: the 52
PartNet requests retain PartNet / ShapeNet terms, and the 50 3DCoMPaT++ requests
retain the 3DCoMPaT++ dataset agreement. We do not add a blanket MIT or Creative
Commons data license. Enabling a gate would not grant upstream permissions.
See [LICENSES.md](../LICENSES.md) before public publication.

## Download and verify

From the code repository root, after signing in with access to the private
draft:

```bash
python -m pip install --upgrade huggingface_hub
hf download bardofcodes/parsel-edit-programs \
    parsel_edit_programs.zip SHA256SUMS \
    --revision 3475ad6c55480b10f1132b525aaae70ad2515385 \
    --repo-type dataset --local-dir ../parsel_artifacts

cd ../parsel_artifacts
shasum -a 256 -c SHA256SUMS
unzip parsel_edit_programs.zip -d ..
cd ../parsel

python scripts/check_replay_outputs.py ../parsel_clean_outputs --show partnet:0
```

On Linux, `sha256sum -c SHA256SUMS` is an alternative. Add
`--revision COMMIT_HASH` to pin a release. No input geometry or API key is
required for this inspection workflow.

Open3D replay needs separately obtained compatible processed inputs; see
[INPUTS.md](INPUTS.md). With inputs available, pass
`--data-root /path/to/parsel_experiment_data` to print the viewer command.
See [EVALUATION.md](EVALUATION.md) for full validation and review workflows.
