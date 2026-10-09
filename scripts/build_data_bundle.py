#!/usr/bin/env python3
"""Build the portable ParSEL experiment-data archive."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import subprocess
import sys
import zipfile


ARCHIVE_ROOT = "parsel_experiment_data"
BUFFER_SIZE = 1024 * 1024
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
PARTNET_ACCEPTANCE_ID = "37807"
REPOSITORY = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Bundle the cached shapes referenced by ParSEL's evaluation "
            "metadata. Source dataset trees are read-only."
        )
    )
    parser.add_argument(
        "--partnet-data",
        required=True,
        type=Path,
        help="root containing the curated PartNet shape directories",
    )
    parser.add_argument(
        "--partnet-acceptance-data",
        required=True,
        type=Path,
        help=(
            "PartNet root containing shape 37807, used by the real-data "
            "acceptance test"
        ),
    )
    parser.add_argument(
        "--compat3d-data",
        required=True,
        type=Path,
        help="root containing the processed Compat3D shape directories",
    )
    parser.add_argument(
        "--partnet-metadata",
        type=Path,
        default=REPOSITORY / "metadata" / "partnet_edits.csv",
    )
    parser.add_argument(
        "--compat3d-metadata",
        type=Path,
        default=REPOSITORY / "metadata" / "compat3d_edits.csv",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_shape_ids(metadata_path):
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    if not rows or "shape_id" not in rows[0]:
        raise ValueError(
            f"{metadata_path} must contain a shape_id column and at least "
            "one row"
        )

    shape_ids = []
    for row in rows:
        shape_id = row["shape_id"].strip()
        if not shape_id:
            raise ValueError(f"{metadata_path} contains an empty shape_id")
        if shape_id not in shape_ids:
            shape_ids.append(shape_id)
    return shape_ids


def _safe_name(name, description):
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {"", ".", ".."}
    ):
        raise ValueError(f"Unsafe {description}: {name!r}")
    return path.name


def referenced_mesh_names(cache_path):
    # These are trusted, locally generated ParSEL caches. Never call this
    # function on an untrusted pickle.
    with cache_path.open("rb") as cache_file:
        cached_data = pickle.load(cache_file)
    if not isinstance(cached_data, tuple) or not cached_data:
        raise ValueError(f"Unexpected cache structure: {cache_path}")

    model = cached_data[0]
    parts = model.values() if isinstance(model, dict) else model
    mesh_names = set()
    for part in parts:
        for raw_name in part["objs"]:
            if not isinstance(raw_name, str):
                raise ValueError(
                    f"Non-string mesh name in cache: {cache_path}"
                )
            mesh_name = raw_name if "." in raw_name else f"{raw_name}.obj"
            mesh_names.add(_safe_name(mesh_name, "mesh name"))
    if not mesh_names:
        raise ValueError(f"Cache references no meshes: {cache_path}")
    return sorted(mesh_names)


def shape_payload(source_root, shape_id):
    shape_id = _safe_name(str(shape_id), "shape id")
    shape_dir = source_root / shape_id
    cache_path = shape_dir / f"{shape_id}.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing shape cache: {cache_path}")

    payload = [(cache_path, f"{shape_id}/{shape_id}.pkl")]
    for mesh_name in referenced_mesh_names(cache_path):
        mesh_path = shape_dir / "objs" / mesh_name
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Cache {cache_path} references missing mesh {mesh_path}"
            )
        payload.append((mesh_path, f"{shape_id}/objs/{mesh_name}"))
    return payload


def _zip_info(archive_path):
    info = zipfile.ZipInfo(archive_path, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _write_bytes(archive, archive_path, contents):
    archive.writestr(_zip_info(archive_path), contents)
    return {
        "path": archive_path,
        "size": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def _write_file(archive, archive_path, source_path):
    digest = hashlib.sha256()
    size = 0
    with source_path.open("rb") as source:
        with archive.open(_zip_info(archive_path), "w") as destination:
            while chunk := source.read(BUFFER_SIZE):
                destination.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    return {
        "path": archive_path,
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def bundle_readme(partnet_count, compat3d_count):
    return f"""# ParSEL experiment data

This directory contains the processed inputs needed by the current ParSEL
repository: {partnet_count} PartNet shapes and {compat3d_count} Compat3D
shapes. The PartNet count includes shape {PARTNET_ACCEPTANCE_ID}, which is an
extra acceptance-test fixture outside the curated edit CSV.

The bundle contains each cached shape (`<id>.pkl`) and exactly the mesh files
referenced by that cache. It also includes copies of the readable evaluation
CSVs. Raw datasets, preprocessing intermediates, historical result folders,
API credentials, and legacy `edit_sys` program pickles are intentionally not
included.

After cloning ParSEL, create a fresh environment and run:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[test,real-data-test]"
.venv/bin/python scripts/validate_data_bundle.py /path/to/parsel_experiment_data
.venv/bin/python scripts/run_tests.py --real-data \\
  --partnet-data /path/to/parsel_experiment_data/partnet \\
  --compat3d-data /path/to/parsel_experiment_data/compat3d
```

`manifest.json` records every bundled file's size and SHA-256 digest. Dataset
inputs should be treated as read-only; experiment outputs belong in a separate
directory. Use of the underlying PartNet and Compat3D assets remains subject
to their original dataset licenses.
"""


def build_bundle(args):
    output_path = args.output.expanduser().resolve()
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if partial_path.exists():
        raise FileExistsError(
            f"Remove the incomplete archive before retrying: {partial_path}"
        )

    sources = {
        "partnet": args.partnet_data.expanduser().resolve(),
        "compat3d": args.compat3d_data.expanduser().resolve(),
    }
    acceptance_root = args.partnet_acceptance_data.expanduser().resolve()
    metadata = {
        "partnet": args.partnet_metadata.expanduser().resolve(),
        "compat3d": args.compat3d_metadata.expanduser().resolve(),
    }
    for path in [*sources.values(), acceptance_root]:
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in metadata.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation_ids = {
        name: read_shape_ids(path) for name, path in metadata.items()
    }
    partnet_ids = list(evaluation_ids["partnet"])
    if PARTNET_ACCEPTANCE_ID not in partnet_ids:
        partnet_ids.append(PARTNET_ACCEPTANCE_ID)
    bundle_ids = {
        "partnet": partnet_ids,
        "compat3d": evaluation_ids["compat3d"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_entries = []
    try:
        with zipfile.ZipFile(
            partial_path,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            readme = bundle_readme(
                len(bundle_ids["partnet"]),
                len(bundle_ids["compat3d"]),
            ).encode("utf-8")
            file_entries.append(
                _write_bytes(
                    archive,
                    f"{ARCHIVE_ROOT}/README.md",
                    readme,
                )
            )
            for dataset_name, metadata_path in metadata.items():
                file_entries.append(
                    _write_file(
                        archive,
                        (
                            f"{ARCHIVE_ROOT}/metadata/"
                            f"{dataset_name}_edits.csv"
                        ),
                        metadata_path,
                    )
                )

            for dataset_name, shape_ids in bundle_ids.items():
                for index, shape_id in enumerate(shape_ids, start=1):
                    source_root = sources[dataset_name]
                    if (
                        dataset_name == "partnet"
                        and shape_id == PARTNET_ACCEPTANCE_ID
                        and shape_id not in evaluation_ids["partnet"]
                    ):
                        source_root = acceptance_root
                    for source_path, relative_path in shape_payload(
                        source_root, shape_id
                    ):
                        file_entries.append(
                            _write_file(
                                archive,
                                (
                                    f"{ARCHIVE_ROOT}/{dataset_name}/"
                                    f"{relative_path}"
                                ),
                                source_path,
                            )
                        )
                    print(
                        f"[{dataset_name}] {index}/{len(shape_ids)} "
                        f"shape {shape_id}",
                        flush=True,
                    )

            commit, dirty = _git_state()
            manifest = {
                "schema_version": 1,
                "archive_root": ARCHIVE_ROOT,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repository": {
                    "url": "https://github.com/BardOfCodes/parsel.git",
                    "commit": commit,
                    "working_tree_dirty": dirty,
                },
                "datasets": {
                    "partnet": {
                        "evaluation_shape_ids": evaluation_ids["partnet"],
                        "acceptance_shape_ids": [PARTNET_ACCEPTANCE_ID],
                        "shape_count": len(bundle_ids["partnet"]),
                    },
                    "compat3d": {
                        "evaluation_shape_ids": evaluation_ids["compat3d"],
                        "acceptance_shape_ids": [],
                        "shape_count": len(bundle_ids["compat3d"]),
                    },
                },
                "files": sorted(
                    file_entries, key=lambda item: item["path"]
                ),
                "file_count": len(file_entries),
                "uncompressed_size": sum(
                    item["size"] for item in file_entries
                ),
            }
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            archive.writestr(
                _zip_info(f"{ARCHIVE_ROOT}/manifest.json"),
                manifest_bytes,
            )
        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    archive_digest = hashlib.sha256()
    with output_path.open("rb") as archive_file:
        while chunk := archive_file.read(BUFFER_SIZE):
            archive_digest.update(chunk)
    print(f"Created {output_path}")
    print(f"Archive SHA-256: {archive_digest.hexdigest()}")


def main():
    try:
        build_bundle(parse_args())
    except (OSError, ValueError, KeyError, pickle.UnpicklingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
