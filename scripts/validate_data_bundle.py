#!/usr/bin/env python3
"""Validate a ParSEL experiment-data archive or extracted directory."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


ARCHIVE_ROOT = "parsel_experiment_data"
BUFFER_SIZE = 1024 * 1024
MANIFEST_PATH = f"{ARCHIVE_ROOT}/manifest.json"


def _safe_archive_name(name):
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.parts
        and path.parts[0] == ARCHIVE_ROOT
    )


def _check_digest(stream, entry):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(BUFFER_SIZE):
        digest.update(chunk)
        size += len(chunk)
    if size != entry["size"]:
        raise ValueError(
            f"Size mismatch for {entry['path']}: "
            f"expected {entry['size']}, found {size}"
        )
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {entry['path']}")


def _validate_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing manifest schema_version")
    if manifest.get("archive_root") != ARCHIVE_ROOT:
        raise ValueError("Unexpected archive_root in manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest contains no file list")
    if manifest.get("file_count") != len(entries):
        raise ValueError("Manifest file_count does not match its file list")

    paths = [entry.get("path") for entry in entries]
    if (
        any(not isinstance(path, str) for path in paths)
        or len(paths) != len(set(paths))
        or any(not _safe_archive_name(path) for path in paths)
    ):
        raise ValueError("Manifest contains unsafe or duplicate paths")
    return entries


def validate_archive(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate paths")
        if any(not _safe_archive_name(name) for name in names):
            raise ValueError("Archive contains an unsafe path")
        try:
            manifest = json.loads(archive.read(MANIFEST_PATH))
        except KeyError as error:
            raise ValueError("Archive has no manifest.json") from error
        entries = _validate_manifest(manifest)
        expected = {entry["path"] for entry in entries} | {MANIFEST_PATH}
        if set(names) != expected:
            missing = sorted(expected - set(names))
            extra = sorted(set(names) - expected)
            raise ValueError(
                f"Archive contents differ from manifest; "
                f"missing={missing}, extra={extra}"
            )
        for index, entry in enumerate(entries, start=1):
            with archive.open(entry["path"]) as stream:
                _check_digest(stream, entry)
            if index % 250 == 0:
                print(f"Validated {index}/{len(entries)} files", flush=True)
    return manifest


def validate_directory(directory):
    root = directory
    if root.name != ARCHIVE_ROOT and (root / ARCHIVE_ROOT).is_dir():
        root = root / ARCHIVE_ROOT
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"No manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _validate_manifest(manifest)
    parent = root.parent.resolve()
    for index, entry in enumerate(entries, start=1):
        file_path = (parent / entry["path"]).resolve()
        if parent not in file_path.parents:
            raise ValueError(f"Unsafe manifest path: {entry['path']}")
        if not file_path.is_file():
            raise ValueError(f"Missing bundled file: {file_path}")
        with file_path.open("rb") as stream:
            _check_digest(stream, entry)
        if index % 250 == 0:
            print(f"Validated {index}/{len(entries)} files", flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundle",
        type=Path,
        help="the .zip archive or its extracted directory",
    )
    return parser.parse_args()


def main():
    path = parse_args().bundle.expanduser().resolve()
    try:
        if path.is_file():
            manifest = validate_archive(path)
        elif path.is_dir():
            manifest = validate_directory(path)
        else:
            raise ValueError(f"Bundle does not exist: {path}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    datasets = manifest["datasets"]
    print(
        "Bundle is valid: "
        f"{manifest['file_count']} files, "
        f"{datasets['partnet']['shape_count']} PartNet shapes, "
        f"{datasets['compat3d']['shape_count']} Compat3D shapes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
