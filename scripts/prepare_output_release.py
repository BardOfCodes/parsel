#!/usr/bin/env python3
"""Prepare an output-only Hugging Face draft from trusted local programs."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from check_replay_outputs import REPOSITORY, load_manifest, sha256


ARCHIVE_NAME = "parsel_edit_programs.zip"
ARCHIVE_ROOT = "parsel_clean_outputs"
LOCAL_PATH = re.compile(r"(?:/home|/Users|/tmp|/private/var/folders)/[^\s\"'<>]+")
CREDENTIAL = re.compile(
    rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|"
    rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"
)


def prepare_release(source, destination):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/check_replay_outputs.py"), str(source)],
        check=True,
    )

    payload = {}
    sanitized_logs = []
    rows = load_manifest(source)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in {"partnet", "compat3d"} or not row["shape_id"].isdigit():
            raise ValueError("Unexpected dataset or shape ID in output manifest")
        folder = f"{dataset}/{int(row['row_index']):03d}_{row['shape_id']}"
        for field, filename in (("program_file", "program.pkl"), ("result_file", "result.json")):
            if row[field] != f"{folder}/{filename}":
                raise ValueError(f"Unexpected {field} path: {row[field]}")
        for filename in ("program.pkl", "result.json", "run.log"):
            path = source / folder / filename
            if not path.resolve().is_relative_to(source):
                raise ValueError(f"Output path escapes the source: {path}")
            contents = path.read_bytes()
            if filename == "run.log":
                text, count = LOCAL_PATH.subn("<local-path>", contents.decode("utf-8"))
                contents = text.encode("utf-8")
                if count:
                    sanitized_logs.append(f"{folder}/{filename}")
            payload[f"{folder}/{filename}"] = contents

    for filename in ("manifest.csv", "summary.json", "generation.json"):
        payload[filename] = (source / filename).read_bytes()
    for filename in ("README.md", "partnet_edits.csv", "compat3d_edits.csv"):
        payload[f"metadata/{filename}"] = (REPOSITORY / "metadata" / filename).read_bytes()
    payload["README.md"] = (
        "# ParSEL edit programs\n\n"
        f"This output-only release contains {len(rows)} saved edit programs, readable\n"
        "results, sanitized replay logs, request CSVs, and provenance. It does not\n"
        "include source meshes, textures, annotations, or processed shape caches.\n\n"
        "Validate with `python scripts/check_replay_outputs.py PATH_TO_THIS_FOLDER`\n"
        "from the ParSEL code repository. Open3D replay additionally requires\n"
        "separately obtained compatible inputs; see docs/INPUTS.md in that repository.\n\n"
        "Program bytes and checksums are unchanged from the original output set.\n"
        "Machine-specific paths in diagnostics are replaced with <local-path>.\n"
        "Only unpickle files from trusted sources.\n"
    ).encode("utf-8")
    payload["release.json"] = json.dumps(
        {
            "scope": "edit-programs-only",
            "row_count": len(rows),
            "source_manifest_sha256": sha256(source / "manifest.csv"),
            "sanitized_logs": sanitized_logs,
            "program_bytes_unchanged": True,
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    for name, contents in payload.items():
        if CREDENTIAL.search(contents):
            raise ValueError(f"Credential-like string in release file: {name}")

    destination.mkdir(parents=True)
    archive_path = destination / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in sorted(payload.items()):
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents)
    for filename in ("LICENSE", "LICENSES.md", "CITATION.cff"):
        shutil.copyfile(REPOSITORY / filename, destination / filename)
    shutil.copyfile(REPOSITORY / "docs/huggingface/README.md", destination / "README.md")
    checksum = f"{sha256(archive_path)}  {ARCHIVE_NAME}\n"
    (destination / "SHA256SUMS").write_text(checksum, encoding="utf-8")
    print(f"Prepared {len(rows)} programs; sanitized {len(sanitized_logs)} logs")
    print(f"Upload folder: {destination}")
    print(checksum, end="")
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted parsel_clean_outputs directory")
    parser.add_argument("destination", type=Path, help="new output-only upload directory")
    args = parser.parse_args()
    prepare_release(args.source, args.destination)


if __name__ == "__main__":
    main()
