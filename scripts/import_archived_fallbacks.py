#!/usr/bin/env python3
"""Fill failed replay rows from provenance-tracked paper program archives."""

import argparse
import csv
import io
import json
from pathlib import Path
import pickle
import zipfile

from replay_archived_outputs import (
    LOADABLE_STATUSES,
    atomic_json,
    atomic_pickle,
    display_value,
    sha256,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--partnet-archive", type=Path, required=True)
    parser.add_argument("--compat3d-archive", type=Path, required=True)
    return parser.parse_args()


def load_rows(output):
    with (output / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))

    rows = []
    for manifest_row in manifest_rows:
        result_path = output / manifest_row["result_file"]
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        result.setdefault("source", "regenerated")
        result.setdefault("archive_file", "")
        result.setdefault("archive_member", "")
        result.setdefault("archive_sha256", "")
        rows.append(result)
    return rows


def load_legacy_program(archive, row_index):
    from parsel.edit_dsl import edits as edit_module

    class LegacyProgramUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "edit_sys.shape_system.edits":
                return getattr(edit_module, name)
            return super().find_class(module, name)

    with zipfile.ZipFile(archive) as handle:
        suffix = f"/programs_{row_index}.pkl"
        members = [name for name in handle.namelist() if name.endswith(suffix)]
        if len(members) != 1:
            raise ValueError(
                f"Expected one {suffix} member in {archive}, found {members}"
            )
        member = members[0]
        program = LegacyProgramUnpickler(
            io.BytesIO(handle.read(member))
        ).load()
    return program, member


def program_signature(program):
    return [
        [generator.edit_class.__name__, operand_type, operand_index]
        for generator, operand_type, operand_index in program
        if generator.edit_class.__name__ != "KeepFixed"
    ]


def describe_program(program):
    return [
        {
            "edit": generator.edit_class.__name__,
            "operand_type": operand_type,
            "operand_index": operand_index,
            "operand": "",
            "amount": display_value(generator.amount),
            "parameters": {
                key: display_value(value)
                for key, value in generator.param_dict.items()
            },
        }
        for generator, operand_type, operand_index in program
    ]


def validate_program(program):
    from parsel.edit_dsl.edits import EditGen

    if not isinstance(program, list):
        raise ValueError("Archived program is not a list")
    for item in program:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"Invalid archived edit: {item!r}")
        generator, operand_type, operand_index = item
        if not isinstance(generator, EditGen):
            raise ValueError(f"Invalid archived edit generator: {generator!r}")
        if operand_type not in {"part", "relation"}:
            raise ValueError(f"Invalid operand type: {operand_type!r}")
        if not isinstance(operand_index, int):
            raise ValueError(f"Invalid operand index: {operand_index!r}")


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    archives = {
        "partnet": args.partnet_archive.expanduser().resolve(),
        "compat3d": args.compat3d_archive.expanduser().resolve(),
    }
    for archive in archives.values():
        if not archive.is_file():
            raise SystemExit(f"Program archive does not exist: {archive}")

    rows = load_rows(output)
    replaced = 0
    for result in rows:
        if result["status"] in LOADABLE_STATUSES:
            atomic_json(output / result["result_file"], result)
            continue
        archive = archives[result["dataset"]]
        program, member = load_legacy_program(
            archive,
            result["row_index"],
        )
        validate_program(program)
        program_path = output / result["program_file"]
        atomic_pickle(program_path, program)

        result.update(
            {
                "status": "archived_fallback",
                "source": "paper_program_archive",
                "replay_error_type": result.get("error_type", ""),
                "replay_error": result.get("error", ""),
                "error_type": "",
                "error": "",
                "edit_count": len(program),
                "program": describe_program(program),
                "program_signature": program_signature(program),
                "program_sha256": sha256(program_path),
                "archive_file": archive.name,
                "archive_member": member,
                "archive_sha256": sha256(archive),
            }
        )
        atomic_json(output / result["result_file"], result)
        replaced += 1

    with (output / "generation.json").open(encoding="utf-8") as handle:
        generation = json.load(handle)
    summary = write_manifest(output, rows, generation)
    print(f"Imported {replaced} archived fallbacks")
    print(
        f"Loadable: {summary['success']}/{summary['total']} "
        f"({summary['by_source']['regenerated']} regenerated, "
        f"{summary['by_source']['archived_fallback']} archived fallbacks)"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
