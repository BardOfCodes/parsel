#!/usr/bin/env python3
"""Validate and inspect an output set from replay_archived_outputs.py."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
LOADABLE_STATUSES = {"success", "archived_fallback"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--show",
        metavar="DATASET:ROW_INDEX",
        help="print one result and its visualization command",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="extracted parsel_experiment_data directory",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(output):
    manifest = output / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_program(path):
    from parsel.edit_dsl.edits import EditGen

    with path.open("rb") as handle:
        program = pickle.load(handle)
    if not isinstance(program, list):
        raise ValueError("program must be a list")
    for item in program:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"invalid saved edit: {item!r}")
        generator, operand_type, operand_index = item
        if not isinstance(generator, EditGen):
            raise ValueError(f"invalid edit generator: {generator!r}")
        if operand_type not in {"part", "relation"}:
            raise ValueError(f"invalid operand type: {operand_type!r}")
        if not isinstance(operand_index, int):
            raise ValueError(f"invalid operand index: {operand_index!r}")
    return program


def visualization_command(output, row, data_root):
    dataset = row["dataset"]
    shape_id = row["shape_id"]
    dataset_root = data_root / dataset
    return [
        sys.executable,
        str(REPOSITORY / "scripts" / "visualize_shape.py"),
        dataset,
        str(dataset_root / shape_id / f"{shape_id}.pkl"),
        str(dataset_root),
        "--program-file",
        str(output / row["program_file"]),
    ]


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    rows = load_manifest(output)
    errors = []

    for row in rows:
        if row["status"] not in LOADABLE_STATUSES:
            errors.append(
                f"{row['dataset']}:{row['row_index']} is {row['status']}"
            )
            continue
        result = output / row["result_file"]
        program = output / row["program_file"]
        if not result.is_file():
            errors.append(f"missing result: {result}")
            continue
        try:
            with result.open(encoding="utf-8") as handle:
                result_record = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid result {result}: {error}")
            continue
        if result_record.get("status") != row["status"]:
            errors.append(f"status mismatch: {result}")
            continue
        if result_record.get("program_sha256") != row["program_sha256"]:
            errors.append(f"result checksum mismatch: {result}")
            continue
        if not program.is_file():
            errors.append(f"missing program: {program}")
            continue
        if sha256(program) != row["program_sha256"]:
            errors.append(f"program checksum mismatch: {program}")
            continue
        try:
            validate_program(program)
        except (OSError, ValueError, pickle.UnpicklingError) as error:
            errors.append(f"invalid program {program}: {error}")

    by_dataset = {
        dataset: sum(row["dataset"] == dataset for row in rows)
        for dataset in ("partnet", "compat3d")
    }
    print(
        f"Rows: {len(rows)} "
        f"(PartNet {by_dataset['partnet']}, Compat3D {by_dataset['compat3d']})"
    )
    print(f"Valid: {len(rows) - len(errors)}; Errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")

    if args.show:
        try:
            dataset, row_index_text = args.show.split(":", 1)
            row_index = int(row_index_text)
        except ValueError as error:
            raise SystemExit("--show must look like partnet:0") from error
        matches = [
            row
            for row in rows
            if row["dataset"] == dataset
            and int(row["row_index"]) == row_index
        ]
        if len(matches) != 1:
            raise SystemExit(f"Output row not found: {args.show}")
        row = matches[0]
        with (output / row["result_file"]).open(encoding="utf-8") as handle:
            result = json.load(handle)
        print(json.dumps(result, indent=2, sort_keys=True))
        if (
            args.data_root is not None
            and row["status"] in LOADABLE_STATUSES
        ):
            command = visualization_command(
                output,
                row,
                args.data_root.expanduser().resolve(),
            )
            print("\nVisualize with:")
            print(shlex.join(command))

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
