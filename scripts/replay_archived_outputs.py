#!/usr/bin/env python3
"""Replay archived prompting decisions through the cleaned ParSEL workflow."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = REPOSITORY / "metadata"
RESULT_FIELDS = (
    "dataset",
    "row_index",
    "dataset_index",
    "shape_id",
    "category",
    "edit_request",
    "status",
    "source",
    "error_type",
    "error",
    "duration_seconds",
    "edit_count",
    "used_breaking_edit",
    "program_file",
    "result_file",
    "program_sha256",
    "decisions_sha256",
    "archive_file",
    "archive_member",
    "archive_sha256",
)

LOADABLE_STATUSES = {"success", "archived_fallback"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fresh, isolated output set using saved paper-era "
            "prompting decisions and the cleaned ParSEL implementation."
        )
    )
    parser.add_argument("--partnet-data", type=Path, required=True)
    parser.add_argument("--compat3d-data", type=Path, required=True)
    parser.add_argument("--partnet-decisions", type=Path, required=True)
    parser.add_argument("--compat3d-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="maximum wall time for one example (default: 1800)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed for deterministic numeric relation checks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep completed results and retry missing or failed rows",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="run at most this many rows per dataset (for smoke tests)",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_rows(metadata_dir, limit):
    datasets = {
        "partnet": metadata_dir / "partnet_edits.csv",
        "compat3d": metadata_dir / "compat3d_edits.csv",
    }
    rows = []
    for dataset, path in datasets.items():
        with path.open(newline="", encoding="utf-8") as handle:
            dataset_rows = list(csv.DictReader(handle))
        if limit is not None:
            dataset_rows = dataset_rows[:limit]
        for position, row in enumerate(dataset_rows):
            row_index = int(row.get("row_index", position))
            rows.append(
                {
                    "dataset": dataset,
                    "row_index": row_index,
                    "dataset_index": row.get("dataset_index", row_index),
                    "shape_id": str(row["shape_id"]),
                    "category": row["category"],
                    "edit_request": row["edit_request"],
                }
            )
    return rows


def result_directory(output_root, row):
    return (
        output_root
        / row["dataset"]
        / f"{row['row_index']:03d}_{row['shape_id']}"
    )


def previous_success(output_root, row):
    path = result_directory(output_root, row) / "result.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") not in LOADABLE_STATUSES:
        return None
    program = output_root / result["program_file"]
    if not program.is_file() or sha256(program) != result["program_sha256"]:
        return None
    return result


def display_value(value):
    if isinstance(value, str):
        return value
    return str(value)


def describe_program(edits):
    program = []
    for edit in edits:
        operand = edit.operand
        if hasattr(operand, "part_index"):
            operand_type = "part"
            operand_index = operand.part_index
        else:
            operand_type = "relation"
            operand_index = operand.relation_index
        program.append(
            {
                "edit": edit.__class__.__name__,
                "operand_type": operand_type,
                "operand_index": operand_index,
                "operand": getattr(operand, "full_label", str(operand)),
                "amount": display_value(edit.amount),
                "parameters": {
                    key: display_value(value)
                    for key, value in edit.params.items()
                },
            }
        )
    return program


def _raise_timeout(signum, frame):
    del signum, frame
    raise TimeoutError("Example exceeded its configured wall-time limit")


def _run_one(task):
    row = task["row"]
    output_root = Path(task["output_root"])
    data_root = Path(task["data_root"])
    decisions_root = Path(task["decisions_root"])
    output_dir = result_directory(output_root, row)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    program_path = output_dir / "program.pkl"
    log_path = output_dir / "run.log"
    decisions_path = decisions_root / f"{row['row_index']}.pkl"
    cache_path = data_root / row["shape_id"] / f"{row['shape_id']}.pkl"
    started = time.monotonic()
    np.random.seed(task["seed"])

    previous_alarm_handler = None
    if hasattr(signal, "SIGALRM"):
        previous_alarm_handler = signal.signal(
            signal.SIGALRM,
            _raise_timeout,
        )
        signal.alarm(task["timeout_seconds"])

    result = {
        **row,
        "status": "error",
        "source": "regenerated",
        "error_type": "",
        "error": "",
        "duration_seconds": 0.0,
        "edit_count": 0,
        "used_breaking_edit": "",
        "program_file": program_path.relative_to(output_root).as_posix(),
        "result_file": result_path.relative_to(output_root).as_posix(),
        "program_sha256": "",
        "decisions_sha256": "",
        "archive_file": "",
        "archive_member": "",
        "archive_sha256": "",
    }

    try:
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing shape cache: {cache_path}")
        if not decisions_path.is_file():
            raise FileNotFoundError(
                f"Missing archived decisions: {decisions_path}"
            )

        with log_path.open("w", encoding="utf-8") as log_handle:
            with redirect_stdout(log_handle), redirect_stderr(log_handle):
                decisions = load_archived_decisions(decisions_path)
                edits, used_breaking_edit, final_state = replay(
                    row,
                    cache_path,
                    data_root,
                    decisions,
                    task["max_iterations"],
                )

        saved_program = [edit.save_format() for edit in edits]
        atomic_pickle(program_path, saved_program)
        result.update(
            {
                "status": "success",
                "edit_count": len(edits),
                "used_breaking_edit": used_breaking_edit,
                "program": describe_program(edits),
                "program_signature": [
                    [
                        item["edit"],
                        item["operand_type"],
                        item["operand_index"],
                    ]
                    for item in describe_program(edits)
                    if item["edit"] != "KeepFixed"
                ],
                "final_state": final_state,
                "program_sha256": sha256(program_path),
                "decisions_sha256": sha256(decisions_path),
            }
        )
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        with log_path.open("a", encoding="utf-8") as log_handle:
            traceback.print_exc(file=log_handle)
    finally:
        if previous_alarm_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)

    result["duration_seconds"] = round(time.monotonic() - started, 6)
    atomic_json(result_path, result)
    return result


def load_archived_decisions(path):
    from parsel.edit_dsl import edits as edit_module

    class LegacyDecisionUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "edit_sys.shape_system.edits":
                return getattr(edit_module, name)
            return super().find_class(module, name)

    with path.open("rb") as handle:
        decisions = LegacyDecisionUnpickler(handle).load()
    required = {
        "fixed_parts",
        "primary_edits",
        "relation_map",
        "init_edit_type_hints",
    }
    if not isinstance(decisions, dict) or set(decisions) != required:
        raise ValueError(f"Unexpected decision record: {path}")
    return decisions


class ArchivedPrompter:
    """Adapter exposing archived decisions to the cleaned workflow."""

    def __init__(self, decisions):
        self.decisions = decisions

    def api_call_initialize_parallel(self, shape, edit_request):
        from parsel.edit_dsl.edits import KeepFixed

        edits = [
            KeepFixed(shape.get(name))
            for name in self.decisions["fixed_parts"]
        ]
        for generator, operand_type, index in self.decisions[
            "primary_edits"
        ]:
            if operand_type == "part":
                matches = [
                    part
                    for part in shape.partset
                    if part.part_index == index
                ]
            else:
                matches = [
                    relation
                    for relation in shape.all_relations()
                    if relation.relation_index == index
                ]
            if not matches:
                raise ValueError(
                    f"Archived edit refers to missing {operand_type} {index}"
                )
            edits.append(generator.employ(matches[0]))

        part_labels = {
            part.full_label for part in ({shape} | shape.partset)
        }
        relation_labels = {
            relation.full_label for relation in shape.all_relations()
        }
        known_labels = part_labels | relation_labels
        hints = {}
        for name, hint in self.decisions["init_edit_type_hints"].items():
            target_name = name.split("(", 1)[0].strip()
            if target_name not in known_labels:
                raise ValueError(
                    f"Archived type hint refers to missing target {name!r}"
                )
            hints[target_name] = hint
        return edits, hints, []

    def api_call_check_relation(
        self,
        shape,
        edit_request,
        relation,
        valid_states,
    ):
        from parsel.shape_system.constants import (
            RESOLVE_RELATION,
            UPDATE_RELATION,
        )
        from parsel.shape_system.relations import (
            PrimitiveRelation,
            ReflectionSymmetry,
        )

        if not isinstance(relation, PrimitiveRelation):
            return min(valid_states), []
        selected = self.decisions["relation_map"].get(
            relation.relation_index,
            RESOLVE_RELATION,
        )
        if (
            not isinstance(relation, ReflectionSymmetry)
            and UPDATE_RELATION in valid_states
            and selected == RESOLVE_RELATION
        ):
            selected = UPDATE_RELATION
        if selected not in valid_states:
            selected = valid_states[0]
        return selected, []

    def api_call_finish_algo(self, shape, edit_request, all_edits):
        return True, None, []


def replay(row, cache_path, data_root, decisions, max_iterations):
    from parsel.data.loading import load_shape
    from parsel.edit_propagation.proposer import EditProposer
    from parsel.workflow import run_edit_workflow

    _, symbolic_data = load_shape(
        cache_path,
        data_root,
        dataset=row["dataset"],
        regenerate=False,
        add_ground=True,
    )
    shape = symbolic_data[0]
    shape.label = row["category"]
    edits, _, used_breaking_edit, final_state = run_edit_workflow(
        shape,
        row["edit_request"],
        ArchivedPrompter(decisions),
        EditProposer(heuristic="intrinsic"),
        max_iterations=max_iterations,
    )
    return edits, used_breaking_edit, final_state


def write_manifest(output_root, results, generation):
    ordered = sorted(
        results,
        key=lambda result: (result["dataset"], result["row_index"]),
    )
    manifest_path = output_root / "manifest.csv"
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in ordered:
            writer.writerow({key: result.get(key, "") for key in RESULT_FIELDS})
    os.replace(temporary, manifest_path)

    successes = [
        result
        for result in ordered
        if result["status"] in LOADABLE_STATUSES
    ]
    failures = [
        result
        for result in ordered
        if result["status"] not in LOADABLE_STATUSES
    ]
    summary = {
        "total": len(ordered),
        "success": len(successes),
        "failed": len(failures),
        "by_source": {
            "regenerated": sum(
                result["status"] == "success" for result in ordered
            ),
            "archived_fallback": sum(
                result["status"] == "archived_fallback"
                for result in ordered
            ),
        },
        "by_dataset": {
            dataset: {
                "total": sum(r["dataset"] == dataset for r in ordered),
                "success": sum(r["dataset"] == dataset for r in successes),
                "failed": sum(r["dataset"] == dataset for r in failures),
            }
            for dataset in ("partnet", "compat3d")
        },
        "failures": [
            {
                "dataset": result["dataset"],
                "row_index": result["row_index"],
                "shape_id": result["shape_id"],
                "error_type": result["error_type"],
                "error": result["error"],
            }
            for result in failures
        ],
        "generation": generation,
    }
    atomic_json(output_root / "summary.json", summary)
    return summary


def validate_inputs(args):
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.max_iterations < 1:
        raise SystemExit("--max-iterations must be positive")
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    inputs = (
        args.partnet_data,
        args.compat3d_data,
        args.partnet_decisions,
        args.compat3d_decisions,
        args.metadata_dir,
    )
    for path in inputs:
        if not path.expanduser().resolve().is_dir():
            raise SystemExit(f"Input directory does not exist: {path}")

    output_root = args.output.expanduser().resolve()
    input_roots = [path.expanduser().resolve() for path in inputs]
    if any(output_root == root or root in output_root.parents for root in input_roots):
        raise SystemExit("Output must not be inside an input directory")
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise SystemExit(
            f"Output directory is not empty: {output_root}; use --resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def main():
    args = parse_args()
    output_root = validate_inputs(args)
    metadata_dir = args.metadata_dir.expanduser().resolve()
    rows = load_rows(metadata_dir, args.limit)
    data_roots = {
        "partnet": args.partnet_data.expanduser().resolve(),
        "compat3d": args.compat3d_data.expanduser().resolve(),
    }
    decision_roots = {
        "partnet": args.partnet_decisions.expanduser().resolve(),
        "compat3d": args.compat3d_decisions.expanduser().resolve(),
    }

    generation = {
        "implementation": "cleaned-parsel",
        "python": sys.version.split()[0],
        "max_iterations": args.max_iterations,
        "timeout_seconds": args.timeout_seconds,
        "seed": args.seed,
        "row_count": len(rows),
        "metadata_sha256": {
            "partnet": sha256(metadata_dir / "partnet_edits.csv"),
            "compat3d": sha256(metadata_dir / "compat3d_edits.csv"),
        },
    }
    atomic_json(output_root / "generation.json", generation)

    completed = []
    pending = []
    for row in rows:
        previous = previous_success(output_root, row) if args.resume else None
        if previous is not None:
            completed.append(previous)
            continue
        pending.append(
            {
                "row": row,
                "output_root": str(output_root),
                "data_root": str(data_roots[row["dataset"]]),
                "decisions_root": str(decision_roots[row["dataset"]]),
                "max_iterations": args.max_iterations,
                "timeout_seconds": args.timeout_seconds,
                "seed": args.seed
                + row["row_index"]
                + (10000 if row["dataset"] == "compat3d" else 0),
            }
        )

    print(
        f"Replaying {len(pending)} rows; keeping {len(completed)} completed rows"
    )
    if args.workers == 1:
        for position, task in enumerate(pending, start=1):
            result = _run_one(task)
            completed.append(result)
            write_manifest(output_root, completed, generation)
            print(
                f"[{position}/{len(pending)}] {result['dataset']} "
                f"{result['row_index']}: {result['status']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_one, task): task for task in pending}
            for position, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                completed.append(result)
                write_manifest(output_root, completed, generation)
                print(
                    f"[{position}/{len(pending)}] {result['dataset']} "
                    f"{result['row_index']}: {result['status']}",
                    flush=True,
                )

    summary = write_manifest(output_root, completed, generation)
    print(
        f"Output set: {output_root}\n"
        f"Success: {summary['success']}/{summary['total']}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
