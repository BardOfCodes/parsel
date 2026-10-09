#!/usr/bin/env python3
"""Solve one metadata edit request and save its analytic edit program."""

import argparse
import csv
import os
from pathlib import Path
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["partnet", "compat3d"])
    parser.add_argument("dataset_index", type=int)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="metadata CSV (defaults to metadata/<dataset>_edits.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="program path (defaults beneath .test-output/programs)",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="read the provider API key from a file instead of the environment",
    )
    parser.add_argument(
        "--seed-edit",
        help=(
            "offline primary-edit expression, such as "
            "'Translate(shape.get(\"leg_front_left\"), RightUnitVector())'"
        ),
    )
    parser.add_argument(
        "--keep-fixed",
        action="append",
        default=[],
        metavar="PART",
        help="part to keep fixed in offline mode (may be repeated)",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output program",
    )
    return parser.parse_args()


def read_item(metadata_path, dataset_index):
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        items = list(csv.DictReader(metadata_file))
    try:
        return next(
            item
            for item in items
            if int(item["row_index"]) == dataset_index
        )
    except StopIteration as error:
        raise SystemExit(
            f"No metadata row with row_index={dataset_index}: "
            f"{metadata_path}"
        ) from error


def main():
    args = parse_args()
    if args.repetitions < 1 or args.repetitions % 2 == 0:
        raise SystemExit("--repetitions must be a positive odd integer")

    repository = Path(__file__).resolve().parents[1]
    metadata_path = (
        args.metadata
        if args.metadata is not None
        else repository / "metadata" / f"{args.dataset}_edits.csv"
    ).expanduser().resolve()
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata CSV does not exist: {metadata_path}")

    item = read_item(metadata_path, args.dataset_index)
    shape_id = item["shape_id"]
    data_dir = args.data_dir.expanduser().resolve()
    cache_file = data_dir / shape_id / f"{shape_id}.pkl"
    if not cache_file.is_file():
        raise SystemExit(f"Cached shape does not exist: {cache_file}")

    output = (
        args.output
        if args.output is not None
        else (
            repository
            / ".test-output"
            / "programs"
            / args.dataset
            / f"programs_{args.dataset_index}.pkl"
        )
    ).expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Output already exists (pass --force): {output}")

    from parsel.data.loading import load_shape
    from parsel.edit_propagation import EditProposer
    from parsel.edit_dsl.edits import KeepFixed
    from parsel.prompting import Prompter
    from parsel.prompting.config import MODEL, PROVIDER, SEED, TEMPERATURE
    from parsel.shape_system.constants import RESOLVE_RELATION
    from parsel.workflow import run_edit_workflow

    _, symbolic_data = load_shape(
        cache_file,
        data_dir,
        dataset=args.dataset,
        regenerate=False,
        add_ground=True,
    )
    shape = symbolic_data[0]
    shape.label = item["category"]

    solve_mode = "offline seed"
    if args.seed_edit is not None:
        parser = Prompter(PROVIDER, "", MODEL)
        response = f"```python\nprimary_edit = {args.seed_edit}\n```"
        seed_edit = parser.parse_initial_instructions(shape, response)

        class SeedPrompter:
            def api_call_initialize_parallel(
                self,
                current_shape,
                edit_request,
            ):
                del edit_request
                fixed = [
                    KeepFixed(current_shape.get(part))
                    for part in args.keep_fixed
                ]
                return fixed + [seed_edit], {}, []

            def api_call_check_relation(
                self,
                current_shape,
                edit_request,
                relation,
                valid_states,
            ):
                del current_shape, edit_request, relation
                if RESOLVE_RELATION in valid_states:
                    return RESOLVE_RELATION, []
                return valid_states[0], []

            def api_call_finish_algo(
                self,
                current_shape,
                edit_request,
                all_edits,
            ):
                del current_shape, edit_request, all_edits
                return True, None, []

        prompter = SeedPrompter()
    else:
        solve_mode = "live prompting"
        key = os.environ.get("PARSEL_LLM_API_KEY", "")
        if args.api_key_file is not None:
            key_path = args.api_key_file.expanduser().resolve()
            if not key_path.is_file():
                raise SystemExit(f"API key file does not exist: {key_path}")
            key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise SystemExit(
                "Set PARSEL_LLM_API_KEY, pass --api-key-file, or provide "
                "--seed-edit for an offline solve"
            )
        prompter = Prompter(
            provider=PROVIDER,
            key=key,
            model=MODEL,
            temperature=TEMPERATURE,
            seed=SEED,
            repetitions=args.repetitions,
        )

    edits, logs, used_fallback, final_state = run_edit_workflow(
        shape,
        item["edit_request"],
        prompter,
        EditProposer(),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "wb" if args.force else "xb"
    with output.open(file_mode) as program_file:
        pickle.dump([edit.save_format() for edit in edits], program_file)

    print(f"dataset: {args.dataset}")
    print(f"dataset index: {args.dataset_index}")
    print(f"shape id: {shape_id}")
    print(f"solve mode: {solve_mode}")
    print(f"edit request: {item['edit_request']}")
    print(f"edits: {len(edits)}")
    for edit in edits:
        print(f"  {edit}")
    print(f"prompt log entries: {len(logs)}")
    print(f"used fallback: {used_fallback}")
    print(f"final state: {final_state}")
    print(f"saved program: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
