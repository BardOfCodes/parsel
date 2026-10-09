#!/usr/bin/env python3
"""Visually review replay outputs in Open3D and record verdicts."""

import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess

from check_replay_outputs import (
    LOADABLE_STATUSES,
    REPOSITORY,
    load_manifest,
    visualization_command,
)


REVIEW_FIELDS = (
    "dataset",
    "row_index",
    "shape_id",
    "category",
    "verdict",
    "notes",
)
VERDICTS = {
    "g": "good",
    "m": "mid",
    "b": "bad",
    "s": "skipped",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--start",
        metavar="DATASET:ROW_INDEX",
        help="start at a specific row, such as compat3d:12",
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        default=REPOSITORY / ".test-output" / "replay_review.csv",
    )
    return parser.parse_args()


def row_key(row):
    return row["dataset"], int(row["row_index"])


def load_reviews(path):
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row_key(row): row for row in csv.DictReader(handle)}


def save_reviews(path, reviews):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(reviews[key] for key in sorted(reviews))
    temporary.replace(path)


def parse_start(value):
    if value is None:
        return None
    try:
        dataset, index = value.split(":", 1)
        return dataset, int(index)
    except ValueError as error:
        raise SystemExit("--start must look like partnet:0") from error


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    review_file = args.review_file.expanduser().resolve()
    review_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = review_file.with_suffix(review_file.suffix + ".lock").open("a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit(
            f"Another review session is already using {review_file}"
        ) from error
    rows = [
        row
        for row in load_manifest(output)
        if row["status"] in LOADABLE_STATUSES
    ]
    reviews = load_reviews(review_file)
    start = parse_start(args.start)
    if start is not None:
        matches = [index for index, row in enumerate(rows) if row_key(row) == start]
        if len(matches) != 1:
            raise SystemExit(f"Output row not found: {args.start}")
        rows = rows[matches[0] :]

    for position, row in enumerate(rows, start=1):
        key = row_key(row)
        reviews = load_reviews(review_file)
        if start is None and key in reviews:
            continue
        result_path = output / row["result_file"]
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)

        print(
            f"\n[{position}/{len(rows)}] {row['dataset']}:{row['row_index']} "
            f"{row['category']} {row['shape_id']}"
        )
        print(result["edit_request"])
        print(f"source: {row['source']}")
        completed = subprocess.run(
            visualization_command(output, row, data_root),
            cwd=REPOSITORY,
            check=False,
        )
        if completed.returncode:
            print(f"Visualizer exited with status {completed.returncode}")

        choice = input(
            "Verdict [g]ood/[m]id/[b]ad/[s]kip/[q]uit: "
        ).strip().lower()
        if choice in {"q", "quit"}:
            break
        verdict = VERDICTS.get(choice)
        if verdict is None:
            print("Unrecognized verdict; leaving this row unreviewed.")
            continue
        notes = input("Notes (optional): ").strip()
        reviews = load_reviews(review_file)
        reviews[key] = {
            "dataset": row["dataset"],
            "row_index": row["row_index"],
            "shape_id": row["shape_id"],
            "category": row["category"],
            "verdict": verdict,
            "notes": notes,
        }
        save_reviews(review_file, reviews)
        print(f"Saved review: {review_file}")

    reviews = load_reviews(review_file)
    good = sum(row["verdict"] == "good" for row in reviews.values())
    mid = sum(row["verdict"] == "mid" for row in reviews.values())
    bad = sum(row["verdict"] == "bad" for row in reviews.values())
    skipped = sum(row["verdict"] == "skipped" for row in reviews.values())
    print(
        f"Review totals: good={good}, mid={mid}, "
        f"bad={bad}, skipped={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
