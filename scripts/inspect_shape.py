#!/usr/bin/env python3
"""Load one cached dataset shape without writing to the dataset."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import tempfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["partnet", "compat3d"])
    parser.add_argument("shape_id")
    parser.add_argument("data_dir")
    parser.add_argument(
        "--output-dir",
        default=(
            Path(__file__).resolve().parents[1]
            / ".test-output"
            / "inspection"
        ),
        type=Path,
    )
    return parser.parse_args()


def fingerprint(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    source_dir = data_dir / args.shape_id
    source_cache = source_dir / f"{args.shape_id}.pkl"
    if not source_cache.is_file():
        raise SystemExit(f"Cached shape does not exist: {source_cache}")

    before = fingerprint(source_dir)
    dataset_output = args.output_dir.expanduser().resolve() / args.dataset
    dataset_output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{args.shape_id}.",
            dir=dataset_output,
        )
    )
    copied_cache = run_dir / source_cache.name
    shutil.copy2(source_cache, copied_cache)

    from parsel.data.loading import load_shape

    visual, symbolic = load_shape(
        str(copied_cache),
        str(data_dir),
        dataset=args.dataset,
        regenerate=False,
        add_ground=False,
    )
    shape, symmetries, contacts = symbolic
    summary = {
        "dataset": args.dataset,
        "shape_id": args.shape_id,
        "category": shape.label,
        "parts": len(shape.partset),
        "part_labels": sorted(part.label for part in shape.partset),
        "symmetries": dict(
            Counter(type(item).__name__ for item in symmetries)
        ),
        "contacts": dict(Counter(type(item).__name__ for item in contacts)),
        "meshes_loaded": all(
            part["mesh"].meshes for part in visual[0].values()
        ),
        "copied_cache": str(copied_cache),
    }
    for key, value in summary.items():
        print(f"{key}: {value}")

    if fingerprint(source_dir) != before:
        raise SystemExit(f"Dataset source was modified: {source_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
