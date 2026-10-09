#!/usr/bin/env python3
"""Render one cached PartNet or Compat3D shape to an isolated PNG."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["partnet", "compat3d"])
    parser.add_argument("cache_file", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG destination (defaults beneath .test-output/rendering)",
    )
    return parser.parse_args()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    source_cache = args.cache_file.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    if not source_cache.is_file():
        raise SystemExit(f"Cached shape does not exist: {source_cache}")
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")

    output = args.output
    if output is None:
        output = (
            repository
            / ".test-output"
            / "rendering"
            / args.dataset
            / f"{source_cache.stem}.png"
        )
    output = output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing render: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_digest = file_digest(source_cache)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{source_cache.stem}.",
            dir=output.parent,
        )
    )
    copied_cache = run_dir / source_cache.name
    shutil.copy2(source_cache, copied_cache)

    from parsel.data.loading import load_shape
    from parsel.visualization.renderer import ShapeRenderer

    processed_data, symbolic_data = load_shape(
        copied_cache,
        data_dir,
        dataset=args.dataset,
        regenerate=False,
        add_ground=False,
    )
    renderer = ShapeRenderer(
        processed_data,
        symbolic_data,
        str(output),
    )
    renderer.render()

    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"Renderer did not produce an image: {output}")
    if file_digest(source_cache) != source_digest:
        raise SystemExit(f"Dataset cache was modified: {source_cache}")

    print(f"render: {output}")
    print(f"copied cache: {copied_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
