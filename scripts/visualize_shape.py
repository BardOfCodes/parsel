#!/usr/bin/env python3
"""Open one cached PartNet or Compat3D shape in the ParSEL GUI."""

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
        "--program-file",
        type=Path,
        help="optional saved edit program to expose as sliders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="generated-file directory (defaults beneath .test-output)",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="construct and immediately close the window",
    )
    return parser.parse_args()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_file(path, label):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} does not exist: {resolved}")
    return resolved


def main():
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    source_cache = require_file(args.cache_file, "Cached shape")
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("Window width and height must be positive")

    program_file = None
    program_digest = None
    if args.program_file is not None:
        program_file = require_file(args.program_file, "Edit program")
        program_digest = file_digest(program_file)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            repository
            / ".test-output"
            / "visualizer"
            / args.dataset
            / source_cache.stem
        )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_digest = file_digest(source_cache)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{source_cache.stem}.",
            dir=output_dir,
        )
    )
    copied_cache = run_dir / source_cache.name
    shutil.copy2(source_cache, copied_cache)

    import open3d.visualization.gui as gui

    from parsel.visualization.app import ParselApp

    gui.Application.instance.initialize()
    app = ParselApp(
        copied_cache,
        data_dir,
        output_dir,
        args.dataset,
        program_file=program_file,
        width=args.width,
        height=args.height,
    )
    if args.smoke_test:
        gui.Application.instance.post_to_main_thread(
            app.window,
            app.window.close,
        )
    gui.Application.instance.run()

    if file_digest(source_cache) != source_digest:
        raise SystemExit(f"Dataset cache was modified: {source_cache}")
    if (
        program_file is not None
        and file_digest(program_file) != program_digest
    ):
        raise SystemExit(f"Edit program was modified: {program_file}")

    print(f"copied cache: {copied_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
