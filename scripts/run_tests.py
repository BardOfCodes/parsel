#!/usr/bin/env python3
"""Run the ParSEL test suite with optional real-dataset configuration."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


DATASET_OPTIONS = {
    "partnet_data": "PARSEL_PARTNET_DATA_DIR",
    "compat3d_data": "PARSEL_COMPAT3D_DATA_DIR",
}


def parse_args():
    parser = argparse.ArgumentParser()
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--portable",
        action="store_true",
        help="run tests that do not require local datasets",
    )
    scope.add_argument(
        "--real-data",
        action="store_true",
        help="run the PartNet and Compat3D acceptance tests",
    )
    scope.add_argument(
        "--visualization",
        action="store_true",
        help="run tests for the optional Open3D visualizer",
    )
    for argument, environment_name in DATASET_OPTIONS.items():
        parser.add_argument(
            f"--{argument.replace('_', '-')}",
            help=f"override {environment_name}",
        )
    parser.add_argument(
        "--output-dir",
        help="isolated output directory for real-data tests",
    )
    args, pytest_args = parser.parse_known_args()
    args.pytest_args = pytest_args
    return args


def main():
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()

    for argument, environment_name in DATASET_OPTIONS.items():
        value = getattr(args, argument)
        if value:
            environment[environment_name] = str(Path(value).expanduser())

    if args.output_dir:
        environment["PARSEL_TEST_OUTPUT_DIR"] = str(
            Path(args.output_dir).expanduser()
        )

    command = [sys.executable, "-m", "pytest"]
    if args.portable:
        command.extend(["-m", "not real_data and not visualization"])
    elif args.real_data:
        missing = [
            name
            for name in DATASET_OPTIONS.values()
            if not environment.get(name)
        ]
        if missing:
            raise SystemExit(
                "Real-data tests require: " + ", ".join(missing)
            )
        command.extend(["-m", "real_data"])
    elif args.visualization:
        command.extend(["-m", "visualization"])
    command.extend(args.pytest_args)

    result = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
