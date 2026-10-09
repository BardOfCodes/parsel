#!/usr/bin/env python3
"""Convert the legacy PartNet and Compat3D selection pickles to CSV."""

import argparse
import csv
from io import StringIO
from pathlib import Path
import pickle


PARTNET_FIELDS = (
    "row_index",
    "dataset_index",
    "shape_id",
    "category",
    "shapenet_model_id",
    "edit_request",
)
COMPAT3D_FIELDS = (
    "row_index",
    "shape_id",
    "category",
    "request_type",
    "edit_request",
    "keep_fixed",
    "edit_type",
    "annotation_verified",
    "remark",
)


def parse_args():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--partnet-edits", type=Path, required=True)
    parser.add_argument("--partnet-catalog", type=Path, required=True)
    parser.add_argument("--compat3d-edits", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "metadata",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="compare generated content with the existing CSV files",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing CSV files when their content differs",
    )
    return parser.parse_args()


def load_pickle(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Metadata file does not exist: {path}")
    with path.open("rb") as input_file:
        return pickle.load(input_file)


def require_keys(row, expected, source):
    missing = set(expected) - set(row)
    if missing:
        raise ValueError(
            f"{source} row is missing fields: {sorted(missing)}"
        )


def optional_int(value, field, shape_id):
    value = str(value).strip()
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"Compat3D shape {shape_id} has invalid {field}: {value!r}"
        ) from error


def partnet_rows(edit_rows, catalog):
    if not isinstance(edit_rows, list):
        raise ValueError("PartNet edit metadata must be a list")
    if not isinstance(catalog, dict):
        raise ValueError("PartNet catalog metadata must be a dictionary")

    shape_metadata = {}
    for category, category_rows in catalog.items():
        if not isinstance(category_rows, list):
            raise ValueError(f"PartNet category {category!r} is not a list")
        for row in category_rows:
            require_keys(
                row,
                ("anno_id", "shapenet_model_id"),
                "PartNet catalog",
            )
            shape_id = str(row["anno_id"])
            metadata = (str(category), str(row["shapenet_model_id"]))
            previous = shape_metadata.setdefault(shape_id, metadata)
            if previous != metadata:
                raise ValueError(
                    f"Conflicting PartNet metadata for shape {shape_id}"
                )

    converted = []
    for row_index, row in enumerate(edit_rows):
        require_keys(
            row,
            ("dataset_index", "shape_id", "edit_request"),
            "PartNet edit",
        )
        shape_id = str(row["shape_id"])
        if shape_id not in shape_metadata:
            raise ValueError(
                f"PartNet edit shape {shape_id} is absent from the catalog"
            )
        category, shapenet_model_id = shape_metadata[shape_id]
        converted.append(
            {
                "row_index": row_index,
                "dataset_index": int(row["dataset_index"]),
                "shape_id": shape_id,
                "category": category,
                "shapenet_model_id": shapenet_model_id,
                "edit_request": str(row["edit_request"]).strip(),
            }
        )
    return converted


def compat3d_rows(edit_rows):
    if not isinstance(edit_rows, list):
        raise ValueError("Compat3D edit metadata must be a list")

    required = (
        "category",
        "type",
        "shape_id",
        "edit_request",
        "keep_fixed",
        "edit_type",
        "Annotation Verification",
        "Remark",
    )
    converted = []
    summary_rows = []
    seen_shape_ids = set()
    for source_row in edit_rows:
        require_keys(source_row, required, "Compat3D edit")
        shape_id = str(source_row["shape_id"]).strip()
        if not shape_id.isdigit():
            summary_rows.append(source_row)
            continue
        if shape_id in seen_shape_ids:
            raise ValueError(f"Duplicate Compat3D shape ID: {shape_id}")
        seen_shape_ids.add(shape_id)
        converted.append(
            {
                "row_index": len(converted),
                "shape_id": shape_id,
                "category": str(source_row["category"]).strip(),
                "request_type": optional_int(
                    source_row["type"],
                    "type",
                    shape_id,
                ),
                "edit_request": str(source_row["edit_request"]).strip(),
                "keep_fixed": optional_int(
                    source_row["keep_fixed"],
                    "keep_fixed",
                    shape_id,
                ),
                "edit_type": optional_int(
                    source_row["edit_type"],
                    "edit_type",
                    shape_id,
                ),
                "annotation_verified": optional_int(
                    source_row["Annotation Verification"],
                    "Annotation Verification",
                    shape_id,
                ),
                "remark": str(source_row["Remark"]).strip(),
            }
        )

    if len(summary_rows) != 1 or summary_rows[0]["category"] != "total":
        raise ValueError(
            "Compat3D metadata must end with exactly one total row"
        )
    return converted


def csv_text(rows, fields):
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def save_csv(path, content, *, check, overwrite):
    if check:
        if not path.is_file():
            raise ValueError(f"Expected generated CSV does not exist: {path}")
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Generated CSV is out of date: {path}")
        print(f"verified: {path}")
        return

    if path.exists() and not overwrite:
        if path.read_text(encoding="utf-8") == content:
            print(f"unchanged: {path}")
            return
        raise ValueError(
            f"Refusing to replace different CSV without --overwrite: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote: {path}")


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    partnet = partnet_rows(
        load_pickle(args.partnet_edits),
        load_pickle(args.partnet_catalog),
    )
    compat3d = compat3d_rows(load_pickle(args.compat3d_edits))

    save_csv(
        output_dir / "partnet_edits.csv",
        csv_text(partnet, PARTNET_FIELDS),
        check=args.check,
        overwrite=args.overwrite,
    )
    save_csv(
        output_dir / "compat3d_edits.csv",
        csv_text(compat3d, COMPAT3D_FIELDS),
        check=args.check,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
