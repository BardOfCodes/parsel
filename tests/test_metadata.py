import csv
from pathlib import Path
import pickle
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
METADATA = REPOSITORY / "metadata"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def test_checked_in_partnet_edit_metadata_preserves_legacy_indices():
    rows = read_csv(METADATA / "partnet_edits.csv")

    assert len(rows) == 52
    assert [int(row["row_index"]) for row in rows] == list(range(52))
    assert len({row["shape_id"] for row in rows}) == 47
    assert {row["category"] for row in rows} == {
        "Bed",
        "Chair",
        "Lamp",
        "StorageFurniture",
        "Table",
    }
    mismatches = {
        int(row["row_index"]): int(row["dataset_index"])
        for row in rows
        if row["row_index"] != row["dataset_index"]
    }
    assert mismatches == {47: 51, 51: 47}


def test_checked_in_compat3d_metadata_excludes_only_summary_row():
    rows = read_csv(METADATA / "compat3d_edits.csv")

    assert len(rows) == 50
    assert [int(row["row_index"]) for row in rows] == list(range(50))
    assert len({row["shape_id"] for row in rows}) == 50
    assert sum(row["annotation_verified"] == "1" for row in rows) == 42
    missing_keep_fixed = [
        row["shape_id"]
        for row in rows
        if row["keep_fixed"] == ""
    ]
    assert missing_keep_fixed == ["7674"]
    assert all(row["shape_id"].isdigit() for row in rows)


def test_converter_generates_and_checks_normalized_csvs(tmp_path):
    partnet_edits = tmp_path / "partnet.pkl"
    partnet_catalog = tmp_path / "catalog.pkl"
    compat3d_edits = tmp_path / "compat3d.pkl"
    output_dir = tmp_path / "output"

    partnet_edits.write_bytes(
        pickle.dumps(
            [
                {
                    "dataset_index": 7,
                    "shape_id": "123",
                    "edit_request": "Move the leg.",
                }
            ]
        )
    )
    partnet_catalog.write_bytes(
        pickle.dumps(
            {
                "Chair": [
                    {
                        "anno_id": 123,
                        "shapenet_model_id": "shape-net-id",
                    }
                ]
            }
        )
    )
    compat3d_edits.write_bytes(
        pickle.dumps(
            [
                {
                    "category": "bench",
                    "type": "0",
                    "shape_id": "456",
                    "edit_request": "Move the legs.",
                    "keep_fixed": "",
                    "edit_type": "1",
                    "Annotation Verification": "1",
                    "Remark": "NIL",
                },
                {
                    "category": "total",
                    "type": "-",
                    "shape_id": "-",
                    "edit_request": "-",
                    "keep_fixed": "0",
                    "edit_type": "1",
                    "Annotation Verification": "1",
                    "Remark": "-",
                },
            ]
        )
    )

    command = [
        sys.executable,
        str(REPOSITORY / "scripts" / "convert_metadata.py"),
        "--partnet-edits",
        str(partnet_edits),
        "--partnet-catalog",
        str(partnet_catalog),
        "--compat3d-edits",
        str(compat3d_edits),
        "--output-dir",
        str(output_dir),
    ]
    generated = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    checked = subprocess.run(
        [*command, "--check"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert generated.returncode == 0, generated.stderr
    assert checked.returncode == 0, checked.stderr
    assert read_csv(output_dir / "partnet_edits.csv") == [
        {
            "row_index": "0",
            "dataset_index": "7",
            "shape_id": "123",
            "category": "Chair",
            "shapenet_model_id": "shape-net-id",
            "edit_request": "Move the leg.",
        }
    ]
    compat3d = read_csv(output_dir / "compat3d_edits.csv")
    assert compat3d[0]["shape_id"] == "456"
    assert compat3d[0]["keep_fixed"] == ""
