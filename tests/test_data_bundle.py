import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]


def _write_metadata(path, shape_ids):
    with path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["row_index", "shape_id"],
        )
        writer.writeheader()
        for index, shape_id in enumerate(shape_ids):
            writer.writerow(
                {"row_index": index, "shape_id": shape_id}
            )


def _write_shape(root, shape_id, mesh_name, contents):
    shape_dir = root / shape_id
    mesh_dir = shape_dir / "objs"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / mesh_name).write_bytes(contents)
    cache = (
        [{"objs": [mesh_name], "label": "part"}],
        [],
        [],
        "shape",
    )
    with (shape_dir / f"{shape_id}.pkl").open("wb") as cache_file:
        pickle.dump(cache, cache_file)


def test_data_bundle_contains_only_resolved_shape_inputs(tmp_path):
    partnet = tmp_path / "partnet"
    partnet_acceptance = tmp_path / "partnet_acceptance"
    compat3d = tmp_path / "compat3d"
    _write_shape(partnet, "1", "part.obj", b"partnet")
    _write_shape(
        partnet_acceptance,
        "37807",
        "acceptance.obj",
        b"acceptance",
    )
    _write_shape(compat3d, "2", "part.glb", b"compat3d")
    extra = partnet / "1" / "renders" / "unused.png"
    extra.parent.mkdir()
    extra.write_bytes(b"not needed")

    partnet_metadata = tmp_path / "partnet.csv"
    compat3d_metadata = tmp_path / "compat3d.csv"
    _write_metadata(partnet_metadata, ["1", "1"])
    _write_metadata(compat3d_metadata, ["2"])
    archive_path = tmp_path / "bundle.zip"

    command = [
        sys.executable,
        str(REPOSITORY / "scripts" / "build_data_bundle.py"),
        "--partnet-data",
        str(partnet),
        "--partnet-acceptance-data",
        str(partnet_acceptance),
        "--compat3d-data",
        str(compat3d),
        "--partnet-metadata",
        str(partnet_metadata),
        "--compat3d-metadata",
        str(compat3d_metadata),
        "--output",
        str(archive_path),
    ]
    subprocess.run(command, cwd=REPOSITORY, check=True)
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "validate_data_bundle.py"),
            str(archive_path),
        ],
        cwd=REPOSITORY,
        check=True,
    )

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(
            archive.read("parsel_experiment_data/manifest.json")
        )
    assert "parsel_experiment_data/partnet/1/objs/part.obj" in names
    assert (
        "parsel_experiment_data/partnet/37807/objs/acceptance.obj"
        in names
    )
    assert "parsel_experiment_data/compat3d/2/objs/part.glb" in names
    assert all("unused.png" not in name for name in names)
    assert manifest["datasets"]["partnet"]["shape_count"] == 2
    assert manifest["datasets"]["compat3d"]["shape_count"] == 1


def test_data_bundle_validator_detects_changed_extracted_file(tmp_path):
    partnet = tmp_path / "partnet"
    partnet_acceptance = tmp_path / "partnet_acceptance"
    compat3d = tmp_path / "compat3d"
    _write_shape(partnet, "1", "part.obj", b"partnet")
    _write_shape(
        partnet_acceptance,
        "37807",
        "acceptance.obj",
        b"acceptance",
    )
    _write_shape(compat3d, "2", "part.glb", b"compat3d")
    partnet_metadata = tmp_path / "partnet.csv"
    compat3d_metadata = tmp_path / "compat3d.csv"
    _write_metadata(partnet_metadata, ["1"])
    _write_metadata(compat3d_metadata, ["2"])
    archive_path = tmp_path / "bundle.zip"

    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "build_data_bundle.py"),
            "--partnet-data",
            str(partnet),
            "--partnet-acceptance-data",
            str(partnet_acceptance),
            "--compat3d-data",
            str(compat3d),
            "--partnet-metadata",
            str(partnet_metadata),
            "--compat3d-metadata",
            str(compat3d_metadata),
            "--output",
            str(archive_path),
        ],
        cwd=REPOSITORY,
        check=True,
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    (
        extracted
        / "parsel_experiment_data"
        / "partnet"
        / "1"
        / "objs"
        / "part.obj"
    ).write_bytes(b"changed")

    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "validate_data_bundle.py"),
            str(extracted),
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stderr
