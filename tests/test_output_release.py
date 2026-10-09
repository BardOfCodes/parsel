import csv
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import zipfile

import sympy as sp

from parsel.edit_dsl.edits import EditGen, PartTranslate


REPOSITORY = Path(__file__).resolve().parents[1]


def _make_outputs(source, log):
    folder = source / "partnet/000_1"
    folder.mkdir(parents=True)
    program = pickle.dumps([(EditGen(PartTranslate, {"direction": sp.Matrix([[1, 0, 0]])}), "part", 0)])
    (folder / "program.pkl").write_bytes(program)
    checksum = hashlib.sha256(program).hexdigest()
    result = {"status": "success", "program_sha256": checksum}
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "run.log").write_text(log, encoding="utf-8")
    row = {
        "dataset": "partnet",
        "row_index": "0",
        "shape_id": "1",
        "status": "success",
        "program_file": "partnet/000_1/program.pkl",
        "result_file": "partnet/000_1/result.json",
        "program_sha256": checksum,
    }
    with (source / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    for name in ("summary.json", "generation.json"):
        (source / name).write_text("{}\n", encoding="utf-8")
    return program


def _prepare(source, destination):
    return subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/prepare_output_release.py"),
            str(source),
            str(destination),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
    )


def test_output_release_excludes_inputs_and_sanitizes_logs(tmp_path):
    source = tmp_path / "source"
    log = 'File "/home/researcher/private/project.py", line 12\n'
    program = _make_outputs(source, log)
    (source / "source_mesh.obj").write_text("mesh", encoding="utf-8")
    (source / "shape_cache.pkl").write_bytes(b"cache")
    destination = tmp_path / "release"

    completed = _prepare(source, destination)

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in destination.iterdir()} == {
        "README.md", "LICENSE", "LICENSES.md", "CITATION.cff",
        "SHA256SUMS", "parsel_edit_programs.zip",
    }
    archive_path = destination / "parsel_edit_programs.zip"
    card = (destination / "README.md").read_text(encoding="utf-8")
    assert "license_link: https://" in card
    assert "license_name: partnet-and-3dcompat-source-terms" in card
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "parsel_clean_outputs/metadata/partnet_edits.csv" in names
        assert "parsel_clean_outputs/metadata/compat3d_edits.csv" in names
        assert not any(name.endswith((".obj", ".glb", "shape_cache.pkl")) for name in names)
        assert archive.read("parsel_clean_outputs/partnet/000_1/program.pkl") == program
        assert archive.read("parsel_clean_outputs/partnet/000_1/run.log") == b'File "<local-path>", line 12\n'
        release = json.loads(archive.read("parsel_clean_outputs/release.json"))
        assert release["program_bytes_unchanged"] is True
        assert release["sanitized_logs"] == ["partnet/000_1/run.log"]
    assert (source / "partnet/000_1/run.log").read_text(encoding="utf-8") == log
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert (destination / "SHA256SUMS").read_text(encoding="utf-8") == f"{checksum}  parsel_edit_programs.zip\n"


def test_output_release_refuses_existing_destination(tmp_path):
    destination = tmp_path / "release"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    completed = _prepare(tmp_path / "missing-source", destination)

    assert completed.returncode != 0
    assert "Refusing to overwrite" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_output_release_rejects_credential_like_logs(tmp_path):
    source = tmp_path / "source"
    _make_outputs(source, "token=" + "hf_" + "x" * 30)
    destination = tmp_path / "release"

    completed = _prepare(source, destination)

    assert completed.returncode != 0
    assert "Credential-like string" in completed.stderr
    assert not destination.exists()
