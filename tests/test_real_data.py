from collections import Counter
import csv
import hashlib
import os
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
import tempfile

import pytest
import sympy as sp

from parsel.edit_dsl.edits import PartTranslate
from parsel.shape_system.geometry import RightUnitVector
from parsel.shape_system.relations import ReflectionSymmetry


PARTNET_SHAPE_ID = "37807"
COMPAT3D_SHAPE_ID = "685"


def _directory_fingerprint(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _load_cached_shape(
    *,
    dataset_name,
    shape_id,
    data_dir,
    output_root,
):
    assert data_dir.is_dir()
    assert output_root != data_dir
    assert data_dir not in output_root.parents
    output_root.mkdir(parents=True, exist_ok=True)

    source_shape_dir = data_dir / shape_id
    source_cache = source_shape_dir / f"{shape_id}.pkl"
    assert source_cache.is_file()
    source_fingerprint = _directory_fingerprint(source_shape_dir)

    dataset_output = output_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix=f"{shape_id}.", dir=dataset_output)
    )
    test_cache = run_dir / f"{shape_id}.pkl"
    shutil.copy2(source_cache, test_cache)

    # Import only after the environment check, so the ordinary unit suite
    # does not require Open3D.
    from parsel.data.loading import load_shape

    visual_data, symbolic_data = load_shape(
        str(test_cache),
        str(data_dir),
        dataset=dataset_name,
        regenerate=False,
        add_ground=False,
    )

    assert _directory_fingerprint(source_shape_dir) == source_fingerprint
    return SimpleNamespace(
        data_dir=data_dir,
        cache_file=test_cache,
        output_dir=run_dir,
        shape=symbolic_data[0],
        symmetries=symbolic_data[1],
        contacts=symbolic_data[2],
        visual_parts=visual_data[0],
        visual_relations=visual_data[1],
        visual_contacts=visual_data[2],
        add_color=visual_data[3],
        source_shape_dir=source_shape_dir,
        source_fingerprint=source_fingerprint,
    )


def _real_data_paths(data_env):
    data_dir_value = os.environ.get(data_env)
    if not data_dir_value:
        pytest.skip(
            f"Set {data_env} to run this real-data acceptance test."
        )

    data_dir = Path(data_dir_value).expanduser().resolve()

    default_output = (
        Path(__file__).resolve().parents[1] / ".test-output" / "real_data"
    )
    output_root = Path(
        os.environ.get("PARSEL_TEST_OUTPUT_DIR", default_output)
    ).expanduser().resolve()
    return data_dir, output_root


@pytest.fixture(scope="module")
def partnet_chair():
    data_dir, output_root = _real_data_paths(
        "PARSEL_PARTNET_DATA_DIR"
    )

    data = _load_cached_shape(
        dataset_name="partnet",
        shape_id=PARTNET_SHAPE_ID,
        data_dir=data_dir,
        output_root=output_root,
    )
    return data


@pytest.fixture(scope="module")
def compat3d_bench():
    data_dir, output_root = _real_data_paths(
        "PARSEL_COMPAT3D_DATA_DIR"
    )

    metadata_path = (
        Path(__file__).resolve().parents[1]
        / "metadata"
        / "compat3d_edits.csv"
    )
    with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
        item = next(
            item
            for item in csv.DictReader(metadata_file)
            if item["shape_id"] == COMPAT3D_SHAPE_ID
        )
    assert item["category"] == "bench"
    assert item["edit_request"] == (
        "I want to bring the legs closer together while keeping the "
        "backrest as is."
    )

    data = _load_cached_shape(
        dataset_name="compat3d",
        shape_id=COMPAT3D_SHAPE_ID,
        data_dir=data_dir,
        output_root=output_root,
    )
    data.item = item
    return data


@pytest.mark.real_data
def test_partnet_shape_builds_expected_part_graph(partnet_chair):
    data = partnet_chair
    shape = data.shape

    assert shape.label == "chair"
    assert len(shape.partset) == 6
    assert len(shape.sub_parts) == 6
    assert {part.label for part in shape.partset} == {
        "chair_back",
        "chair_seat",
        "leg_back_left",
        "leg_back_right",
        "leg_front_left",
        "leg_front_right",
    }
    assert Counter(type(item).__name__ for item in data.symmetries) == {
        "ReflectionSymmetry": 2,
    }
    assert Counter(type(item).__name__ for item in data.contacts) == {
        "LineContact": 1,
        "PointContact": 6,
    }
    assert len(shape.all_relations(only_active=True)) == 9
    assert all(part["mesh"].meshes for part in data.visual_parts.values())
    assert all(
        not relation.broken()
        for relation in shape.all_relations(only_active=True)
    )


@pytest.mark.real_data
def test_compat3d_shape_builds_expected_part_graph(compat3d_bench):
    data = compat3d_bench
    shape = data.shape

    assert shape.label == "bench"
    assert len(shape.partset) == 9
    assert len(shape.sub_parts) == 9
    assert {part.label for part in shape.partset} == {
        "backrest",
        "backrest_connector_left",
        "backrest_connector_right",
        "leg_back_left",
        "leg_back_right",
        "leg_front_left",
        "leg_front_right",
        "seat",
        "seat_frame",
    }
    assert Counter(type(item).__name__ for item in data.symmetries) == {
        "ReflectionSymmetry": 5,
    }
    assert Counter(type(item).__name__ for item in data.contacts) == {
        "FaceContact": 1,
        "LineContact": 2,
        "PointContact": 12,
    }
    assert len(shape.all_relations(only_active=True)) == 20
    assert all(part["mesh"].meshes for part in data.visual_parts.values())
    assert all(
        not relation.broken()
        for relation in shape.all_relations(only_active=True)
    )


@pytest.mark.real_data
def test_workflow_shape_adds_ground_without_optional_io_dependencies(
    partnet_chair,
):
    from parsel.data.loading import load_shape

    _, symbolic_data = load_shape(
        partnet_chair.cache_file,
        partnet_chair.data_dir,
        dataset="partnet",
        regenerate=False,
        add_ground=True,
    )

    shape = symbolic_data[0]
    ground = shape.get("ground")
    assert ground.original_label == "ground"
    assert ground in shape.partset


def _assert_front_leg_propagation(data, fixed_part_label, program_name):
    shape = data.shape
    shape.clean_up_motion()

    left_leg = shape.get("leg_front_left")
    right_leg = shape.get("leg_front_right")
    fixed_part = shape.get(fixed_part_label)
    relation = next(
        relation
        for relation in shape.all_relations()
        if (
            isinstance(relation, ReflectionSymmetry)
            and set(relation.parts) == {left_leg, right_leg}
        )
    )

    seed_edit = PartTranslate(
        left_leg,
        direction=RightUnitVector(),
        amount=sp.Rational(1, 10),
    )
    seed_edit.propagate()
    assert relation.broken()

    propagated_edits = relation.resolve()
    assert len(propagated_edits) == 1
    assert isinstance(propagated_edits[0], PartTranslate)
    assert propagated_edits[0].operand is right_leg
    propagated_edits[0].propagate()

    assert not relation.broken()
    assert fixed_part.primitive.edit_sequence == []

    program_dir = data.output_dir / "programs"
    program_dir.mkdir(parents=True, exist_ok=True)
    program_path = program_dir / program_name
    with program_path.open("wb") as program_file:
        pickle.dump(
            [
                seed_edit.save_format(),
                propagated_edits[0].save_format(),
            ],
            program_file,
        )

    assert program_path.is_file()
    assert data.data_dir not in program_path.resolve().parents
    assert (
        _directory_fingerprint(data.source_shape_dir)
        == data.source_fingerprint
    )
    shape.clean_up_motion()


@pytest.mark.real_data
def test_partnet_leg_edit_propagates_without_touching_data(partnet_chair):
    _assert_front_leg_propagation(
        partnet_chair,
        fixed_part_label="chair_back",
        program_name="front_leg_edit.pkl",
    )


@pytest.mark.real_data
def test_compat3d_leg_edit_propagates_without_touching_data(
    compat3d_bench,
):
    _assert_front_leg_propagation(
        compat3d_bench,
        fixed_part_label="backrest",
        program_name="front_leg_edit.pkl",
    )


def _assert_headless_visualization_state(data):
    from parsel.visualization.helpers import (
        create_geom_map,
        update_switch_board,
    )

    geometries, labels = create_geom_map(
        data.visual_parts,
        data.visual_relations,
        data.visual_contacts,
        data.add_color,
    )
    switches = update_switch_board(
        geometries,
        labels,
        data.shape,
    )

    part_indices = set(data.visual_parts)
    for part_index in part_indices:
        assert f"{part_index}_mesh" in geometries
        assert f"{part_index}_obb" in geometries
        assert f"{part_index}_3d_label" in labels
        assert switches[f"{part_index}_mesh"] is True
        assert switches[f"{part_index}_3d_label"] is True
        assert switches[f"{part_index}_obb"] is False

    assert sum(switches.values()) == 2 * len(part_indices)
    assert (
        _directory_fingerprint(data.source_shape_dir)
        == data.source_fingerprint
    )


@pytest.mark.real_data
@pytest.mark.visualization
def test_partnet_builds_headless_visualization_state(partnet_chair):
    _assert_headless_visualization_state(partnet_chair)


@pytest.mark.real_data
@pytest.mark.visualization
def test_compat3d_builds_headless_visualization_state(compat3d_bench):
    _assert_headless_visualization_state(compat3d_bench)
