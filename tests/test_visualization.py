import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

o3d = pytest.importorskip("open3d")
pytest.importorskip("torch")

from parsel.edit_dsl.edits import PartTranslate, load_edits
from parsel.shape_system.constants import PART_INACTIVE
from parsel.shape_system.geometry import RightUnitVector
from parsel.visualization.helpers import (
    create_arrow,
    create_geom_map,
    create_plane,
    create_switch_board,
    update_switch_board,
)


pytestmark = pytest.mark.visualization


def _mesh_model(vertices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    model = o3d.visualization.rendering.TriangleMeshModel()
    model.meshes = [
        o3d.visualization.rendering.TriangleMeshModel.MeshInfo(
            mesh,
            "mesh",
            0,
        )
    ]
    model.materials = [o3d.visualization.rendering.MaterialRecord()]
    return model


@pytest.fixture
def visualizer_modules():
    app = importlib.import_module("parsel.visualization.app")
    online = importlib.import_module("parsel.visualization.online")
    yield app, online
    sys.modules.pop("parsel.visualization.online", None)
    sys.modules.pop("parsel.visualization.app", None)


def test_geometry_map_and_switch_board_follow_shape_activity(
    box_factory,
    root_factory,
):
    box = box_factory("box", part_index=1)
    shape = root_factory([box])
    vertices = np.asarray(box.primitive.static_expression(), dtype=float)
    parts = {
        1: {
            "mesh": _mesh_model(vertices),
            "obb": box.primitive.obb,
            "label": box.label,
        }
    }
    relations = {
        2: {
            "type": "T",
            "params": [np.array([1.0, 0.0, 0.0])],
            "indices": [1],
            "name": "translation",
        }
    }
    contacts = {
        3: {
            "points": np.array([[0.0, 0.0, 0.0]]),
            "name": "contact",
        }
    }

    geometries, labels = create_geom_map(
        parts,
        relations,
        contacts,
        add_color=False,
    )
    switches = create_switch_board(geometries, labels)

    assert set(geometries) == {"1_mesh", "1_obb", "2_R", "3_I"}
    assert set(labels) == {
        "1_3d_label",
        "2_R_3d_label",
        "3_I_3d_label",
    }
    assert not any(switches.values())

    switches = update_switch_board(geometries, labels, shape)
    assert switches["1_mesh"] is True
    assert switches["1_3d_label"] is True
    assert switches["1_obb"] is False
    assert switches["2_R"] is False
    assert switches["3_I"] is False

    box.state[0] = PART_INACTIVE
    switches = update_switch_board(geometries, labels, shape)
    assert switches["1_mesh"] is False
    assert switches["1_3d_label"] is False


def test_switch_board_tracks_each_part_label_independently(
    box_factory,
    root_factory,
):
    active = box_factory("active", part_index=1)
    inactive = box_factory("inactive", part_index=2)
    inactive.state[0] = PART_INACTIVE
    shape = root_factory([active, inactive])
    geometries = {
        "1_mesh": object(),
        "2_mesh": object(),
    }
    labels = {
        "1_3d_label": ("active", np.zeros(3), None),
        "2_3d_label": ("inactive", np.zeros(3), None),
    }

    switches = update_switch_board(geometries, labels, shape)

    assert switches == {
        "1_mesh": True,
        "2_mesh": False,
        "1_3d_label": True,
        "2_3d_label": False,
    }


def test_saved_edit_program_replays_into_mesh_vertices(
    box_factory,
    root_factory,
    visualizer_modules,
    capsys,
):
    app_module, _ = visualizer_modules
    box = box_factory("box", part_index=1)
    shape = root_factory([box])
    slider = sp.Symbol("slider")
    saved_program = [
        PartTranslate(
            box,
            direction=RightUnitVector(),
            amount=slider,
        ).save_format()
    ]

    edits, edited_operands = load_edits(shape, saved_program)
    assert capsys.readouterr().out == ""
    for edit in edits:
        edit.propagate()

    static_vertices = np.asarray(
        box.primitive.static_expression(),
        dtype=float,
    )
    model = _mesh_model(static_vertices)
    app = object.__new__(app_module.ParselApp)
    app.edits = {"Move": (edits, slider)}
    app.symbolic_data = (shape, [], [])
    app.name_to_geom = {"1_mesh": model}
    app.part_dict = {
        1: {
            "mesh": model,
            "obb": box.primitive.obb,
            "label": box.label,
        }
    }

    app._edit_prebake()

    assert app.symbol_to_parts == {slider: [box]}
    app.symbol_value_dict = {slider: 0.25}
    app.switch_board = {"1_mesh": False}
    app._update_part_mesh(
        box.part_index,
        box.primitive.dynamic_expression(),
    )

    displayed_vertices = np.asarray(model.meshes[0].mesh.vertices)
    np.testing.assert_allclose(
        displayed_vertices,
        static_vertices + np.array([0.25, 0.0, 0.0]),
        atol=1e-6,
    )
    assert edited_operands == [box]


def test_saved_program_adjusts_range_before_building_controls(
    box_factory,
    root_factory,
    visualizer_modules,
    tmp_path,
):
    app_module, _ = visualizer_modules
    box = box_factory("box", part_index=1)
    shape = root_factory([box])
    box.parent = None
    program_file = tmp_path / "program.pkl"
    with program_file.open("wb") as handle:
        pickle.dump(
            [
                PartTranslate(
                    box,
                    direction=RightUnitVector(),
                ).save_format()
            ],
            handle,
        )

    events = []
    app = object.__new__(app_module.ParselApp)
    app.program_file = program_file
    app.symbolic_data = (shape, [], [])
    app.name_to_geom = {}
    app.label_dict = {}
    app.switch_board = {}
    app.update_from_switch_board = lambda: None
    app.set_range = lambda: events.append("range")
    app.update_procedural_panel = lambda: events.append("controls")
    app._edit_prebake = lambda: events.append("prebake")

    app.get_edit()

    assert events == ["range", "controls", "prebake"]


def test_slider_range_shrinks_until_both_directions_are_safe(
    visualizer_modules,
):
    app_module, _ = visualizer_modules
    app = object.__new__(app_module.ParselApp)
    app.check_flipping = lambda value: abs(value) > 0.5
    app.check_distortion = lambda value: False
    app.check_minimal_motion = lambda value: False

    lower, upper = app.set_range()

    assert lower == pytest.approx(-0.35859375)
    assert upper == pytest.approx(0.35859375)


def test_slider_callback_updates_value_before_executing(
    visualizer_modules,
):
    app_module, _ = visualizer_modules
    symbol = sp.Symbol("slider")
    app = object.__new__(app_module.ParselApp)
    app.symbol_value_dict = {symbol: 0.0}
    observed = []
    app._edit_execute = lambda selected: observed.append(
        (selected, app.symbol_value_dict[selected])
    )

    callback = app.get_edit_callback(symbol)
    callback(0.4)

    assert observed == [(symbol, 0.4)]


def test_remove_edit_resets_all_values_before_rebuilding_controls(
    visualizer_modules,
):
    _, online_module = visualizer_modules
    first = sp.Symbol("first")
    second = sp.Symbol("second")
    app = object.__new__(online_module.OnlineParselApp)
    app.edits = {
        "First": ([], first),
        "Second": ([], second),
    }
    app.symbol_value_dict = {first: 0.3, second: -0.2}
    executions = []
    refreshes = []
    app._edit_execute = lambda symbol: executions.append(
        (symbol, app.symbol_value_dict[symbol])
    )
    app.update_from_switch_board = lambda: refreshes.append("switches")
    app.update_procedural_panel = lambda: refreshes.append("controls")
    app._edit_prebake = lambda: refreshes.append("prebake")

    app.remove_edit("First", first)()

    assert executions == [(first, 0), (second, 0)]
    assert app.edits == {"Second": ([], second)}
    assert app.symbol_value_dict == {second: 0}
    assert refreshes == ["switches", "controls", "prebake"]


def test_arrow_geometry_points_toward_a_negative_axis():
    arrow = create_arrow(
        np.zeros(3),
        np.array([0.0, 0.0, -2.0]),
    )

    center = arrow.get_axis_aligned_bounding_box().get_center()
    assert center[2] == pytest.approx(-1.0, abs=0.05)


def test_zero_length_arrow_is_rejected():
    with pytest.raises(ValueError, match="distinct finite points"):
        create_arrow(np.zeros(3), np.zeros(3))


def test_plane_accepts_a_negative_axis_normal():
    plane = create_plane(
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(
        plane.get_axis_aligned_bounding_box().get_center(),
        np.array([1.0, 2.0, 3.0]),
        atol=0.01,
    )


def test_unknown_relation_geometry_is_rejected(box_factory):
    box = box_factory("box", part_index=1)
    parts = {
        1: {
            "mesh": _mesh_model(
                np.asarray(box.primitive.static_expression(), dtype=float)
            ),
            "obb": box.primitive.obb,
            "label": box.label,
        }
    }

    with pytest.raises(ValueError, match="Unsupported"):
        create_geom_map(
            parts,
            {
                2: {
                    "type": "experimental",
                    "params": [],
                    "indices": [1],
                    "name": "unsupported",
                }
            },
            {},
            add_color=False,
        )


def test_visualizer_classes_use_parsel_names(visualizer_modules):
    app_module, online_module = visualizer_modules

    assert hasattr(app_module, "ParselApp")
    assert not hasattr(app_module, "EditSysApp")
    assert hasattr(online_module, "OnlineParselApp")
    assert not hasattr(online_module, "OnlineEditSysApp")


def test_headless_renderer_import_does_not_require_export_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import parsel.visualization.renderer",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_visualization_import_does_not_reset_numpy_random_state():
    code = """
import numpy as np
np.random.seed(7)
expected = np.random.random()
np.random.seed(7)
import parsel.visualization.constants
observed = np.random.random()
raise SystemExit(0 if observed == expected else 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_render_script_exposes_a_dataset_explicit_cli():
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_shape.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "{partnet,compat3d}" in result.stdout
    assert "--output" in result.stdout


def test_gui_script_exposes_cache_program_and_smoke_inputs():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "visualize_shape.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "{partnet,compat3d}" in result.stdout
    assert "--program-file" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--smoke-test" in result.stdout
