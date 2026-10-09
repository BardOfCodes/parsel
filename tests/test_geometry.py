import numpy as np
import sympy as sp

from parsel.edit_dsl.edits import MAIN_VAR, PartTranslate
from parsel.shape_system.geometry import (
    BackUnitVector,
    DownUnitVector,
    FrontUnitVector,
    LeftToRightLine,
    LeftUnitVector,
    Line,
    Plane,
    RightUnitVector,
    UpUnitVector,
)
from parsel.shape_system.math import (
    always_real,
    always_within_bounds,
    evaluate_equals_zero,
)


def test_global_direction_convention(assert_symbolic_matrix_equal):
    assert_symbolic_matrix_equal(RightUnitVector(), [[1, 0, 0]])
    assert_symbolic_matrix_equal(LeftUnitVector(), [[-1, 0, 0]])
    assert_symbolic_matrix_equal(UpUnitVector(), [[0, 1, 0]])
    assert_symbolic_matrix_equal(DownUnitVector(), [[0, -1, 0]])
    assert_symbolic_matrix_equal(FrontUnitVector(), [[0, 0, 1]])
    assert_symbolic_matrix_equal(BackUnitVector(), [[0, 0, -1]])


def test_plane_and_line_geometric_equality():
    plane_a = Plane(point=[0, 0, 0], normal=[1, 0, 0])
    plane_b = Plane(point=[0, 2, -3], normal=[-1, 0, 0])
    assert plane_a == plane_b

    line_a = LeftToRightLine()
    line_b = Line(point=[3, 0, 0], direction=[-1, 0, 0])
    assert line_a == line_b


def test_hexahedron_exposes_named_features(box_factory, evaluate_matrix):
    part = box_factory("box", center=(1, 2, 3), size=(2, 4, 6))

    np.testing.assert_allclose(evaluate_matrix(part.center()), [[1, 2, 3]])
    np.testing.assert_allclose(evaluate_matrix(part.face_center("right")), [[2, 2, 3]])
    np.testing.assert_allclose(evaluate_matrix(part.face_center("up")), [[1, 4, 3]])
    np.testing.assert_allclose(evaluate_matrix(part.face_center("front")), [[1, 2, 6]])
    np.testing.assert_allclose(
        evaluate_matrix(part.edge_center("right", "up")),
        [[2, 4, 3]],
    )
    np.testing.assert_allclose(
        evaluate_matrix(part.corner("right", "up", "front")),
        [[2, 4, 6]],
    )


def test_harmonic_point_follows_proxy_edit(
    box_factory,
    point_feature_factory,
    evaluate_matrix,
):
    part = box_factory("box")
    center_feature = point_feature_factory(part, [sp.Rational(1, 8)] * 8)
    edit = PartTranslate(part, direction=RightUnitVector(), amount=MAIN_VAR)
    edit.propagate()

    np.testing.assert_allclose(
        evaluate_matrix(center_feature.static_expression()),
        [[0, 0, 0]],
    )
    np.testing.assert_allclose(
        evaluate_matrix(center_feature.dynamic_expression(), {MAIN_VAR: 0.25}),
        [[0.25, 0, 0]],
    )


def test_numeric_zero_check_preserves_symbolic_threshold_semantics(monkeypatch):
    samples = np.linspace(0.01, 0.5, 50).reshape(50, 1)
    monkeypatch.setattr(np.random, "uniform", lambda size: samples.copy())
    symbol = sp.Symbol("sample")

    assert evaluate_equals_zero(sp.Float("0.001") * symbol)
    assert not evaluate_equals_zero(symbol)
    assert not evaluate_equals_zero(symbol + 1)


def test_numeric_validity_checks_preserve_symbolic_semantics(monkeypatch):
    samples = np.linspace(0.01, 0.5, 50).reshape(50, 1)
    monkeypatch.setattr(np.random, "uniform", lambda size: samples.copy())
    symbol = sp.Symbol("sample")

    assert always_real(symbol)
    assert not always_real(sp.sqrt(symbol - 1))
    assert always_within_bounds(symbol)
    assert not always_within_bounds(symbol + 10)
