import numpy as np
import sympy as sp

from parsel.edit_dsl.language import (
    AxisScale,
    PlaneScale,
    Shear,
    Translate,
    VolumeScale,
)
from parsel.edit_dsl.edits import (
    MAIN_VAR,
    FaceTranslate,
    PartRotate,
    PartScale1D,
    PartScale2D,
    PartScale3D,
    PartShear,
    PartTranslate,
    load_edits,
)
from parsel.shape_system.geometry import (
    RightUnitVector,
    UpUnitVector,
)


def test_translate_moves_every_proxy_vertex(box_factory, evaluate_matrix):
    part = box_factory("box")
    edit = PartTranslate(part, direction=RightUnitVector(), amount=MAIN_VAR)
    edit.propagate()

    static = evaluate_matrix(part.primitive.static_expression())
    dynamic = evaluate_matrix(
        part.primitive.dynamic_expression(),
        {MAIN_VAR: 0.25},
    )
    np.testing.assert_allclose(dynamic, static + np.array([0.25, 0, 0]))


def test_rotation_uses_axis_and_origin(box_factory, evaluate_matrix):
    part = box_factory("box")
    edit = PartRotate(
        part,
        rotation_origin=sp.zeros(1, 3),
        rotation_axis=sp.Matrix([[0, 0, 1]]),
        amount=sp.pi / 2,
    )

    rotated = evaluate_matrix(edit.apply(sp.Matrix([[1, 0, 0]])))
    np.testing.assert_allclose(rotated, [[0, 1, 0]], atol=1e-10)


def test_axis_plane_and_volume_scaling(box_factory, evaluate_matrix):
    part = box_factory("box")
    points = sp.Matrix([[2, 3, 4]])
    origin = sp.zeros(1, 3)

    axis_scale = PartScale1D(
        part,
        origin=origin,
        direction=RightUnitVector(),
        amount=sp.Rational(1, 2),
    )
    plane_scale = PartScale2D(
        part,
        origin=origin,
        plane_normal=sp.Matrix([[0, 0, 1]]),
        amount=sp.Rational(1, 2),
    )
    volume_scale = PartScale3D(
        part,
        origin=origin,
        amount=sp.Rational(1, 2),
    )

    np.testing.assert_allclose(evaluate_matrix(axis_scale.apply(points)), [[3, 3, 4]])
    np.testing.assert_allclose(
        evaluate_matrix(plane_scale.apply(points)),
        [[3, 4.5, 4]],
    )
    np.testing.assert_allclose(
        evaluate_matrix(volume_scale.apply(points)),
        [[3, 4.5, 6]],
    )


def test_shear_uses_distance_from_shear_plane(box_factory, evaluate_matrix):
    part = box_factory("box")
    edit = PartShear(
        part,
        shear_plane_origin=sp.zeros(1, 3),
        shear_plane_normal=UpUnitVector(),
        shear_direction=RightUnitVector(),
        amount=sp.Rational(1, 2),
    )

    sheared = evaluate_matrix(edit.apply(sp.Matrix([[2, 4, 6]])))
    np.testing.assert_allclose(sheared, [[4, 4, 6]])


def test_restricted_face_translate_only_moves_selected_face(
    box_factory,
    evaluate_matrix,
):
    part = box_factory("box")
    edit = Translate(
        part.face("right"),
        direction=UpUnitVector(),
        amount=sp.Rational(1, 2),
    )
    assert isinstance(edit, FaceTranslate)

    static = evaluate_matrix(part.primitive.static_expression())
    dynamic = evaluate_matrix(edit.apply(part.primitive.static_expression()))
    right_indices = set(part.name_to_indices["right"])
    for index in range(static.shape[0]):
        expected = static[index] + np.array([0, 0.5, 0]) if index in right_indices else static[index]
        np.testing.assert_allclose(dynamic[index], expected)


def test_edit_wrappers_map_to_concrete_operations(box_factory):
    part = box_factory("box")
    origin = part.center()

    assert isinstance(Translate(part, RightUnitVector()), PartTranslate)
    assert isinstance(
        AxisScale(part, "contract", origin, RightUnitVector()),
        PartScale1D,
    )
    assert isinstance(
        PlaneScale(part, "expand", origin, UpUnitVector()),
        PartScale2D,
    )
    assert isinstance(VolumeScale(part, "expand", origin), PartScale3D)
    assert isinstance(
        Shear(part, RightUnitVector(), origin, UpUnitVector()),
        PartShear,
    )
    assert AxisScale(part, "contract", origin, RightUnitVector()).amount == -MAIN_VAR


def test_edit_save_format_round_trips_to_equivalent_shape(
    box_factory,
    root_factory,
    assert_symbolic_matrix_equal,
):
    old_part = box_factory("box", part_index=7)
    old_shape = root_factory([old_part])
    original = PartTranslate(
        old_part,
        direction=RightUnitVector(),
        amount=2 * MAIN_VAR,
    )

    new_part = box_factory("box", part_index=7)
    new_shape = root_factory([new_part])
    loaded, operands = load_edits(new_shape, [original.save_format()])

    assert old_shape.get("box") is old_part
    assert len(loaded) == 1
    assert operands == [new_part]
    assert isinstance(loaded[0], PartTranslate)
    assert loaded[0].operand is new_part
    assert loaded[0].amount == 2 * MAIN_VAR
    assert_symbolic_matrix_equal(loaded[0].direction, RightUnitVector())
