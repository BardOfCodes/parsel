import sympy as sp

import numpy as np
import pytest

from parsel.shape_system.constants import PART_INACTIVE
from parsel.edit_dsl.edits import (
    ChangeCount,
    ChangeDelta,
    MAIN_VAR,
    PartTranslate,
)
from parsel.shape_system.geometry import (
    Line,
    Plane,
    RightUnitVector,
    UpUnitVector,
)
from parsel.shape_system.relations import (
    PointContact,
    ReflectionSymmetry,
    RotationSymmetry,
    TranslationSymmetry,
)


@pytest.fixture(autouse=True)
def _relation_operations_are_quiet(capsys):
    yield
    assert capsys.readouterr().out == ""


def _center_weights():
    return [sp.Rational(1, 8)] * 8


def test_point_contact_detects_relative_motion(
    box_factory,
    point_feature_factory,
):
    first = box_factory("first", part_index=1)
    second = box_factory("second", part_index=2)
    first_point = point_feature_factory(first, _center_weights())
    second_point = point_feature_factory(second, _center_weights())
    relation = PointContact((first_point, second_point), relation_index=3)

    assert relation.broken() is False

    PartTranslate(first, direction=UpUnitVector(), amount=MAIN_VAR).propagate()

    assert relation.broken() is True
    assert relation.resolvable() is True
    assert relation.updatable() is True


def test_reflection_symmetry_resolves_an_edit_on_one_side(box_factory):
    left = box_factory("left", center=(-2, 0, 0), part_index=1)
    right = box_factory("right", center=(2, 0, 0), part_index=2)
    relation = ReflectionSymmetry(
        (left.primitive, right.primitive),
        Plane(point=[0, 0, 0], normal=RightUnitVector()),
        relation_index=4,
    )

    assert relation.broken() is False

    PartTranslate(left, direction=UpUnitVector(), amount=MAIN_VAR).propagate()
    assert relation.broken() is True

    mirrored_edits = relation.resolve()
    assert len(mirrored_edits) == 1
    mirrored_edits[0].propagate()

    assert mirrored_edits[0].operand is right
    assert mirrored_edits[0].amount == MAIN_VAR
    assert relation.broken() is False


def test_reflection_symmetry_rejects_resolution_when_both_parts_are_edited(
    box_factory,
):
    left = box_factory("left", center=(-2, 0, 0), part_index=1)
    right = box_factory("right", center=(2, 0, 0), part_index=2)
    relation = ReflectionSymmetry(
        (left.primitive, right.primitive),
        Plane(point=[0, 0, 0], normal=RightUnitVector()),
        relation_index=5,
    )

    PartTranslate(left, direction=UpUnitVector(), amount=MAIN_VAR).propagate()
    PartTranslate(right, direction=UpUnitVector(), amount=MAIN_VAR).propagate()

    assert relation.resolvable() is False
    with pytest.raises(ValueError, match="both parts"):
        relation.resolve()


def test_translation_symmetry_resolves_member_edit(
    box_factory,
    root_factory,
):
    members = [
        box_factory("first", center=(-2, 0, 0), part_index=1),
        box_factory("second", center=(0, 0, 0), part_index=2),
        box_factory("third", center=(2, 0, 0), part_index=3),
    ]
    group = box_factory(
        "group",
        center=(0, 0, 0),
        size=(6, 2, 2),
        part_index=4,
    )
    group.sub_parts = set(members)
    for member in members:
        member.parent = group
    root_factory([group])
    relation = TranslationSymmetry(
        [member.primitive for member in members],
        delta=sp.Matrix([[2, 0, 0]]),
        parent_part=group,
        relation_index=5,
    )
    group.state[0] = PART_INACTIVE

    assert relation.broken() is False

    PartTranslate(members[0], direction=UpUnitVector(), amount=MAIN_VAR).propagate()
    assert relation.broken() is True
    propagated = relation.resolve()
    for edit in propagated:
        edit.propagate()

    assert {edit.operand for edit in propagated} == set(members[1:])
    assert relation.broken() is False


def test_rotation_symmetry_resolves_member_edit(
    box_factory,
    root_factory,
):
    centers = [(2, 0, 0), (0, 0, -2), (-2, 0, 0), (0, 0, 2)]
    members = [
        box_factory(f"member_{index}", center=center, size=(1, 1, 1), part_index=index)
        for index, center in enumerate(centers)
    ]
    group = box_factory(
        "group",
        center=(0, 0, 0),
        size=(6, 2, 6),
        part_index=10,
    )
    group.sub_parts = set(members)
    for member in members:
        member.parent = group
    root_factory([group])
    relation = RotationSymmetry(
        [member.primitive for member in members],
        axis=Line(point=[0, 0, 0], direction=[0, 1, 0]),
        angle=np.pi / 2,
        parent_part=group,
        relation_index=6,
    )
    group.state[0] = PART_INACTIVE

    assert relation.closed_loop is True
    assert relation.broken() is False

    PartTranslate(members[0], direction=UpUnitVector(), amount=MAIN_VAR).propagate()
    assert relation.broken() is True
    propagated = relation.resolve()
    for edit in propagated:
        edit.propagate()

    assert {edit.operand for edit in propagated} == set(members[1:])
    assert relation.broken() is False


def test_translation_symmetry_parameter_edits(
    box_factory,
):
    members = [
        box_factory("first", center=(-2, 0, 0), part_index=1),
        box_factory("second", center=(0, 0, 0), part_index=2),
        box_factory("third", center=(2, 0, 0), part_index=3),
    ]
    group = box_factory(
        "group",
        center=(0, 0, 0),
        size=(6, 2, 2),
        part_index=4,
    )
    group.sub_parts = set(members)
    for member in members:
        member.parent = group
    relation = TranslationSymmetry(
        [member.primitive for member in members],
        delta=sp.Matrix([[2, 0, 0]]),
        parent_part=group,
        relation_index=7,
    )

    count_edit = ChangeCount(
        relation,
        stretch_annotation="center",
        relation_update_mode="BU",
        amount=1,
    )
    count_edit.propagate()
    count_parameters = relation.dynamic_expression()
    assert count_parameters[-1] == 4

    relation.edit_sequence = []
    delta_edit = ChangeDelta(
        relation,
        stretch_annotation="center",
        relation_update_mode="BU",
        amount=sp.Rational(1, 2),
    )
    delta_edit.propagate()
    delta_parameters = relation.dynamic_expression()

    start, _, end, delta, count = delta_parameters
    assert count == 3
    assert delta == sp.Matrix([[3, 0, 0]])
    assert end - start == sp.Matrix([[6, 0, 0]])
