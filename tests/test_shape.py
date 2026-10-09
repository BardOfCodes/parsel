import pytest

from parsel.shape_system.constants import (
    PART_ACTIVE,
    PART_INACTIVE,
    PART_UNEDITED,
)
from parsel.edit_dsl.edits import PartTranslate
from parsel.shape_system.geometry import RightUnitVector


def test_hierarchy_labels_lookup_and_partset(box_factory, root_factory):
    leaf = box_factory("leaf", part_index=1)
    group = root_factory([leaf], label="group", part_index=2)
    root = root_factory([group], label="root", part_index=3)

    assert leaf.full_label == "root/group/leaf"
    assert root.get("root/group/leaf") is leaf
    assert root.get("group/leaf") is leaf
    assert root.get("group") is group
    assert root.get("leaf") is leaf
    assert root.partset == {group, leaf}


def test_lookup_reports_missing_and_ambiguous_labels(
    box_factory,
    root_factory,
):
    first_leaf = box_factory("leaf", part_index=1)
    second_leaf = box_factory("leaf", part_index=2)
    first_group = root_factory(
        [first_leaf],
        label="first",
        part_index=3,
    )
    second_group = root_factory(
        [second_leaf],
        label="second",
        part_index=4,
    )
    root = root_factory(
        [first_group, second_group],
        label="root",
        part_index=5,
    )

    assert root.get("first/leaf") is first_leaf
    assert root.get("root/second/leaf") is second_leaf
    with pytest.raises(ValueError, match="ambiguous"):
        root.get("leaf")
    with pytest.raises(ValueError, match="not found"):
        root.get("missing")
    with pytest.raises(ValueError, match="not found"):
        root.get("first/missing")


def test_recursive_feature_and_primitive_collection(
    box_factory,
    root_factory,
    point_feature_factory,
):
    leaf = box_factory("leaf", part_index=1)
    group = root_factory([leaf], label="group", part_index=2)
    root = root_factory([group], label="root", part_index=3)
    feature = point_feature_factory(leaf, [1 / 8] * 8)

    assert root.features(get_all=True) == {feature}
    assert root.primitives(get_all=True) == {
        root.primitive,
        group.primitive,
        leaf.primitive,
    }


def test_clean_up_motion_recurses_over_hierarchy(box_factory, root_factory):
    first = box_factory("first", part_index=1)
    second = box_factory("second", part_index=2)
    root = root_factory([first, second])

    PartTranslate(first, RightUnitVector()).propagate()
    PartTranslate(second, RightUnitVector()).propagate()
    assert len(first.primitive.edit_sequence) == 1
    assert len(second.primitive.edit_sequence) == 1

    root.clean_up_motion()

    assert first.primitive.edit_sequence == []
    assert second.primitive.edit_sequence == []


def test_deactivate_parent_activates_its_children(box_factory, root_factory):
    first = box_factory("first", part_index=1)
    second = box_factory("second", part_index=2)
    group = root_factory([first, second], label="group", part_index=3)
    root = root_factory([group], label="root", part_index=4)

    first.state[0] = PART_INACTIVE
    second.state[0] = PART_INACTIVE
    root.deactivate_parent(group)

    assert group.state[0] == PART_INACTIVE
    assert first.state[0] == PART_ACTIVE
    assert second.state[0] == PART_ACTIVE
    assert first.state[1] == PART_UNEDITED
