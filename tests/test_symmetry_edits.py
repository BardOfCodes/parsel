import sympy as sp

from parsel.edit_dsl.language import SymGroupEdit
from parsel.edit_dsl.edits import ChangeCount
from parsel.shape_system.relations import TranslationSymmetry


def test_sym_group_count_can_keep_group_extent_fixed(
    box_factory,
    assert_symbolic_matrix_equal,
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
        relation_index=9,
    )

    edits = SymGroupEdit(
        group,
        change_type="count",
        extend_from="keep_fixed",
        amount=1,
    )

    assert len(edits) == 1
    assert isinstance(edits[0], ChangeCount)
    assert edits[0].stretch_annotation == "keep_fixed"

    original_start, _, original_end, _, _ = relation.static_expression()
    edits[0].propagate()
    start, _, end, delta, count = relation.dynamic_expression()

    assert count == 4
    assert start == original_start
    assert end == original_end
    assert_symbolic_matrix_equal(
        delta,
        (original_end - original_start) / 3,
    )
