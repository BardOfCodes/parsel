import sympy as sp
import pytest

from parsel.shape_system.constants import RELATION_RETAINED, RESOLVE_RELATION
from parsel.edit_dsl.edits import EditGen, MAIN_VAR, PartTranslate
from parsel.shape_system.geometry import UpUnitVector
from parsel.edit_propagation.proposer import (
    EditProposer,
    _opposite_restrictor_center,
    solve_edits,
)
from parsel.edit_propagation import proposer as proposer_module
from parsel.edit_propagation.parallel import ParallelEditProposer
from parsel.shape_system.relations import PointContact, Relation


@pytest.fixture(autouse=True)
def _propagation_operations_are_quiet(capsys):
    yield
    assert capsys.readouterr().out == ""


class _EmptyRelation(Relation):
    def gather_update_options(self):
        return []


def _center_weights():
    return [sp.Rational(1, 8)] * 8


def test_opposite_restrictor_center_handles_edges(
    box_factory,
    assert_symbolic_matrix_equal,
):
    part = box_factory("part")

    assert_symbolic_matrix_equal(
        _opposite_restrictor_center(part, ("down", "right")),
        part.edge_center("up", "left"),
    )


def test_proposer_selects_unedited_part_with_broken_contact(
    box_factory,
    root_factory,
    point_feature_factory,
):
    source = box_factory("source", part_index=1)
    target = box_factory("target", part_index=2)
    shape = root_factory([source, target])
    relation = PointContact(
        (
            point_feature_factory(source, _center_weights()),
            point_feature_factory(target, _center_weights()),
        ),
        relation_index=10,
    )
    relation.state[2] = RELATION_RETAINED
    relation.state[3] = RESOLVE_RELATION
    PartTranslate(source, direction=UpUnitVector(), amount=MAIN_VAR).propagate()

    proposer = EditProposer(heuristic="ARAP")
    proposer.register_broken_relations([relation])

    assert proposer.propose_part_to_edit(shape) is target


def test_symbolic_solver_recovers_secondary_translation_amount(
    box_factory,
    point_feature_factory,
):
    source = box_factory("source", part_index=1)
    target = box_factory("target", part_index=2)
    relation = PointContact(
        (
            point_feature_factory(source, _center_weights()),
            point_feature_factory(target, _center_weights()),
        ),
        relation_index=11,
    )
    PartTranslate(source, direction=UpUnitVector(), amount=MAIN_VAR).propagate()
    candidate = EditGen(PartTranslate, {"direction": UpUnitVector()})

    solutions = solve_edits(target, [relation], [candidate])

    assert len(solutions) == 1
    assert solutions[0].edit_class is PartTranslate
    assert sp.simplify(solutions[0].amount - MAIN_VAR) == 0
    # Candidate search must clean up its temporary edit.
    assert target.primitive.edit_sequence == []
    assert len(source.primitive.edit_sequence) == 1


def test_proposer_gathers_and_solves_typed_edit_candidates(
    box_factory,
    root_factory,
    point_feature_factory,
):
    source = box_factory("source", part_index=1)
    target = box_factory("target", part_index=2)
    shape = root_factory([source, target])
    relation = PointContact(
        (
            point_feature_factory(source, _center_weights()),
            point_feature_factory(target, _center_weights()),
        ),
        relation_index=12,
    )
    relation.state[2] = RELATION_RETAINED
    relation.state[3] = RESOLVE_RELATION
    PartTranslate(source, direction=UpUnitVector(), amount=MAIN_VAR).propagate()

    solved, least_breaking = EditProposer().solve_edit_by_type(
        target,
        "translate",
        shape,
    )

    assert isinstance(solved, PartTranslate)
    assert solved.operand is target
    assert sp.simplify(solved.amount - MAIN_VAR) == 0
    assert least_breaking is None


def test_proposer_gathers_additional_candidates_from_sibling_edit(
    box_factory,
    root_factory,
    point_feature_factory,
    monkeypatch,
):
    source = box_factory("source", part_index=1)
    target = box_factory("target", part_index=2)
    shape = root_factory([source, target])
    relation = PointContact(
        (
            point_feature_factory(source, _center_weights()),
            point_feature_factory(target, _center_weights()),
        ),
        relation_index=13,
    )
    PartTranslate(source, direction=UpUnitVector(), amount=MAIN_VAR).propagate()

    calls = []

    def gather_additional(part, edit):
        calls.append((part, edit))
        return []

    monkeypatch.setattr(proposer_module, "solve_edits", lambda *args: [])
    monkeypatch.setattr(
        proposer_module,
        "relation_based_edit_candidates",
        lambda *args: [],
    )
    monkeypatch.setattr(
        proposer_module,
        "get_additional_edit_candidates",
        gather_additional,
    )

    candidates = EditProposer().gather_options(
        target,
        [relation],
        previous_edits=None,
        shape=shape,
        type_hints={PartTranslate},
    )

    assert candidates == []
    assert calls == [(target, source.primitive.edit_sequence[0])]


@pytest.mark.parametrize(
    "proposer",
    [EditProposer(), ParallelEditProposer()],
)
def test_relation_with_no_update_candidates_returns_no_solution(proposer):
    solved, least_breaking = proposer.solve_edit_by_type(
        _EmptyRelation(),
        "translate",
        shape=None,
    )

    assert solved is None
    assert least_breaking is None
