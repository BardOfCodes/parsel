import pytest

from parsel.shape_system.constants import RESOLVE_RELATION
from parsel.edit_dsl.edits import KeepFixed, PartTranslate
from parsel.shape_system.geometry import Plane, RightUnitVector, UpUnitVector
from parsel.edit_propagation.proposer import EditProposer
from parsel.shape_system.relations import ReflectionSymmetry
from parsel.workflow import run_edit_workflow


class RecordedPromptDecisions:
    """Offline stand-in for the prompting decisions used by the workflow."""

    def __init__(self, seed_edit):
        self.seed_edit = seed_edit
        self.checked_relation_states = []

    def api_call_initialize_parallel(self, shape, edit_request):
        return [self.seed_edit], {}, []

    def api_call_check_relation(
        self,
        shape,
        edit_request,
        relation,
        valid_states,
    ):
        self.checked_relation_states.append(tuple(valid_states))
        return RESOLVE_RELATION, []

    def api_call_finish_algo(self, shape, edit_request, all_edits):
        return True, None, []


def test_workflow_propagates_seed_edit_across_reflection(
    box_factory,
    root_factory,
):
    ground = box_factory(
        "ground",
        center=(0, -5, 0),
        size=(10, 1, 10),
        part_index=0,
    )
    left = box_factory("left", center=(-2, 0, 0), part_index=1)
    right = box_factory("right", center=(2, 0, 0), part_index=2)
    shape = root_factory([ground, left, right])
    relation = ReflectionSymmetry(
        (left.primitive, right.primitive),
        Plane(point=[0, 0, 0], normal=RightUnitVector()),
        relation_index=5,
    )
    prompt_decisions = RecordedPromptDecisions(
        PartTranslate(left, direction=UpUnitVector()),
    )

    edits, logs, used_breaking_edit, final_state = run_edit_workflow(
        shape,
        "move the reflected pair upward",
        prompt_decisions,
        EditProposer(heuristic="ARAP"),
    )

    assert [(type(edit), edit.operand.label) for edit in edits] == [
        (KeepFixed, "ground"),
        (PartTranslate, "left"),
        (PartTranslate, "right"),
    ]
    assert prompt_decisions.checked_relation_states == [(0, RESOLVE_RELATION)]
    assert relation.broken() is False
    assert logs == []
    assert used_breaking_edit is False
    assert final_state == {
        "r_5": 1,
        "p_0": 1,
        "p_1": 1,
        "p_2": 1,
    }


def test_workflow_has_an_explicit_iteration_limit(
    box_factory,
    root_factory,
):
    ground = box_factory(
        "ground",
        center=(0, -5, 0),
        size=(10, 1, 10),
        part_index=0,
    )
    box = box_factory("box", part_index=1)
    shape = root_factory([ground, box])

    class NeverFinishes(RecordedPromptDecisions):
        def api_call_finish_algo(self, shape, edit_request, all_edits):
            return False, None, []

    with pytest.raises(
        RuntimeError,
        match="did not finish within 2 iterations",
    ):
        run_edit_workflow(
            shape,
            "move the box upward",
            NeverFinishes(PartTranslate(box, direction=UpUnitVector())),
            EditProposer(heuristic="ARAP"),
            max_iterations=2,
        )
