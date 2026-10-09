from .shape_system.constants import (
    NOTHING_TO_DO,
    PART_ACTIVE,
    PART_EDITED,
    RELATION_BROKEN,
    RELATION_REJECTED,
    RELATION_RETAINED,
    RELATION_STABLE,
    RELATION_UNCHECKED,
    RESOLVE_RELATION,
    UPDATE_RELATION,
)
from .edit_dsl.edits import KeepFixed
from .edit_propagation.proposer import EditProposer
from .shape_system.relations import FeatureRelation, PrimitiveRelation
from .shape_system.shape import Part


def infer_relation_state(
    shape,
    edit_request,
    prompter,
    log_info,
    broken_relations,
):
    """Classify broken relations and perform direct relation edits."""
    relations_to_resolve = []
    relations_to_update = []

    for relation in broken_relations:
        edit_counts = [
            len(part.primitive.edit_sequence)
            for part in relation.parts
        ]
        if min(edit_counts) > 0:
            relation.state[2] = RELATION_REJECTED
            relation.state[3] = NOTHING_TO_DO

        if relation.state[2] == RELATION_UNCHECKED:
            valid_states = (
                [] if isinstance(relation, FeatureRelation) else [NOTHING_TO_DO]
            )
            if relation.resolvable():
                valid_states.append(RESOLVE_RELATION)
            if relation.updatable():
                valid_states.append(UPDATE_RELATION)

            if not valid_states:
                selected_state = NOTHING_TO_DO
            elif len(valid_states) == 1:
                selected_state = valid_states[0]
            else:
                selected_state, new_logs = prompter.api_call_check_relation(
                    shape,
                    edit_request,
                    relation,
                    valid_states,
                )
                log_info.extend(new_logs)
                if selected_state not in valid_states:
                    raise ValueError(
                        "Prompter selected an invalid relation state: "
                        f"{selected_state}; expected one of {valid_states}"
                    )

            if selected_state == NOTHING_TO_DO:
                relation.state[2] = RELATION_REJECTED
            elif selected_state == RESOLVE_RELATION:
                relation.state[2] = RELATION_RETAINED
                relation.state[3] = RESOLVE_RELATION
                relations_to_resolve.append(relation)
            elif selected_state == UPDATE_RELATION:
                relation.state[2] = RELATION_RETAINED
                relation.state[3] = UPDATE_RELATION
                relations_to_update.append(relation)
        elif relation.state[2] == RELATION_RETAINED:
            if (
                relation.state[3] == UPDATE_RELATION
                and relation.updatable()
            ):
                relations_to_update.append(relation)
            elif (
                relation.state[3] == RESOLVE_RELATION
                and relation.resolvable()
            ):
                relations_to_resolve.append(relation)
            else:
                relation.state[2] = RELATION_REJECTED

    unresolved = []
    new_edits = []
    for relation in relations_to_resolve:
        if relation.automatically_resolvable():
            new_edits.extend(relation.resolve())
            break
        unresolved.append(relation)

    if not new_edits:
        for relation in relations_to_update:
            if relation.automatically_updatable():
                new_edits.extend(relation.update())
                break
            unresolved.append(relation)

    return ([] if new_edits else unresolved), new_edits


def _initialize(shape, edit_request, prompter):
    return prompter.api_call_initialize_parallel(
        shape,
        edit_request,
    )


def _propagate(shape, edits):
    for edit in edits:
        edit.operand.state[1] = PART_EDITED

    for part in shape.partset:
        if part.sub_parts and any(
            child.state[1] == PART_EDITED for child in part.sub_parts
        ):
            shape.deactivate_parent(part)

    shape.clean_up_motion()
    for edit in edits:
        edit.propagate()


def _update_relation_status(shape):
    for relation in shape.all_relations(only_active=True):
        relation.state[1] = (
            RELATION_BROKEN if relation.broken() else RELATION_STABLE
        )

    potential = [
        relation
        for relation in shape.all_relations(only_active=True)
        if (
            relation.state[1] == RELATION_BROKEN
            and relation.state[2] != RELATION_REJECTED
        )
    ]
    for relation in potential:
        if not relation.resolvable() and not relation.updatable():
            relation.state[2] = RELATION_REJECTED
            relation.state[3] = NOTHING_TO_DO
    return [
        relation
        for relation in potential
        if relation.resolvable() or relation.updatable()
    ]


def _solve_next_edit(part, shape, proposer, type_hints):
    edit_type = type_hints.get(part.full_label)
    solved_edit, least_breaking = proposer.solve_edit_by_type(
        part,
        edit_type,
        shape,
    )
    if solved_edit is not None:
        return solved_edit, False
    if least_breaking is not None:
        return least_breaking[0], True
    return KeepFixed(part), False


def _is_unedited(part_or_relation):
    if isinstance(part_or_relation, Part):
        return len(part_or_relation.primitive.edit_sequence) == 0
    if isinstance(part_or_relation, PrimitiveRelation):
        return len(part_or_relation.edit_sequence) == 0
    return False


def _final_state(shape):
    state = {}
    for relation in shape.all_relations(only_active=True):
        retained = (
            relation.state[2] != RELATION_REJECTED
            and relation.state[1] != RELATION_BROKEN
        )
        state[f"r_{relation.relation_index}"] = int(retained)
    for part in shape.partset:
        state[f"p_{part.part_index}"] = int(part.state[0] == PART_ACTIVE)
    return state


def run_edit_workflow(
    shape: Part,
    edit_request: str,
    prompter,
    edit_proposer: EditProposer,
    *,
    max_iterations=100,
):
    """Map a natural-language request to a propagated analytic edit program."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    ground_edit = KeepFixed(shape.get("ground"))
    all_edits = [ground_edit]
    log_info = []
    used_breaking_edit = False

    seed_edits, type_hints, initialization_logs = _initialize(
        shape,
        edit_request,
        prompter,
    )
    all_edits.extend(seed_edits)
    log_info.extend(initialization_logs)

    active_relations = shape.all_relations(only_active=True)
    relation_relations = set().union(
        *(relation._relations for relation in active_relations)
    )

    for _ in range(max_iterations):
        _propagate(shape, all_edits)

        for relation in relation_relations:
            if relation.broken():
                all_edits.extend(relation.resolve())

        broken_relations = _update_relation_status(shape)
        if not broken_relations:
            complete, new_edits, new_logs = prompter.api_call_finish_algo(
                shape,
                edit_request,
                all_edits,
            )
            log_info.extend(new_logs)
            if complete:
                break
            if new_edits is not None:
                if not isinstance(new_edits, list):
                    new_edits = [new_edits]
                all_edits.extend(new_edits)
            continue

        unresolved, relation_edits = infer_relation_state(
            shape,
            edit_request,
            prompter,
            log_info,
            broken_relations,
        )
        if relation_edits:
            all_edits.extend(relation_edits)
            continue
        if not unresolved:
            continue

        edit_proposer.register_broken_relations(unresolved)
        part_to_edit = edit_proposer.propose_part_to_edit(shape)
        selected_edits = []

        if part_to_edit is not None and _is_unedited(part_to_edit):
            if isinstance(part_to_edit, Part):
                unchecked_contacts = [
                    relation
                    for relation in part_to_edit.all_relations(
                        only_active=True
                    )
                    if (
                        relation.state[2] == RELATION_UNCHECKED
                        and isinstance(relation, FeatureRelation)
                        and relation.other_part_edited(part_to_edit)
                    )
                ]
                if unchecked_contacts:
                    _, relation_edits = infer_relation_state(
                        shape,
                        edit_request,
                        prompter,
                        log_info,
                        unchecked_contacts,
                    )
                    all_edits.extend(relation_edits)
                    edit_proposer.refresh()
                    continue

            selected_edit, breaking = _solve_next_edit(
                part_to_edit,
                shape,
                edit_proposer,
                type_hints,
            )
            selected_edits.append(selected_edit)
            used_breaking_edit = used_breaking_edit or breaking

        if selected_edits:
            all_edits.extend(selected_edits)
        else:
            for relation in unresolved:
                relation.state[2] = RELATION_REJECTED
                relation.state[3] = NOTHING_TO_DO
        edit_proposer.refresh()
    else:
        raise RuntimeError(
            f"Edit workflow did not finish within {max_iterations} iterations"
        )

    return (
        all_edits,
        log_info,
        used_breaking_edit,
        _final_state(shape),
    )
