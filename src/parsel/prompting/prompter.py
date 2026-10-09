from collections import Counter
import multiprocessing as mp
import sqlite3
from string import Template
import time

import anthropic
import openai

from ..shape_system.constants import RESOLVE_RELATION
from ..edit_dsl.edits import Edit, EditGen, KeepFixed, PartEdit
from .context import (
    get_relation_in_detail_init,
    shape_to_file_hier,
)
from ..shape_system.relations import (
    PrimitiveRelation,
    ReflectionSymmetry,
    RotationSymmetry,
    TranslationSymmetry,
)
from . import state
from .api import EDIT_API, INIT_EDIT_API, SHAPE_API
from .clients import api_log_path, make_api_call
from .errors import APIFailure, PromptSyntaxError
from .parsing import response_to_snippet, snippet_to_variable
from .prompts import initialize, set_edit_types, set_relations


_TYPE_HINTS = {
    "keep_fixed",
    "translate",
    "tilt",
    "scale",
    "change_count",
    "change_delta",
}


class Prompter:
    """The single prompting workflow supported by Parsel."""

    def __init__(
        self,
        provider,
        key,
        model,
        temperature=0.0,
        seed=42,
        max_tries=2,
        repetitions=3,
        client=None,
    ):
        if repetitions < 1 or repetitions % 2 == 0:
            raise ValueError("repetitions must be a positive odd integer")
        if max_tries < 1:
            raise ValueError("max_tries must be positive")
        if provider not in {"openai", "anthropic"}:
            raise ValueError("provider must be 'openai' or 'anthropic'")

        self.provider = provider
        self.key = key
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tries = max_tries
        self.repetitions = repetitions
        self.relation_to_option = {}
        self.client = client

    def _get_client(self):
        if self.client is None:
            if self.provider == "openai":
                self.client = openai.OpenAI(api_key=self.key)
            else:
                self.client = anthropic.Client(api_key=self.key)
        return self.client

    def _worker_config(self):
        return {
            "provider": self.provider,
            "key": self.key,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tries": self.max_tries,
            "repetitions": self.repetitions,
        }

    def create_new_init_prompt(self, shape, edit_request):
        instructions = Template(initialize.get_instructions()).substitute(
            {
                "shape_class": shape.label,
                "shape_specification": shape_to_file_hier(shape),
                "edit_request": edit_request,
                "SHAPE_API": SHAPE_API,
                "EDIT_API": EDIT_API,
                "API": INIT_EDIT_API,
            }
        )
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions},
        ]

    def create_new_edit_types_prompt(self, shape, edit_request):
        instructions = Template(set_edit_types.get_instructions()).substitute(
            {
                "shape_class": shape.label,
                "shape_specification": shape_to_file_hier(shape),
                "edit_request": edit_request,
            }
        )
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions},
        ]

    def create_check_relation_prompt(
        self,
        shape,
        edit_request,
        relation,
    ):
        if isinstance(relation, ReflectionSymmetry):
            template = set_relations.ref_instructions()
        elif isinstance(
            relation,
            (TranslationSymmetry, RotationSymmetry),
        ):
            template = set_relations.rot_trans_instructions()
        else:
            raise ValueError(
                f"Unsupported prompted relation: {type(relation).__name__}"
            )

        instructions = Template(template).substitute(
            {
                "shape_class": shape.label,
                "shape_specification": shape_to_file_hier(shape),
                "edit_request": edit_request,
                "relation_in_detail": get_relation_in_detail_init(
                    relation,
                    shape,
                ),
            }
        )
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": instructions},
        ]

    def parse_initial_instructions(self, shape, response):
        edit = snippet_to_variable(
            response_to_snippet(response),
            shape,
            "primary_edit",
        )
        if not isinstance(edit, Edit):
            raise PromptSyntaxError("primary_edit must be an edit DSL value")
        return edit

    def parse_init_edit_hints(self, shape, response):
        hints = snippet_to_variable(
            response_to_snippet(response),
            shape,
            "type_hints",
        )
        if not isinstance(hints, dict):
            raise PromptSyntaxError("type_hints must be a dictionary")

        normalized = {}
        for name, edit_type in hints.items():
            if not isinstance(name, str):
                raise PromptSyntaxError("type_hints keys must be part names")
            if not isinstance(edit_type, str) or edit_type not in _TYPE_HINTS:
                raise PromptSyntaxError(
                    f"Unknown edit type hint for {name}: {edit_type}"
                )
            try:
                part = shape.get(name)
            except ValueError as error:
                raise PromptSyntaxError(
                    f"Unknown part in type_hints: {name}"
                ) from error
            normalized[part.full_label] = edit_type
        return normalized

    def parse_check_relation(self, shape, response):
        relation_state = snippet_to_variable(
            response_to_snippet(response),
            shape,
            "relation_state",
        )
        if type(relation_state) is not int or relation_state not in {1, 2, 3}:
            raise PromptSyntaxError("relation_state must be 1, 2, or 3")
        return relation_state

    def _call_and_parse(self, shape, messages, parse):
        current_messages = list(messages)
        last_error = None
        for attempt in range(self.max_tries):
            content, role, completed = make_api_call(
                provider=self.provider,
                client=self._get_client(),
                model=self.model,
                messages=current_messages,
                temperature=self.temperature,
                seed=self.seed + attempt,
            )
            if not completed:
                last_error = APIFailure(
                    "The model response stopped before completion."
                )
                continue
            try:
                return parse(shape, content), content
            except PromptSyntaxError as error:
                last_error = error
                current_messages.extend(
                    [
                        {"role": role, "content": content},
                        {
                            "role": "user",
                            "content": (
                                "The output did not match the required edit "
                                f"DSL: {error}"
                            ),
                        },
                    ]
                )
                state.record_error()
        raise APIFailure(
            f"Could not parse a valid model response after {self.max_tries} "
            "attempts."
        ) from last_error

    def _get_new_edit_types(self, shape, messages):
        return self._call_and_parse(
            shape,
            messages,
            self.parse_init_edit_hints,
        )

    def _get_primary_edit(self, shape, messages):
        return self._call_and_parse(
            shape,
            messages,
            self.parse_initial_instructions,
        )

    def _get_check_relation_option(self, shape, messages):
        return self._call_and_parse(
            shape,
            messages,
            self.parse_check_relation,
        )

    def api_call_initialize_parallel(
        self,
        shape,
        edit_request,
    ):
        fixed_parts, type_hints, edit_specs, relation_states = (
            self._api_call_initialize_parallel(
                shape,
                edit_request,
                self._worker_config(),
            )
        )
        self.relation_to_option = relation_states

        edits = []
        for part_name in fixed_parts:
            edits.append(KeepFixed(shape.get(part_name)))
        for edit_gen, operand_type, index in edit_specs:
            if operand_type == "part":
                matches = [
                    part
                    for part in shape.partset
                    if part.part_index == index
                ]
            else:
                matches = [
                    relation
                    for relation in shape.all_relations()
                    if relation.relation_index == index
                ]
            if not matches:
                raise APIFailure(
                    f"Prompt output referred to unknown {operand_type} {index}."
                )
            edits.append(edit_gen.employ(matches[0]))
        return edits, type_hints, []

    def _api_call_initialize_parallel(
        self,
        shape,
        edit_request,
        prompter_info,
    ):
        n_rows = get_n_rows()
        started = time.time()
        queue = mp.Manager().Queue()
        processes = []

        for task in ("TYPE_HINTS", "INIT"):
            for _ in range(self.repetitions):
                processes.append(
                    mp.Process(
                        target=prompt_worker,
                        args=(
                            self.__class__,
                            prompter_info,
                            shape,
                            edit_request,
                            task,
                            queue,
                        ),
                    )
                )
        for relation in shape.all_relations():
            if isinstance(relation, PrimitiveRelation):
                for _ in range(self.repetitions):
                    processes.append(
                        mp.Process(
                            target=relation_worker,
                            args=(
                                self.__class__,
                                prompter_info,
                                shape,
                                edit_request,
                                relation.relation_index,
                                queue,
                            ),
                        )
                    )

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        outputs = []
        while not queue.empty():
            outputs.append(queue.get())
        state.add_api_time(time.time() - started)

        hint_outputs = [
            value
            for value, metadata in outputs
            if metadata["step"] == "TYPE_HINTS"
        ]
        edit_outputs = [
            value
            for value, metadata in outputs
            if metadata["step"] == "INIT"
        ]
        relation_outputs = [
            value
            for value, metadata in outputs
            if metadata["step"] == "check_relation"
        ]
        if not hint_outputs or not edit_outputs:
            raise APIFailure(
                "Parallel prompting did not return both type hints and a "
                "primary edit."
            )

        type_hints = _vote_type_hints(hint_outputs)
        fixed_parts = sorted(
            name
            for name, edit_type in type_hints.items()
            if edit_type == "keep_fixed"
        )
        type_hints = {
            name: edit_type
            for name, edit_type in type_hints.items()
            if edit_type != "keep_fixed"
        }
        edit_specs = _vote_edit_specs(edit_outputs)
        relation_states = _vote_relation_states(relation_outputs)

        update_state(n_rows)
        return fixed_parts, type_hints, edit_specs, relation_states

    def api_call_check_relation(
        self,
        shape,
        edit_request,
        relation,
        valid_states,
    ):
        if not isinstance(relation, PrimitiveRelation):
            return min(valid_states), [{"step": "check_relation"}]

        selected = self.relation_to_option.get(
            relation.relation_index,
            RESOLVE_RELATION,
        )
        if (
            not isinstance(relation, ReflectionSymmetry)
            and 2 in valid_states
            and selected == RESOLVE_RELATION
        ):
            selected = 2
        if selected not in valid_states:
            selected = valid_states[0]
        return selected, [{"step": "check_relation"}]

    def api_call_finish_algo(self, shape, edit_request, all_edits):
        return True, None, []


def _vote_type_hints(outputs):
    threshold = len(outputs) // 2 + 1
    votes = Counter(
        (part_name, edit_type)
        for output in outputs
        for part_name, edit_type in output.items()
    )
    return {
        part_name: edit_type
        for (part_name, edit_type), count in votes.items()
        if count >= threshold
    }


def _edit_vote_key(edit_specs):
    return tuple(
        (
            edit_gen.edit_class.__name__,
            operand_type,
            str(index),
        )
        for edit_gen, operand_type, index in edit_specs
    )


def _vote_edit_specs(outputs):
    vote_counts = Counter(_edit_vote_key(output) for output in outputs)
    ranked = vote_counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise APIFailure("Primary-edit prompting ended in a tie.")
    selected_key, _ = ranked[0]
    return next(
        output
        for output in outputs
        if _edit_vote_key(output) == selected_key
    )


def _vote_relation_states(outputs):
    grouped = {}
    for relation_index, state in outputs:
        grouped.setdefault(relation_index, []).append(state)
    selected = {}
    for relation_index, states in grouped.items():
        ranked = Counter(states).most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            raise APIFailure(
                f"Relation {relation_index} prompting ended in a tie."
            )
        selected[relation_index] = ranked[0][0]
    return selected


def prompt_worker(
    prompter_class,
    prompter_info,
    shape,
    edit_request,
    task,
    queue,
):
    prompter = prompter_class(**prompter_info)
    if task == "TYPE_HINTS":
        messages = prompter.create_new_edit_types_prompt(
            shape,
            edit_request,
        )
        type_hints, _ = prompter._get_new_edit_types(shape, messages)
        for part in shape.partset:
            if not part.partset:
                continue
            if part.full_label in type_hints:
                for child in part.partset:
                    type_hints[child.full_label] = type_hints[part.full_label]
            else:
                child_types = {
                    type_hints.get(child.full_label)
                    for child in part.partset
                }
                if len(child_types) == 1:
                    type_hints[part.full_label] = child_types.pop()
            if (
                hasattr(part, "core_relation")
                and part.full_label in type_hints
            ):
                type_hints[part.core_relation.full_label] = type_hints[
                    part.full_label
                ]
        queue.put((type_hints, {"step": "TYPE_HINTS"}))
        return

    if task != "INIT":
        raise ValueError(f"Unknown prompting task: {task}")

    messages = prompter.create_new_init_prompt(shape, edit_request)
    edit, _ = prompter._get_primary_edit(shape, messages)
    if isinstance(edit, PartEdit):
        operand_type = "part"
        index = edit.operand.part_index
    else:
        operand_type = "relation"
        index = edit.operand.relation_index
    edit_specs = [
        (
            EditGen(
                edit.__class__,
                edit.params,
                amount=edit.amount,
            ),
            operand_type,
            index,
        )
    ]
    queue.put((edit_specs, {"step": "INIT"}))


def relation_worker(
    prompter_class,
    prompter_info,
    shape,
    edit_request,
    relation_index,
    queue,
):
    prompter = prompter_class(**prompter_info)
    relation = next(
        relation
        for relation in shape.all_relations()
        if relation.relation_index == relation_index
    )
    messages = prompter.create_check_relation_prompt(
        shape,
        edit_request,
        relation,
    )
    selected, _ = prompter._get_check_relation_option(
        shape,
        messages,
    )
    queue.put(
        (
            (relation_index, selected),
            {"step": "check_relation"},
        )
    )


def get_n_rows():
    with sqlite3.connect(api_log_path()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS api_calls "
            "(call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, total_tokens INTEGER)"
        )
        cursor.execute("SELECT COUNT(*) FROM api_calls")
        return cursor.fetchone()[0]


def update_state(n_rows):
    with sqlite3.connect(api_log_path()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS api_calls "
            "(call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, total_tokens INTEGER)"
        )
        cursor.execute(
            "SELECT * FROM api_calls WHERE rowid > ?",
            (n_rows,),
        )
        new_rows = cursor.fetchall()
    state.record_usage(
        sum(row[2] for row in new_rows),
        sum(row[3] for row in new_rows),
        calls=len(new_rows),
    )
