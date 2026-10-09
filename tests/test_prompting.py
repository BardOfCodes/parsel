import os
import queue

import pytest

from parsel.prompting.parsing import response_to_snippet
from parsel.prompting import APIFailure, Prompter, PromptSyntaxError
from parsel.prompting import state
import parsel.prompting.prompter as prompter_module
from parsel.shape_system.constants import RESOLVE_RELATION, UPDATE_RELATION
from parsel.edit_dsl.edits import MAIN_VAR, PartRotate, PartTranslate
from parsel.shape_system.geometry import Plane, RightUnitVector
from parsel.shape_system.relations import ReflectionSymmetry


@pytest.fixture
def prompter():
    return Prompter(
        provider="openai",
        key="",
        model="gpt-4o",
        repetitions=3,
    )


def test_prompts_include_shape_request_and_expected_output(
    box_factory,
    root_factory,
    prompter,
):
    box = box_factory("box", part_index=1)
    shape = root_factory([box], label="object")
    request = "move the box to the right"

    seed_messages = prompter.create_new_init_prompt(shape, request)
    type_messages = prompter.create_new_edit_types_prompt(shape, request)

    assert seed_messages[0] == {"role": "system", "content": ""}
    assert request in seed_messages[1]["content"]
    assert "object/" in seed_messages[1]["content"]
    assert "    box" in seed_messages[1]["content"]
    assert "primary_edit" in seed_messages[1]["content"]
    assert request in type_messages[1]["content"]
    assert "type_hints" in type_messages[1]["content"]


def test_prompter_parses_seed_edit_type_hints_and_relation_state(
    box_factory,
    root_factory,
    prompter,
):
    box = box_factory("box", part_index=1)
    shape = root_factory([box], label="object")

    seed = prompter.parse_initial_instructions(
        shape,
        '```python\nprimary_edit = Translate(shape.get("box"), RightUnitVector())\n```',
    )
    hints = prompter.parse_init_edit_hints(
        shape,
        '```python\ntype_hints = {"box": "translate"}\n```',
    )
    relation_state = prompter.parse_check_relation(
        shape,
        "Analysis text.\n```python\nrelation_state = 1\n```",
    )

    assert isinstance(seed, PartTranslate)
    assert seed.operand is box
    assert hints == {"object/box": "translate"}
    assert relation_state == 1


def test_prompter_parses_edit_with_intermediate_geometry(
    box_factory,
    root_factory,
    prompter,
):
    box = box_factory("box", part_index=1)
    shape = root_factory([box], label="object")

    seed = prompter.parse_initial_instructions(
        shape,
        (
            "```python\n"
            'part = shape.get("box")\n'
            'origin = part.edge_center("back", "down")\n'
            "primary_edit = Rotate(part, origin, UpUnitVector())\n"
            "```"
        ),
    )

    assert isinstance(seed, PartRotate)
    assert seed.operand is box


def test_response_parser_concatenates_multiple_python_blocks():
    response = (
        "```python\nfirst = 1\n```\n"
        "some explanation\n"
        "```python\nsecond = 2\n```"
    )

    assert response_to_snippet(response) == "first = 1\nsecond = 2"


def test_parser_rejects_missing_output_variable(
    box_factory,
    root_factory,
    prompter,
):
    shape = root_factory([box_factory("box", part_index=1)])

    with pytest.raises(PromptSyntaxError):
        prompter.parse_check_relation(
            shape,
            "```python\nunrelated_value = 1\n```",
        )


def test_parser_rejects_code_outside_the_edit_dsl(
    box_factory,
    root_factory,
    prompter,
):
    shape = root_factory([box_factory("box", part_index=1)])

    with pytest.raises(PromptSyntaxError):
        prompter.parse_check_relation(
            shape,
            '```python\nshape.label = "changed"\nrelation_state = 1\n```',
        )

    assert shape.label == "shape"


@pytest.mark.parametrize(
    "response, parser, message",
    [
        (
            "```python\nprimary_edit = 7\n```",
            "parse_initial_instructions",
            "primary_edit must be an edit DSL value",
        ),
        (
            "```python\ntype_hints = []\n```",
            "parse_init_edit_hints",
            "type_hints must be a dictionary",
        ),
        (
            '```python\ntype_hints = {"box": "warp"}\n```',
            "parse_init_edit_hints",
            "Unknown edit type hint",
        ),
        (
            '```python\ntype_hints = {"box": []}\n```',
            "parse_init_edit_hints",
            "Unknown edit type hint",
        ),
        (
            "```python\nrelation_state = 7\n```",
            "parse_check_relation",
            "relation_state must be 1, 2, or 3",
        ),
    ],
)
def test_prompt_parsers_validate_output_types(
    box_factory,
    root_factory,
    prompter,
    response,
    parser,
    message,
):
    shape = root_factory([box_factory("box", part_index=1)])

    with pytest.raises(PromptSyntaxError, match=message):
        getattr(prompter, parser)(shape, response)


def test_prompter_does_not_mutate_openai_process_state(monkeypatch):
    monkeypatch.setattr(prompter_module.openai, "api_key", "existing")
    monkeypatch.setenv("OPENAI_API_KEY", "existing-environment-key")

    prompter = Prompter(
        provider="openai",
        key="new-key",
        model="gpt-4o",
    )

    assert prompter_module.openai.api_key == "existing"
    assert os.environ["OPENAI_API_KEY"] == (
        "existing-environment-key"
    )
    assert prompter.client is None
    assert prompter._worker_config()["provider"] == "openai"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_prompter_constructs_each_supported_client(provider):
    prompter = Prompter(
        provider=provider,
        key="test-key",
        model="test-model",
    )

    client = prompter._get_client()

    assert client is prompter.client
    client.close()


def test_prompt_retry_does_not_hide_programming_errors(
    prompter,
    monkeypatch,
):
    prompter.client = object()
    state.reset()
    monkeypatch.setattr(
        prompter_module,
        "make_api_call",
        lambda **kwargs: ("response", "assistant", True),
    )

    def broken_parser(shape, response):
        raise RuntimeError("parser bug")

    with pytest.raises(RuntimeError, match="parser bug"):
        prompter._call_and_parse(None, [], broken_parser)

    assert state.error_count == 0


def test_relation_vote_ties_fail_explicitly():
    with pytest.raises(APIFailure, match="ended in a tie"):
        prompter_module._vote_relation_states(
            [(4, RESOLVE_RELATION), (4, UPDATE_RELATION)]
        )


def test_parallel_initialization_votes_without_live_api_calls(
    box_factory,
    root_factory,
    prompter,
    monkeypatch,
):
    left = box_factory("left", center=(-2, 0, 0), part_index=1)
    right = box_factory("right", center=(2, 0, 0), part_index=2)
    shape = root_factory([left, right], label="object")
    relation = ReflectionSymmetry(
        (left.primitive, right.primitive),
        Plane(point=[0, 0, 0], normal=RightUnitVector()),
        relation_index=8,
    )
    seed_spec = PartTranslate(
        left,
        direction=RightUnitVector(),
        amount=MAIN_VAR,
    ).save_format()
    relation_votes = iter(
        [RESOLVE_RELATION, UPDATE_RELATION, RESOLVE_RELATION]
    )

    class InlineManager:
        @staticmethod
        def Queue():
            return queue.Queue()

    class InlineProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

        def join(self):
            pass

    def fake_prompt_call(
        prompter_class,
        prompter_info,
        current_shape,
        edit_request,
        task,
        output_queue,
    ):
        if task == "TYPE_HINTS":
            output_queue.put(
                (
                    {
                        "object/left": "translate",
                        "object/right": "keep_fixed",
                    },
                    {"step": "TYPE_HINTS"},
                )
            )
        else:
            output_queue.put(([seed_spec], {"step": "INIT"}))

    def fake_relation_call(
        prompter_class,
        prompter_info,
        current_shape,
        edit_request,
        relation_index,
        output_queue,
    ):
        output_queue.put(
            (
                (relation_index, next(relation_votes)),
                {"step": "check_relation"},
            )
        )

    monkeypatch.setattr(prompter_module.mp, "Manager", InlineManager)
    monkeypatch.setattr(prompter_module.mp, "Process", InlineProcess)
    monkeypatch.setattr(prompter_module, "prompt_worker", fake_prompt_call)
    monkeypatch.setattr(prompter_module, "relation_worker", fake_relation_call)
    monkeypatch.setattr(prompter_module, "get_n_rows", lambda: 0)
    monkeypatch.setattr(prompter_module, "update_state", lambda n_rows: None)

    fixed, type_hints, seeds, relation_states = (
        prompter._api_call_initialize_parallel(
            shape,
            "move the left side right",
            {},
        )
    )

    assert fixed == ["object/right"]
    assert type_hints == {"object/left": "translate"}
    assert len(seeds) == 1
    assert seeds[0][0].edit_class is PartTranslate
    assert seeds[0][1:] == ("part", left.part_index)
    assert relation_states == {relation.relation_index: RESOLVE_RELATION}
