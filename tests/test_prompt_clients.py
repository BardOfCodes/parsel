from types import SimpleNamespace

import pytest

from parsel.prompting import state
from parsel.prompting.clients import make_api_call


class _CreateRecorder:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_openai_call_uses_the_given_client_and_records_usage(
    tmp_path,
    monkeypatch,
):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer", role="assistant"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
        model="test-openai-model",
    )
    recorder = _CreateRecorder(response)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=recorder,
        )
    )
    log_path = tmp_path / "api-calls.db"
    monkeypatch.setenv("PARSEL_API_LOG", str(log_path))
    state.reset()

    result = make_api_call(
        provider="openai",
        client=client,
        model="test-openai-model",
        messages=[
            {"role": "system", "content": "real instructions"},
            {"role": "user", "content": "request"},
        ],
        temperature=0.0,
        seed=42,
    )

    assert result == ("answer", "assistant", True)
    assert recorder.kwargs["messages"][0]["content"] == "real instructions"
    assert log_path.is_file()
    assert state.n_api_calls == 1
    assert state.n_prompt_tokens == 11
    assert state.n_completion_tokens == 5


def test_anthropic_call_adapts_only_provider_specific_arguments(
    tmp_path,
    monkeypatch,
):
    response = SimpleNamespace(
        content=[SimpleNamespace(text="answer")],
        role="assistant",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        model="test-anthropic-model",
    )
    recorder = _CreateRecorder(response)
    client = SimpleNamespace(messages=recorder)
    monkeypatch.setenv("PARSEL_API_LOG", str(tmp_path / "api-calls.db"))
    state.reset()

    result = make_api_call(
        provider="anthropic",
        client=client,
        model="test-anthropic-model",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "request"},
        ],
        temperature=0.0,
        seed=42,
    )

    assert result == ("answer", "assistant", True)
    assert recorder.kwargs["messages"] == [
        {"role": "user", "content": "request"}
    ]
    assert recorder.kwargs["max_tokens"] == 1024
    assert "seed" not in recorder.kwargs


@pytest.mark.parametrize("provider", ["unknown", "", None])
def test_api_call_rejects_unknown_providers(provider):
    with pytest.raises(
        ValueError,
        match="provider must be 'openai' or 'anthropic'",
    ):
        make_api_call(
            provider=provider,
            client=object(),
            messages=[],
        )


def test_api_call_requires_an_explicit_client():
    with pytest.raises(ValueError, match="client is required"):
        make_api_call(
            provider="openai",
            client=None,
            messages=[],
        )
