import os
from pathlib import Path
import sqlite3
import time

from . import state


def api_log_path():
    return Path(
        os.environ.get(
            "PARSEL_API_LOG",
            "parsel_api_calls.db",
        )
    ).expanduser()


def _record_usage(model, prompt_tokens, completion_tokens):
    total_tokens = prompt_tokens + completion_tokens
    with sqlite3.connect(api_log_path()) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS api_calls "
            "(call_model TEXT, time_of_call TEXT, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, total_tokens INTEGER)"
        )
        cursor.execute(
            "INSERT INTO api_calls VALUES (?, ?, ?, ?, ?)",
            (
                model,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                prompt_tokens,
                completion_tokens,
                total_tokens,
            ),
        )
    state.record_usage(prompt_tokens, completion_tokens)


def make_api_call(provider, client, **kwargs):
    """Call one supported chat API and return text, role, and completion state."""
    if provider not in {"openai", "anthropic"}:
        raise ValueError("provider must be 'openai' or 'anthropic'")
    if client is None:
        raise ValueError("client is required")

    started = time.time()
    messages = kwargs.pop("messages")
    if messages[:1] == [{"role": "system", "content": ""}]:
        messages = messages[1:]

    if provider == "openai":
        response = client.chat.completions.create(
            messages=messages,
            **kwargs,
        )
        content = response.choices[0].message.content
        role = response.choices[0].message.role
        completed = response.choices[0].finish_reason == "stop"
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
    else:
        kwargs.pop("seed", None)
        response = client.messages.create(
            messages=messages,
            max_tokens=1024,
            **kwargs,
        )
        content = response.content[0].text
        role = response.role
        completed = response.stop_reason == "end_turn"
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

    _record_usage(
        response.model,
        prompt_tokens,
        completion_tokens,
    )
    state.add_api_time(time.time() - started)
    return content, role, completed
