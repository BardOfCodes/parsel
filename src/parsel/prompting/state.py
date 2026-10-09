"""Process-local counters for prompt API usage."""

n_api_calls = 0
n_prompt_tokens = 0
n_completion_tokens = 0
api_time = 0.0
error_count = 0


def record_usage(prompt_tokens, completion_tokens, calls=1):
    global n_api_calls, n_prompt_tokens, n_completion_tokens
    n_api_calls += calls
    n_prompt_tokens += prompt_tokens
    n_completion_tokens += completion_tokens


def add_api_time(seconds):
    global api_time
    api_time += seconds


def record_error():
    global error_count
    error_count += 1


def reset():
    global n_api_calls, n_prompt_tokens, n_completion_tokens
    global api_time, error_count
    n_api_calls = 0
    n_prompt_tokens = 0
    n_completion_tokens = 0
    api_time = 0.0
    error_count = 0
