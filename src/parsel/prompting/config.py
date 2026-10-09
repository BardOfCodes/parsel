import os


PROVIDER = os.environ.get("PARSEL_LLM_PROVIDER", "openai")
KEY = os.environ.get("PARSEL_LLM_API_KEY", "")
MODEL = os.environ.get("PARSEL_LLM_MODEL", "gpt-4o")
TEMPERATURE = float(os.environ.get("PARSEL_LLM_TEMPERATURE", "0"))
SEED = int(os.environ.get("PARSEL_LLM_SEED", "42"))
