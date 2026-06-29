from enum import StrEnum


class Model(StrEnum):
    GPT4O = "openai/gpt-4o-mini"
    OPUS = "~anthropic/claude-opus-latest"
    SONNET = "anthropic/claude-sonnet-latest"
    HAIKU = "~anthropic/claude-haiku-latest"
    GEMINI_FLASH = "~google/gemini-flash-latest"
    GEMINI_PRO = "~google/gemini-pro-latest"
