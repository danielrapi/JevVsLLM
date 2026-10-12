from __future__ import annotations

import string
from typing import Any

from pydantic import BaseModel, Field, field_validator


LETTERS = string.ascii_uppercase


class Example(BaseModel):
    id: str
    question: str
    options: list[str] = Field(min_length=2, max_length=26)
    answer: str | int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def non_empty_options(cls, options: list[str]) -> list[str]:
        for option in options:
            if not option.strip():
                raise ValueError("options must not contain empty strings")
        return options

    @property
    def correct_letter(self) -> str:
        return normalize_prediction(self.answer, self.options)


class Prediction(BaseModel):
    example_id: str
    adapter: str
    correct_label: str | None = None
    prediction: str | None
    normalized_prediction: str | None
    correct: bool
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    raw_output: Any = None
    error: str | None = None


def option_letters(options: list[str]) -> list[str]:
    if len(options) > len(LETTERS):
        raise ValueError("Only up to 26 answer options are supported.")
    return list(LETTERS[: len(options)])


def normalize_prediction(value: str | int | None, options: list[str]) -> str:
    if value is None:
        raise ValueError("prediction is missing")

    letters = option_letters(options)

    if isinstance(value, int):
        if value < 0 or value >= len(options):
            raise ValueError(f"index {value} is outside the option range")
        return letters[value]

    text = str(value).strip()
    if not text:
        raise ValueError("prediction is empty")

    upper = text.upper()
    if upper in letters:
        return upper

    # Accept common model outputs like "A." or "A) positive".
    first = upper[0]
    if first in letters and (len(upper) == 1 or upper[1] in {".", ")", ":", " "}):
        return first

    for i, option in enumerate(options):
        if text.casefold() == option.strip().casefold():
            return letters[i]

    raise ValueError(f"could not normalize prediction {value!r}")


def format_letter_prompt(example: Example) -> str:
    letters = option_letters(example.options)
    rendered_options = "\n".join(
        f"{letter}. {option}" for letter, option in zip(letters, example.options, strict=True)
    )
    allowed = ", ".join(letters)
    return (
        f"{example.question.strip()}\n\n"
        f"Options:\n{rendered_options}\n\n"
        f"Answer with exactly one letter from {allowed}."
    )
