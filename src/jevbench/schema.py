from __future__ import annotations

import string
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


LETTERS = string.ascii_uppercase


class Example(BaseModel):
    id: str
    question: str
    options: list[str] = Field(min_length=2)
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


    @property
    def correct_code(self) -> str:
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
        return [str(index) for index in range(len(options))]
    return list(LETTERS[: len(options)])


def normalize_prediction(value: str | int | None, options: list[str]) -> str:
    if value is None:
        raise ValueError("prediction is missing")

    letters = option_letters(options)
    code_set = set(letters)

    if isinstance(value, int):
        if value < 0 or value >= len(options):
            raise ValueError(f"index {value} is outside the option range")
        return letters[value]

    text = str(value).strip()
    if not text:
        raise ValueError("prediction is empty")

    upper = text.upper()
    if text in code_set:
        return text
    if upper in code_set:
        return upper

    if _uses_numeric_codes(options):
        number_match = re.search(r"(?:^|\b)(\d{1,3})(?:\b|$)", text)
        if number_match and number_match.group(1) in code_set:
            return number_match.group(1)

    trailing_letter = re.search(r"(?:^|\n)\s*([A-Z])\s*$", upper)
    if trailing_letter and trailing_letter.group(1) in code_set:
        return trailing_letter.group(1)

    # Accept common model outputs like "A." or "A) positive".
    first = upper[0]
    if first in code_set and (len(upper) == 1 or upper[1] in {".", ")", ":", " "}):
        return first

    # Accept markdown-wrapped answers like "## A" or "**B**".
    leading_letter = re.match(r"^[^A-Z0-9]*([A-Z])(?:[^A-Z0-9]|$)", upper)
    if leading_letter and leading_letter.group(1) in code_set:
        return leading_letter.group(1)

    # Accept explicit answer phrases while avoiding arbitrary letters in explanations.
    answer_phrase = re.search(
        r"\b(?:ANSWER|FINAL ANSWER|CORRECT ANSWER)\s*(?:IS|:|-)?\s*([A-Z])\b",
        upper,
    )
    if answer_phrase and answer_phrase.group(1) in code_set:
        return answer_phrase.group(1)

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
    code_kind = "number" if _uses_numeric_codes(example.options) else "letter"
    exact_length = "one number" if code_kind == "number" else "one character"
    return (
        f"{example.question.strip()}\n\n"
        f"Options:\n{rendered_options}\n\n"
        f"Answer with exactly one {code_kind} from {allowed}.\n"
        f"Your entire response must be exactly {exact_length}. "
        "Do not include markdown, punctuation, explanation, or the option text."
    )


def format_structured_label_prompt(example: Example) -> str:
    rendered_options = "\n".join(f"- {option}" for option in example.options)
    labels_json = json_array(example.options)
    return (
        f"{example.question.strip()}\n\n"
        f"Valid labels:\n{rendered_options}\n\n"
        'Return JSON with exactly this shape: {"answer": "<label>"}.\n'
        f"The answer value must be exactly one of: {labels_json}.\n"
        "Do not use letters like A, B, C. Do not include explanation or markdown."
    )


def json_array(values: list[str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False)


def _uses_numeric_codes(options: list[str]) -> bool:
    return len(options) > len(LETTERS)
