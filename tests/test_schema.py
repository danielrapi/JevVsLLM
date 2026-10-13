import json

import pytest

from jevbench.schema import (
    Example,
    format_letter_prompt,
    option_letters,
    format_structured_label_prompt,
    normalize_prediction,
)


def test_correct_letter_accepts_letter_index_and_text():
    options = ["negative", "neutral", "positive"]

    assert normalize_prediction("C", options) == "C"
    assert normalize_prediction(2, options) == "C"
    assert normalize_prediction("positive", options) == "C"
    assert normalize_prediction("C. positive", options) == "C"
    assert normalize_prediction("## A", options) == "A"
    assert normalize_prediction("**B**", options) == "B"
    assert normalize_prediction("Final answer: C", options) == "C"
    assert normalize_prediction("A short explanation.\n\nB", options) == "B"


def test_normalize_rejects_unknown_prediction():
    with pytest.raises(ValueError):
        normalize_prediction("unclear", ["yes", "no"])


def test_format_letter_prompt():
    example = Example(
        id="x",
        question="Pick one.",
        options=["alpha", "beta"],
        answer="A",
    )

    prompt = format_letter_prompt(example)

    assert "A. alpha" in prompt
    assert "B. beta" in prompt
    assert "Answer with exactly one letter from A, B." in prompt
    assert "exactly one character" in prompt


def test_large_option_sets_use_numeric_codes():
    options = [f"intent_{index}" for index in range(77)]

    assert option_letters(options)[:3] == ["0", "1", "2"]
    assert option_letters(options)[76] == "76"
    assert normalize_prediction(76, options) == "76"
    assert normalize_prediction("76", options) == "76"
    assert normalize_prediction("intent_76", options) == "76"

    prompt = format_letter_prompt(
        Example(id="x", question="Pick one.", options=options, answer=76)
    )

    assert "0. intent_0" in prompt
    assert "76. intent_76" in prompt
    assert "Answer with exactly one number" in prompt


def test_format_structured_label_prompt_uses_actual_labels_not_letters():
    example = Example(
        id="x",
        question="Pick one.",
        options=["alpha", "beta"],
        answer="alpha",
    )

    prompt = format_structured_label_prompt(example)

    assert "- alpha" in prompt
    assert "- beta" in prompt
    assert '{"answer": "<label>"}' in prompt
    assert "Do not use letters like A, B, C." in prompt
    assert json.dumps(["alpha", "beta"]) in prompt
