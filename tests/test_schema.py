import pytest

from jevbench.schema import Example, format_letter_prompt, normalize_prediction


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
