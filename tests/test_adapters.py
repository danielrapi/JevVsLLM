import pytest

from jevbench.adapters import _extract_structured_answer


def test_extract_structured_answer_accepts_plain_json():
    assert _extract_structured_answer('{"answer": "positive"}') == "positive"


def test_extract_structured_answer_accepts_fenced_json():
    assert _extract_structured_answer('```json\n{"answer": "World"}\n```') == "World"


def test_extract_structured_answer_rejects_missing_answer():
    with pytest.raises(ValueError):
        _extract_structured_answer('{"label": "positive"}')
