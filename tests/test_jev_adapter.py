import pytest

from jevbench.adapters import JevAPIAdapter
from jevbench.schema import Example


def test_jev_payload_and_response_parsing(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    adapter = JevAPIAdapter()

    def fake_post(payload):
        assert payload["model"] == "jev-latest"
        assert payload["questions"]["answer"]["type"] == "choice"
        assert payload["questions"]["answer"]["criteria"] == {
            "A": "negative",
            "B": "positive",
        }
        return {
            "model": "jev-1.13.0",
            "answers": {
                "answer": {
                    "type": "choice",
                    "choice": "B",
                    "probabilities": {"A": 0.1, "B": 0.9},
                    "confidence": 0.8,
                }
            },
            "usage": {"input_tokens": 100, "output_tokens": 0},
        }

    adapter._post = fake_post
    result = adapter.predict(
        Example(
            id="x",
            question="Classify the sentiment: excellent movie.",
            options=["negative", "positive"],
            answer="B",
        )
    )

    assert result.correct is True
    assert result.normalized_prediction == "B"
    assert result.input_tokens == 100
    assert result.estimated_cost_usd == pytest.approx(0.0000042)
