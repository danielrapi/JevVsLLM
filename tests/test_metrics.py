from jevbench.metrics import summarize
from jevbench.schema import Prediction


def test_summarize_predictions():
    predictions = [
        Prediction(
            example_id="1",
            adapter="x",
            correct_label="A",
            prediction="A",
            normalized_prediction="A",
            correct=True,
            latency_ms=10,
            input_tokens=100,
            output_tokens=1,
            estimated_cost_usd=0.001,
        ),
        Prediction(
            example_id="2",
            adapter="x",
            correct_label="A",
            prediction="B",
            normalized_prediction="B",
            correct=False,
            latency_ms=20,
            input_tokens=200,
            output_tokens=1,
            estimated_cost_usd=0.002,
        ),
    ]

    summary = summarize(predictions)

    assert summary.accuracy == 0.5
    assert summary.macro_f1 == 1 / 3
    assert summary.weighted_f1 == 2 / 3
    assert summary.confusion_matrix == {"A": {"A": 1, "B": 1}, "B": {"A": 0, "B": 0}}
    assert summary.total_cost_usd == 0.003
    assert summary.cost_per_1k_usd == 1.5
    assert summary.cost_per_correct_usd == 0.003
    assert summary.input_tokens == 300
    assert summary.output_tokens == 2


def test_summarize_calibration_from_jev_probabilities():
    predictions = [
        Prediction(
            example_id="1",
            adapter="jev-api",
            correct_label="A",
            prediction="A",
            normalized_prediction="A",
            correct=True,
            latency_ms=1,
            raw_output={
                "choice_answer": {
                    "choice": "A",
                    "probabilities": {"A": 0.8, "B": 0.2},
                }
            },
        ),
        Prediction(
            example_id="2",
            adapter="jev-api",
            correct_label="B",
            prediction="A",
            normalized_prediction="A",
            correct=False,
            latency_ms=1,
            raw_output={
                "choice_answer": {
                    "choice": "A",
                    "probabilities": {"A": 0.7, "B": 0.3},
                }
            },
        ),
    ]

    summary = summarize(predictions)

    assert summary.mean_confidence == 0.75
    assert summary.ece is not None
    assert summary.brier_score is not None
