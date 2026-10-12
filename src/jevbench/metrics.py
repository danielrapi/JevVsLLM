from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from .schema import Prediction


@dataclass(frozen=True)
class ClassStats:
    label: str
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class Summary:
    adapter: str
    n: int
    accuracy: float
    errors: int
    invalid: int
    error_rate: float
    invalid_rate: float
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    weighted_precision: float | None
    weighted_recall: float | None
    weighted_f1: float | None
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_cost_usd: float | None
    mean_cost_usd: float | None
    cost_per_1k_usd: float | None
    cost_per_correct_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    per_class: list[ClassStats] = field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    brier_score: float | None = None
    ece: float | None = None
    mean_confidence: float | None = None


def summarize(predictions: list[Prediction]) -> Summary:
    if not predictions:
        raise ValueError("cannot summarize an empty prediction list")

    costs = [p.estimated_cost_usd for p in predictions if p.estimated_cost_usd is not None]
    input_tokens = [p.input_tokens for p in predictions if p.input_tokens is not None]
    output_tokens = [p.output_tokens for p in predictions if p.output_tokens is not None]
    latencies = [p.latency_ms for p in predictions]
    correct_count = sum(p.correct for p in predictions)
    errors = sum(1 for p in predictions if p.error)
    invalid = sum(1 for p in predictions if p.normalized_prediction is None)
    total_cost = sum(costs) if costs else None

    class_metrics = _classification_metrics(predictions)
    calibration = _calibration_metrics(predictions)

    return Summary(
        adapter=predictions[0].adapter,
        n=len(predictions),
        accuracy=correct_count / len(predictions),
        errors=errors,
        invalid=invalid,
        error_rate=errors / len(predictions),
        invalid_rate=invalid / len(predictions),
        macro_precision=class_metrics["macro_precision"],
        macro_recall=class_metrics["macro_recall"],
        macro_f1=class_metrics["macro_f1"],
        weighted_precision=class_metrics["weighted_precision"],
        weighted_recall=class_metrics["weighted_recall"],
        weighted_f1=class_metrics["weighted_f1"],
        mean_latency_ms=mean(latencies),
        median_latency_ms=median(latencies),
        p90_latency_ms=_percentile(latencies, 90),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        total_cost_usd=total_cost,
        mean_cost_usd=mean(costs) if costs else None,
        cost_per_1k_usd=(total_cost / len(predictions) * 1000) if total_cost is not None else None,
        cost_per_correct_usd=(total_cost / correct_count)
        if total_cost is not None and correct_count
        else None,
        input_tokens=sum(input_tokens) if input_tokens else None,
        output_tokens=sum(output_tokens) if output_tokens else None,
        mean_input_tokens=mean(input_tokens) if input_tokens else None,
        mean_output_tokens=mean(output_tokens) if output_tokens else None,
        per_class=class_metrics["per_class"],
        confusion_matrix=class_metrics["confusion_matrix"],
        brier_score=calibration["brier_score"],
        ece=calibration["ece"],
        mean_confidence=calibration["mean_confidence"],
    )


def _classification_metrics(predictions: list[Prediction]) -> dict[str, Any]:
    labelled = [
        p
        for p in predictions
        if p.correct_label is not None and p.normalized_prediction is not None
    ]
    if not labelled:
        return {
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
            "per_class": [],
            "confusion_matrix": {},
        }

    labels = sorted(
        {
            label
            for prediction in labelled
            for label in (prediction.correct_label, prediction.normalized_prediction)
            if label is not None
        }
    )
    confusion: dict[str, dict[str, int]] = {
        gold: {predicted: 0 for predicted in labels} for gold in labels
    }
    for prediction in labelled:
        confusion[prediction.correct_label][prediction.normalized_prediction] += 1

    per_class: list[ClassStats] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        support = sum(confusion[label].values())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            ClassStats(
                label=label,
                support=support,
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    total_support = sum(stats.support for stats in per_class)
    return {
        "macro_precision": mean(stats.precision for stats in per_class),
        "macro_recall": mean(stats.recall for stats in per_class),
        "macro_f1": mean(stats.f1 for stats in per_class),
        "weighted_precision": _weighted_average(
            [(stats.precision, stats.support) for stats in per_class], total_support
        ),
        "weighted_recall": _weighted_average(
            [(stats.recall, stats.support) for stats in per_class], total_support
        ),
        "weighted_f1": _weighted_average(
            [(stats.f1, stats.support) for stats in per_class], total_support
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _calibration_metrics(predictions: list[Prediction], buckets: int = 10) -> dict[str, float | None]:
    rows: list[tuple[float, bool, str | None]] = []
    brier_rows: list[tuple[dict[str, float], str]] = []

    for prediction in predictions:
        probabilities = _extract_probabilities(prediction)
        if not probabilities:
            continue
        confidence = max(probabilities.values())
        rows.append((confidence, prediction.correct, prediction.correct_label))
        if prediction.correct_label is not None:
            brier_rows.append((probabilities, prediction.correct_label))

    if not rows:
        return {"brier_score": None, "ece": None, "mean_confidence": None}

    ece = 0.0
    for bucket_index in range(buckets):
        low = bucket_index / buckets
        high = (bucket_index + 1) / buckets
        if bucket_index == buckets - 1:
            bucket = [row for row in rows if low <= row[0] <= high]
        else:
            bucket = [row for row in rows if low <= row[0] < high]
        if not bucket:
            continue
        bucket_confidence = mean(row[0] for row in bucket)
        bucket_accuracy = mean(row[1] for row in bucket)
        ece += (len(bucket) / len(rows)) * abs(bucket_accuracy - bucket_confidence)

    brier_score = None
    if brier_rows:
        labels = sorted({label for probabilities, gold in brier_rows for label in {*probabilities, gold}})
        brier_score = mean(
            sum((probabilities.get(label, 0.0) - (1.0 if label == gold else 0.0)) ** 2 for label in labels)
            for probabilities, gold in brier_rows
        )

    return {
        "brier_score": brier_score,
        "ece": ece,
        "mean_confidence": mean(row[0] for row in rows),
    }


def _extract_probabilities(prediction: Prediction) -> dict[str, float] | None:
    raw = prediction.raw_output
    if not isinstance(raw, dict):
        return None
    answer = raw.get("choice_answer")
    if not isinstance(answer, dict):
        return None
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        return None
    clean: dict[str, float] = {}
    for key, value in probabilities.items():
        try:
            clean[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return clean or None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile / 100)
    return sorted_values[index]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _weighted_average(values: list[tuple[float, int]], total_weight: int) -> float:
    if not total_weight:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight

