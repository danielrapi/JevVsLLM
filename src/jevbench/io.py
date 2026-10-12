from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Example, Prediction


def read_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                examples.append(Example.model_validate_json(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid example at {path}:{line_number}: {exc}") from exc
    return examples


def write_predictions(path: Path, predictions: Iterable[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(prediction.model_dump_json() + "\n")


def read_predictions(path: Path) -> list[Prediction]:
    rows: list[Prediction] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(Prediction.model_validate_json(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid prediction at {path}:{line_number}: {exc}") from exc
    return rows


def read_example_map(path: Path) -> dict[str, Example]:
    return {example.id: example for example in read_examples(path)}


def read_jsonl_dicts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows
