from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .io import read_jsonl_dicts
from .schema import (
    Example,
    Prediction,
    format_letter_prompt,
    format_structured_label_prompt,
    normalize_prediction,
    option_letters,
)


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class Adapter(ABC):
    name: str

    @abstractmethod
    def predict(self, example: Example) -> Prediction:
        raise NotImplementedError

    def _prediction(
        self,
        example: Example,
        raw_prediction: str | int | None,
        latency_ms: float,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        raw_output: Any = None,
        error: str | None = None,
    ) -> Prediction:
        normalized: str | None = None
        correct = False
        final_error = error
        try:
            normalized = normalize_prediction(raw_prediction, example.options)
            correct = normalized == example.correct_letter
        except Exception as exc:
            final_error = final_error or str(exc)

        return Prediction(
            example_id=example.id,
            adapter=self.name,
            correct_label=example.correct_letter,
            prediction=None if raw_prediction is None else str(raw_prediction),
            normalized_prediction=normalized,
            correct=correct,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            raw_output=raw_output,
            error=final_error,
        )


class MajorityAdapter(Adapter):
    """Tiny offline smoke-test baseline: always pick A."""

    name = "majority"

    def predict(self, example: Example) -> Prediction:
        start = time.perf_counter()
        return self._prediction(example, "A", (time.perf_counter() - start) * 1000)


class SentimentLexiconAdapter(Adapter):
    """Small deterministic sentiment baseline for smoke-testing sentiment datasets."""

    name = "sentiment-lexicon"

    positive_words = {
        "absorbing",
        "amazing",
        "best",
        "brilliant",
        "charming",
        "clever",
        "compelling",
        "delight",
        "delightful",
        "effective",
        "enjoy",
        "excellent",
        "fascinating",
        "fun",
        "good",
        "great",
        "heart",
        "honest",
        "impressive",
        "love",
        "loved",
        "masterpiece",
        "moving",
        "original",
        "perfect",
        "refreshing",
        "remarkable",
        "rewarding",
        "smart",
        "strong",
        "superb",
        "triumph",
        "wonderful",
        "worth",
    }
    negative_words = {
        "awful",
        "bad",
        "boring",
        "crushed",
        "disappointing",
        "dull",
        "failed",
        "fails",
        "flat",
        "forget",
        "forgettable",
        "hate",
        "hated",
        "mess",
        "poor",
        "problem",
        "sad",
        "static",
        "terrible",
        "tepid",
        "thin",
        "tired",
        "unfunny",
        "weak",
        "worst",
    }

    def predict(self, example: Example) -> Prediction:
        start = time.perf_counter()
        text = example.question.casefold()
        positive = sum(1 for word in self.positive_words if word in text)
        negative = sum(1 for word in self.negative_words if word in text)
        raw_prediction = "B" if positive >= negative else "A"
        return self._prediction(example, raw_prediction, (time.perf_counter() - start) * 1000)


class JevAPIAdapter(Adapter):
    name = "jev-api"

    def __init__(
        self,
        model: str = "jev-latest",
        api_key: str | None = None,
        base_url: str = "https://api.typesafe.ai/v1/systemone",
        input_price_per_1m: float = 0.042,
        timeout_seconds: float = 60,
        api_key_env_names: tuple[str, ...] = (
            "TYPESAFE_API_KEY",
            "JEV_API_KEY",
            "TYPESAFE_JEV_API_KEY",
        ),
        api_key_label: str = "Jev API",
    ):
        self.model = model
        self.api_key = api_key or _first_env(api_key_env_names)
        if not self.api_key:
            env_list = ", ".join(api_key_env_names)
            raise RuntimeError(f"{api_key_label} requires one of: {env_list}.")
        self.base_url = base_url
        self.input_price_per_1m = input_price_per_1m
        self.timeout_seconds = timeout_seconds

    def predict(self, example: Example) -> Prediction:
        letters = option_letters(example.options)
        payload = {
            "model": self.model,
            "state": {"question": example.question},
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": (
                        "Choose the single best answer for the classification question in "
                        "`question`."
                    ),
                    "criteria": {
                        letter: option
                        for letter, option in zip(letters, example.options, strict=True)
                    },
                }
            },
        }

        start = time.perf_counter()
        try:
            response_payload = self._post(payload)
            latency_ms = (time.perf_counter() - start) * 1000
            answer = response_payload["answers"]["answer"]
            raw_prediction = answer["choice"]
            usage = response_payload.get("usage") or {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            cost = (
                (input_tokens * self.input_price_per_1m) / 1_000_000
                if input_tokens is not None
                else None
            )
            return self._prediction(
                example,
                raw_prediction,
                latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                raw_output={
                    "model": response_payload.get("model"),
                    "choice_answer": answer,
                },
            )
        except Exception as exc:
            return self._prediction(
                example,
                None,
                (time.perf_counter() - start) * 1000,
                error=str(exc),
            )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Jev API HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Jev API request failed: {exc.reason}") from exc


class JevOpenRouterAdapter(JevAPIAdapter):
    name = "jev-openrouter"

    def __init__(
        self,
        model: str = "~typesafe/jev-latest",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/alpha/decisions",
        input_price_per_1m: float = 0.042,
        timeout_seconds: float = 60,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            input_price_per_1m=input_price_per_1m,
            timeout_seconds=timeout_seconds,
            api_key_env_names=("OPENROUTER_API_KEY",),
            api_key_label="OpenRouter Jev API",
        )

    def predict(self, example: Example) -> Prediction:
        letters = option_letters(example.options)
        payload = {
            "model": self.model,
            "state": example.question,
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": "Choose the single best answer.",
                    "criteria": {
                        letter: option
                        for letter, option in zip(letters, example.options, strict=True)
                    },
                }
            },
        }

        start = time.perf_counter()
        try:
            response_payload = self._post(payload)
            latency_ms = (time.perf_counter() - start) * 1000
            answer = response_payload["answers"]["answer"]
            raw_prediction = answer["choice"]
            usage = response_payload.get("usage") or {}
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
            cost = (
                _estimate_cost(input_tokens, output_tokens, self.input_price_per_1m, 0.0)
                if input_tokens is not None
                else usage.get("cost")
            )
            return self._prediction(
                example,
                raw_prediction,
                latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                raw_output={
                    "model": response_payload.get("model", self.model),
                    "choice_answer": answer,
                    "response": _safe_dump(response_payload),
                },
            )
        except Exception as exc:
            return self._prediction(
                example,
                None,
                (time.perf_counter() - start) * 1000,
                error=str(exc),
            )


class PredictionsFileAdapter(Adapter):
    name = "predictions-file"

    def __init__(self, predictions_path: Path):
        rows = read_jsonl_dicts(predictions_path)
        self.predictions_by_id: dict[str, Any] = {}
        for row in rows:
            example_id = row.get("id") or row.get("example_id")
            if not example_id:
                raise ValueError(f"Prediction row is missing id/example_id: {row}")
            value = row.get("prediction", row.get("answer", row.get("label")))
            self.predictions_by_id[str(example_id)] = value

    def predict(self, example: Example) -> Prediction:
        start = time.perf_counter()
        raw_prediction = self.predictions_by_id.get(example.id)
        return self._prediction(example, raw_prediction, (time.perf_counter() - start) * 1000)


class CommandAdapter(Adapter):
    name = "command"

    def __init__(self, command: str, timeout_seconds: float = 60):
        self.command = command
        self.timeout_seconds = timeout_seconds

    def predict(self, example: Example) -> Prediction:
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                shlex.split(self.command),
                input=example.model_dump_json(),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            if completed.returncode != 0:
                return self._prediction(
                    example,
                    None,
                    latency_ms,
                    raw_output={"stderr": completed.stderr, "stdout": completed.stdout},
                    error=f"command exited with status {completed.returncode}",
                )
            payload = json.loads(completed.stdout)
            raw_prediction = payload.get("prediction", payload.get("answer", payload.get("label")))
            return self._prediction(example, raw_prediction, latency_ms, raw_output=payload)
        except Exception as exc:
            return self._prediction(
                example,
                None,
                (time.perf_counter() - start) * 1000,
                error=str(exc),
            )


class OpenAILetterAdapter(Adapter):
    name = "openai-letter"

    def __init__(
        self,
        model: str,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        max_output_tokens: int = 256,
        reasoning_effort: str = "none",
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI adapter requires the optional dependency: pip install -e '.[openai]'"
            ) from exc

        self.client = OpenAI()
        self.model = model
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def predict(self, example: Example) -> Prediction:
        prompt = format_letter_prompt(example)
        start = time.perf_counter()
        try:
            response = self.client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                instructions=(
                    "You are evaluating a multiple-choice classification task. "
                    "Return exactly one answer code and no other text."
                ),
                input=prompt,
                max_output_tokens=self.max_output_tokens,
                text={"format": {"type": "text"}},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            raw_text = response.output_text.strip()
            input_tokens, output_tokens = _extract_response_tokens(response)
            cost = _estimate_cost(
                input_tokens,
                output_tokens,
                self.input_price_per_1m,
                self.output_price_per_1m,
            )
            return self._prediction(
                example,
                raw_text,
                latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                raw_output={
                    "response_id": response.id,
                    "model": response.model,
                    "status": getattr(response, "status", None),
                    "incomplete_details": _safe_dump(getattr(response, "incomplete_details", None)),
                },
            )
        except Exception as exc:
            return self._prediction(
                example,
                None,
                (time.perf_counter() - start) * 1000,
                error=str(exc),
            )


class OpenRouterLetterAdapter(Adapter):
    name = "openrouter-letter"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_output_tokens: int = 8,
        reasoning_enabled: bool | None = False,
        timeout_seconds: float = 60,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OpenRouter requires OPENROUTER_API_KEY.")
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m
        self.base_url = base_url
        self.max_output_tokens = max_output_tokens
        self.reasoning_enabled = reasoning_enabled
        self.timeout_seconds = timeout_seconds

    def predict(self, example: Example) -> Prediction:
        prompt = format_letter_prompt(example)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are evaluating a multiple-choice classification task. "
                        "Return exactly one answer code and no other text. "
                        "For large option sets the code may be a number."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
        }
        if self.reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": self.reasoning_enabled}

        start = time.perf_counter()
        try:
            response_payload = self._post(payload)
            latency_ms = (time.perf_counter() - start) * 1000
            choice = response_payload.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            raw_text = content.strip() if isinstance(content, str) else ""
            usage = response_payload.get("usage") or {}
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            cost = usage.get("cost")
            if cost is None or cost == 0:
                cost = _estimate_cost(
                    input_tokens,
                    output_tokens,
                    self.input_price_per_1m,
                    self.output_price_per_1m,
                )
            return self._prediction(
                example,
                raw_text,
                latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                raw_output={
                    "response_id": response_payload.get("id"),
                    "model": response_payload.get("model", self.model),
                    "provider": response_payload.get("provider"),
                    "finish_reason": choice.get("finish_reason"),
                    "message": _safe_dump(message),
                },
            )
        except Exception as exc:
            return self._prediction(
                example,
                None,
                (time.perf_counter() - start) * 1000,
                error=str(exc),
            )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(3):
            request = Request(
                self.base_url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "HTTP-Referer": "https://codex.local/jevbench",
                    "X-Title": "Jev vs Good Models Benchmark",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenRouter API HTTP {exc.code}: {body}") from exc
            except URLError as exc:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenRouter API request failed: {exc.reason}") from exc
        raise RuntimeError("OpenRouter API request failed after retries")


class OpenRouterStructuredLabelAdapter(OpenRouterLetterAdapter):
    name = "openrouter-structured-label"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_output_tokens: int = 80,
        reasoning_enabled: bool | None = False,
        timeout_seconds: float = 60,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            reasoning_enabled=reasoning_enabled,
            timeout_seconds=timeout_seconds,
        )

    def predict(self, example: Example) -> Prediction:
        prompt = format_structured_label_prompt(example)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are evaluating a classification task. Return only valid JSON "
                        'with this exact shape: {"answer": "<label>"}. The answer must be '
                        "one of the provided label strings."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
        }
        if self.reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": self.reasoning_enabled}

        start = time.perf_counter()
        try:
            response_payload = self._post(payload)
            latency_ms = (time.perf_counter() - start) * 1000
            choice = response_payload.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            raw_text = content.strip() if isinstance(content, str) else ""
            usage = response_payload.get("usage") or {}
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            cost = usage.get("cost")
            if cost is None or cost == 0:
                cost = _estimate_cost(
                    input_tokens,
                    output_tokens,
                    self.input_price_per_1m,
                    self.output_price_per_1m,
                )

            parse_error = None
            try:
                raw_label = _extract_structured_answer(raw_text)
            except Exception as exc:
                raw_label = raw_text
                parse_error = str(exc)

            return self._prediction(
                example,
                raw_label,
                latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                raw_output={
                    "response_id": response_payload.get("id"),
                    "model": response_payload.get("model", self.model),
                    "provider": response_payload.get("provider"),
                    "finish_reason": choice.get("finish_reason"),
                    "message": _safe_dump(message),
                    "structured_text": raw_text,
                },
                error=parse_error,
            )
        except Exception as exc:
            return self._prediction(
                example,
                None,
                (time.perf_counter() - start) * 1000,
                error=str(exc),
            )


def _extract_response_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return input_tokens, output_tokens


def _extract_structured_answer(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("structured output is not a JSON object")
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("structured output is missing a string `answer` field")
    return answer.strip()


def _safe_dump(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
) -> float | None:
    if (
        input_tokens is None
        or output_tokens is None
        or input_price_per_1m is None
        or output_price_per_1m is None
    ):
        return None
    return (input_tokens * input_price_per_1m + output_tokens * output_price_per_1m) / 1_000_000
