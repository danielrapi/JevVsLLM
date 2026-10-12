from __future__ import annotations

import json
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import typer

from .adapters import (
    CommandAdapter,
    JevAPIAdapter,
    JevOpenRouterAdapter,
    MajorityAdapter,
    OpenAILetterAdapter,
    OpenRouterLetterAdapter,
    PredictionsFileAdapter,
    SentimentLexiconAdapter,
)
from .datasets import (
    download_20newsgroups,
    download_ag_news,
    download_arc_challenge,
    download_boolq,
    download_sst2,
    download_starter_suite,
    download_trec,
)
from .env import load_dotenv
from .experiments import read_manifest, run_file_name, safe_name
from .io import read_example_map, read_examples, read_predictions, write_predictions
from .metrics import Summary, summarize

app = typer.Typer(help="Benchmark language classification adapters.")
datasets_app = typer.Typer(help="Download and convert benchmark datasets.")
app.add_typer(datasets_app, name="datasets")


@app.command()
def run(
    dataset: Annotated[Path, typer.Option(help="JSONL dataset path.")],
    adapter: Annotated[
        str,
        typer.Option(
            help=(
                "Adapter: majority, sentiment-lexicon, jev-api, jev-openrouter, "
                "openai-letter, openrouter-letter, predictions-file, or command."
            )
        ),
    ],
    out: Annotated[Path, typer.Option(help="Where to write JSONL predictions.")],
    model: Annotated[str, typer.Option(help="OpenAI model for openai-letter.")] = "gpt-5-mini",
    input_price_per_1m: Annotated[
        float | None,
        typer.Option(help="Input token price in USD per 1M tokens."),
    ] = None,
    output_price_per_1m: Annotated[
        float | None,
        typer.Option(help="Output token price in USD per 1M tokens."),
    ] = None,
    reasoning_effort: Annotated[
        str,
        typer.Option(help="OpenAI reasoning effort for openai-letter."),
    ] = "none",
    jev_base_url: Annotated[
        str,
        typer.Option(help="Jev API endpoint for jev-api or jev-openrouter."),
    ] = "https://api.typesafe.ai/v1/systemone",
    predictions: Annotated[
        Path | None,
        typer.Option(help="Predictions JSONL path for predictions-file adapter."),
    ] = None,
    command: Annotated[
        str | None,
        typer.Option(help="Command to run for the command adapter."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(help="Number of concurrent examples to run for hosted adapters."),
    ] = 1,
) -> None:
    load_dotenv()
    examples = read_examples(dataset)
    selected = _build_adapter(
        adapter,
        model=model,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        reasoning_effort=reasoning_effort,
        jev_base_url=jev_base_url,
        predictions=predictions,
        command=command,
    )

    results = _predict_all(selected, examples, concurrency=concurrency)
    write_predictions(out, results)
    _print_summary(summarize(results))
    typer.echo(f"Wrote predictions to {out}")


@app.command("run-manifest")
def run_manifest(
    manifest: Annotated[Path, typer.Option(help="Experiment manifest JSON.")],
    dataset: Annotated[Path, typer.Option(help="Dataset JSONL path.")],
    out_dir: Annotated[Path, typer.Option(help="Directory for per-model prediction files.")],
    summaries_out: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file for combined summary metrics."),
    ] = None,
    limit_models: Annotated[
        int | None,
        typer.Option(help="Optional maximum number of models from the manifest to run."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(help="Number of concurrent examples per model."),
    ] = 1,
) -> None:
    load_dotenv()
    config = read_manifest(manifest)
    examples = read_examples(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    models = config["models"][:limit_models] if limit_models is not None else config["models"]
    for model_config in models:
        label = model_config["label"]
        adapter_name = model_config["adapter"]
        model_name = model_config["model"]
        typer.echo(f"Running {label} ({model_name})...")
        selected = _build_adapter(
            adapter_name,
            model=model_name,
            input_price_per_1m=model_config.get("price_per_1m_input_usd"),
            output_price_per_1m=model_config.get("price_per_1m_output_usd"),
            reasoning_effort="none",
            jev_base_url="https://api.typesafe.ai/v1/systemone",
            predictions=None,
            command=None,
        )
        results = _predict_all(selected, examples, concurrency=concurrency)
        run_path = out_dir / run_file_name(label)
        write_predictions(run_path, results)
        summary = summarize(results)
        summaries.append(
            {
                "label": label,
                "adapter": adapter_name,
                "model": model_name,
                "run_path": str(run_path),
                "summary": asdict(summary),
            }
        )
        _print_summary(summary)

    _print_comparison([_summary_from_dict(row["summary"]) for row in summaries])
    if summaries_out is not None:
        summaries_out.parent.mkdir(parents=True, exist_ok=True)
        summaries_out.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote combined summary to {summaries_out}")


@app.command("run-suite")
def run_suite(
    models_manifest: Annotated[Path, typer.Option(help="Model manifest JSON.")],
    datasets_manifest: Annotated[Path, typer.Option(help="Dataset suite manifest JSON.")],
    out_dir: Annotated[Path, typer.Option(help="Directory for prediction files.")],
    summaries_out: Annotated[
        Path | None,
        typer.Option(help="Optional JSON file for combined summary metrics."),
    ] = None,
    limit_models: Annotated[
        int | None,
        typer.Option(help="Optional maximum number of models to run."),
    ] = None,
    limit_datasets: Annotated[
        int | None,
        typer.Option(help="Optional maximum number of datasets to run."),
    ] = None,
    concurrency: Annotated[int, typer.Option(help="Concurrent examples per model.")] = 4,
) -> None:
    load_dotenv()
    model_config = read_manifest(models_manifest)
    dataset_config = read_manifest(datasets_manifest)
    models = (
        model_config["models"][:limit_models]
        if limit_models is not None
        else model_config["models"]
    )
    datasets = (
        dataset_config["models"][:limit_datasets]
        if limit_datasets is not None
        else dataset_config["models"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: list[dict] = []
    for dataset_entry in datasets:
        dataset_label = dataset_entry["label"]
        dataset_path = Path(dataset_entry["path"])
        typer.echo(f"Dataset {dataset_label}: {dataset_path}")
        examples = read_examples(dataset_path)
        dataset_out_dir = out_dir / safe_name(dataset_label)
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        for model_entry in models:
            label = model_entry["label"]
            typer.echo(f"Running {dataset_label} / {label} ({model_entry['model']})...")
            selected = _build_adapter(
                model_entry["adapter"],
                model=model_entry["model"],
                input_price_per_1m=model_entry.get("price_per_1m_input_usd"),
                output_price_per_1m=model_entry.get("price_per_1m_output_usd"),
                reasoning_effort="none",
                jev_base_url="https://api.typesafe.ai/v1/systemone",
                predictions=None,
                command=None,
            )
            results = _predict_all(selected, examples, concurrency=concurrency)
            run_path = dataset_out_dir / run_file_name(label)
            write_predictions(run_path, results)
            summary = summarize(results)
            combined.append(
                {
                    "dataset": dataset_label,
                    "dataset_path": str(dataset_path),
                    "label": label,
                    "adapter": model_entry["adapter"],
                    "model": model_entry["model"],
                    "run_path": str(run_path),
                    "summary": asdict(summary),
                }
            )
            _print_summary(summary)

    if summaries_out is not None:
        summaries_out.parent.mkdir(parents=True, exist_ok=True)
        summaries_out.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote combined suite summary to {summaries_out}")


@app.command("retry-invalid")
def retry_invalid(
    run: Annotated[Path, typer.Option(help="Existing prediction JSONL to repair.")],
    dataset: Annotated[Path, typer.Option(help="Dataset JSONL used for the run.")],
    adapter: Annotated[str, typer.Option(help="Adapter to rerun.")],
    model: Annotated[str, typer.Option(help="Model to rerun.")],
    out: Annotated[Path, typer.Option(help="Output JSONL with repaired predictions.")],
    input_price_per_1m: Annotated[
        float | None,
        typer.Option(help="Input token price in USD per 1M tokens."),
    ] = None,
    output_price_per_1m: Annotated[
        float | None,
        typer.Option(help="Output token price in USD per 1M tokens."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(help="Number of concurrent invalid examples to rerun."),
    ] = 4,
) -> None:
    load_dotenv()
    predictions = read_predictions(run)
    examples_by_id = read_example_map(dataset)
    selected = _build_adapter(
        adapter,
        model=model,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        reasoning_effort="none",
        jev_base_url="https://api.typesafe.ai/v1/systemone",
        predictions=None,
        command=None,
    )

    invalid_indices = [
        index
        for index, prediction in enumerate(predictions)
        if prediction.error or prediction.normalized_prediction is None
    ]
    examples_to_retry = [examples_by_id[predictions[index].example_id] for index in invalid_indices]
    typer.echo(f"Retrying {len(examples_to_retry)} invalid/error rows from {run}")
    retried = _predict_all(selected, examples_to_retry, concurrency=concurrency)
    for index, replacement in zip(invalid_indices, retried, strict=True):
        predictions[index] = replacement

    write_predictions(out, predictions)
    _print_summary(summarize(predictions))
    typer.echo(f"Wrote repaired predictions to {out}")


@app.command()
def compare(
    runs: Annotated[list[Path], typer.Argument(help="Prediction JSONL files to compare.")],
    dataset: Annotated[
        Path | None,
        typer.Option(help="Optional dataset JSONL to backfill labels for older run files."),
    ] = None,
) -> None:
    if len(runs) < 2:
        raise typer.BadParameter("Provide at least two run files.")
    examples = read_examples(dataset) if dataset is not None else None
    summaries = [summarize(_with_dataset_labels(read_predictions(path), examples)) for path in runs]
    _print_comparison(summaries)


@app.command("summarize")
def summarize_run(
    run: Annotated[Path, typer.Argument(help="Prediction JSONL file to summarize.")],
    dataset: Annotated[
        Path | None,
        typer.Option(help="Optional dataset JSONL to backfill labels for older run files."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Optional path to write full JSON metrics."),
    ] = None,
) -> None:
    examples = read_examples(dataset) if dataset is not None else None
    summary = summarize(_with_dataset_labels(read_predictions(run), examples))
    _print_summary(summary)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote summary metrics to {out}")


@datasets_app.command("sst2")
def datasets_sst2(
    out: Annotated[
        Path,
        typer.Option(help="Where to write the converted SST-2 JSONL file."),
    ] = Path("data/sst2_dev.jsonl"),
    split: Annotated[str, typer.Option(help="SST-2 split: train or dev.")] = "dev",
    limit: Annotated[
        int | None,
        typer.Option(help="Optional maximum number of examples to convert."),
    ] = None,
) -> None:
    count = download_sst2(out, split=split, limit=limit)
    typer.echo(f"Wrote {count} SST-2 examples to {out}")


@datasets_app.command("ag-news")
def datasets_ag_news(
    out: Annotated[Path, typer.Option(help="Where to write AG News JSONL.")] = Path(
        "data/ag_news_test_100.jsonl"
    ),
    limit: Annotated[int | None, typer.Option(help="Maximum examples to convert.")] = 100,
) -> None:
    count = download_ag_news(out, limit=limit)
    typer.echo(f"Wrote {count} AG News examples to {out}")


@datasets_app.command("trec")
def datasets_trec(
    out: Annotated[Path, typer.Option(help="Where to write TREC JSONL.")] = Path(
        "data/trec_test_100.jsonl"
    ),
    limit: Annotated[int | None, typer.Option(help="Maximum examples to convert.")] = 100,
) -> None:
    count = download_trec(out, limit=limit)
    typer.echo(f"Wrote {count} TREC examples to {out}")


@datasets_app.command("boolq")
def datasets_boolq(
    out: Annotated[Path, typer.Option(help="Where to write BoolQ JSONL.")] = Path(
        "data/boolq_val_100.jsonl"
    ),
    limit: Annotated[int | None, typer.Option(help="Maximum examples to convert.")] = 100,
) -> None:
    count = download_boolq(out, limit=limit)
    typer.echo(f"Wrote {count} BoolQ examples to {out}")


@datasets_app.command("arc-challenge")
def datasets_arc_challenge(
    out: Annotated[Path, typer.Option(help="Where to write ARC-Challenge JSONL.")] = Path(
        "data/arc_challenge_dev_100.jsonl"
    ),
    split: Annotated[str, typer.Option(help="ARC split: Train, Dev, or Test.")] = "Dev",
    limit: Annotated[int | None, typer.Option(help="Maximum examples to convert.")] = 100,
) -> None:
    count = download_arc_challenge(out, split=split, limit=limit)
    typer.echo(f"Wrote {count} ARC-Challenge examples to {out}")


@datasets_app.command("20-newsgroups")
def datasets_20_newsgroups(
    out: Annotated[Path, typer.Option(help="Where to write 20 Newsgroups JSONL.")] = Path(
        "data/20_newsgroups_test_100.jsonl"
    ),
    limit: Annotated[int | None, typer.Option(help="Maximum examples to convert.")] = 100,
) -> None:
    count = download_20newsgroups(out, limit=limit)
    typer.echo(f"Wrote {count} 20 Newsgroups examples to {out}")


@datasets_app.command("starter-suite")
def datasets_starter_suite(
    out_dir: Annotated[
        Path,
        typer.Option(help="Directory for converted starter-suite JSONL files."),
    ] = Path("data"),
    limit: Annotated[int, typer.Option(help="Examples per dataset.")] = 100,
) -> None:
    counts = download_starter_suite(out_dir, limit=limit)
    for name, count in counts.items():
        typer.echo(f"{name}: {count}")


def _build_adapter(
    adapter: str,
    *,
    model: str,
    input_price_per_1m: float | None,
    output_price_per_1m: float | None,
    reasoning_effort: str,
    jev_base_url: str,
    predictions: Path | None,
    command: str | None,
):
    if adapter == "majority":
        return MajorityAdapter()
    if adapter == "sentiment-lexicon":
        return SentimentLexiconAdapter()
    if adapter == "jev-api":
        return JevAPIAdapter(
            model=model,
            base_url=jev_base_url,
            input_price_per_1m=0.042 if input_price_per_1m is None else input_price_per_1m,
        )
    if adapter == "jev-openrouter":
        base_url = (
            jev_base_url
            if jev_base_url != "https://api.typesafe.ai/v1/systemone"
            else "https://openrouter.ai/api/alpha/decisions"
        )
        return JevOpenRouterAdapter(
            model=model if model != "gpt-5-mini" else "~typesafe/jev-latest",
            base_url=base_url,
            input_price_per_1m=0.042 if input_price_per_1m is None else input_price_per_1m,
        )
    if adapter == "openai-letter":
        return OpenAILetterAdapter(
            model=model,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
            reasoning_effort=reasoning_effort,
        )
    if adapter == "openrouter-letter":
        return OpenRouterLetterAdapter(
            model=model,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
            reasoning_enabled=None if "gemini" in model.lower() else False,
        )
    if adapter == "predictions-file":
        if predictions is None:
            raise typer.BadParameter("--predictions is required for predictions-file")
        return PredictionsFileAdapter(predictions)
    if adapter == "command":
        if command is None:
            raise typer.BadParameter("--command is required for command")
        return CommandAdapter(command)
    raise typer.BadParameter(f"unknown adapter: {adapter}")


def _predict_all(adapter, examples, *, concurrency: int):
    if concurrency <= 1:
        return [adapter.predict(example) for example in examples]
    if concurrency < 1:
        raise typer.BadParameter("--concurrency must be at least 1")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(adapter.predict, examples))


def _print_summary(summary: Summary) -> None:
    typer.echo(f"Adapter: {summary.adapter}")
    typer.echo(f"Examples: {summary.n}")
    typer.echo(f"Accuracy: {summary.accuracy:.3f}")
    if summary.macro_f1 is not None:
        typer.echo(f"Macro F1: {summary.macro_f1:.3f}")
        typer.echo(f"Weighted F1: {summary.weighted_f1:.3f}")
    typer.echo(f"Rows with errors: {summary.errors}")
    typer.echo(f"Invalid outputs: {summary.invalid}")
    typer.echo(f"Mean latency: {summary.mean_latency_ms:.1f} ms")
    typer.echo(f"Median latency: {summary.median_latency_ms:.1f} ms")
    typer.echo(f"P95 latency: {summary.p95_latency_ms:.1f} ms")
    if summary.total_cost_usd is not None:
        typer.echo(f"Total cost: ${summary.total_cost_usd:.6f}")
        typer.echo(f"Mean cost/example: ${summary.mean_cost_usd:.6f}")
        typer.echo(f"Cost / 1k examples: ${summary.cost_per_1k_usd:.6f}")
        if summary.cost_per_correct_usd is not None:
            typer.echo(f"Cost / correct answer: ${summary.cost_per_correct_usd:.6f}")
    if summary.input_tokens is not None:
        typer.echo(f"Input tokens: {summary.input_tokens}")
    if summary.output_tokens is not None:
        typer.echo(f"Output tokens: {summary.output_tokens}")


def _with_dataset_labels(predictions, examples):
    if examples is None:
        return predictions
    labels_by_id = {example.id: example.correct_letter for example in examples}
    enriched = []
    for prediction in predictions:
        if prediction.correct_label is None and prediction.example_id in labels_by_id:
            prediction = prediction.model_copy(
                update={"correct_label": labels_by_id[prediction.example_id]}
            )
        enriched.append(prediction)
    return enriched


def _summary_from_dict(data: dict) -> Summary:
    return Summary(
        adapter=data["adapter"],
        n=data["n"],
        accuracy=data["accuracy"],
        errors=data["errors"],
        invalid=data["invalid"],
        error_rate=data["error_rate"],
        invalid_rate=data["invalid_rate"],
        macro_precision=data["macro_precision"],
        macro_recall=data["macro_recall"],
        macro_f1=data["macro_f1"],
        weighted_precision=data["weighted_precision"],
        weighted_recall=data["weighted_recall"],
        weighted_f1=data["weighted_f1"],
        mean_latency_ms=data["mean_latency_ms"],
        median_latency_ms=data["median_latency_ms"],
        p90_latency_ms=data["p90_latency_ms"],
        p95_latency_ms=data["p95_latency_ms"],
        p99_latency_ms=data["p99_latency_ms"],
        total_cost_usd=data["total_cost_usd"],
        mean_cost_usd=data["mean_cost_usd"],
        cost_per_1k_usd=data["cost_per_1k_usd"],
        cost_per_correct_usd=data["cost_per_correct_usd"],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        mean_input_tokens=data["mean_input_tokens"],
        mean_output_tokens=data["mean_output_tokens"],
        brier_score=data["brier_score"],
        ece=data["ece"],
        mean_confidence=data["mean_confidence"],
    )


def _print_comparison(summaries: list[Summary]) -> None:
    header = (
        "adapter",
        "n",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "errors",
        "invalid_rate",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "total_cost_usd",
        "cost_per_1k_usd",
        "cost_per_correct_usd",
    )
    typer.echo("\t".join(header))
    for summary in summaries:
        row = (
            summary.adapter,
            str(summary.n),
            f"{summary.accuracy:.3f}",
            "" if summary.macro_f1 is None else f"{summary.macro_f1:.3f}",
            "" if summary.weighted_f1 is None else f"{summary.weighted_f1:.3f}",
            str(summary.errors),
            f"{summary.invalid_rate:.3f}",
            f"{summary.mean_latency_ms:.1f}",
            f"{summary.median_latency_ms:.1f}",
            f"{summary.p95_latency_ms:.1f}",
            "" if summary.total_cost_usd is None else f"{summary.total_cost_usd:.6f}",
            "" if summary.cost_per_1k_usd is None else f"{summary.cost_per_1k_usd:.6f}",
            "" if summary.cost_per_correct_usd is None else f"{summary.cost_per_correct_usd:.6f}",
        )
        typer.echo("\t".join(row))


if __name__ == "__main__":
    app()
