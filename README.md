# Jev vs. LLMs for Classification

This repo benchmarks Jev against low-cost LLMs on language classification tasks.

The core question:

> For classification, can general-purpose LLMs match a decision model like Jev on accuracy, latency, and cost?

All models are evaluated through OpenRouter. Each task is converted into a common multiple-choice format. Chat models receive the question plus answer options and must return a single letter (`A`, `B`, `C`, ...). Jev is evaluated through OpenRouter’s Decisions API.

## Headline results

Starter suite: 6 datasets × 100 examples each.

| Model | Avg accuracy | Avg p50 latency | Cost / 1k, summed across suite |
|---|---:|---:|---:|
| Qwen 3.7 Flash | 89.0% | 750 ms | $0.0357 |
| Jev | 88.8% | 292 ms | $0.1161 |
| DeepSeek V4.1 Flash | 88.7% | 418 ms | $0.1473 |
| OpenAI Luna | 88.2% | 836 ms | $0.2396 |
| Claude Haiku | 87.7% | 1144 ms | $0.1589 |
| Gemini Flash Lite | 80.5% | 615 ms | $0.3204 |
| Llama Scout | 80.3% | 763 ms | $0.1564 |
| Mistral Small | 64.0% | 573 ms | $0.1262 |

Early takeaways:

- Jev is the latency leader by a large margin.
- Cheap LLMs can be competitive on accuracy for simple classification.
- Qwen and DeepSeek are especially strong low-cost baselines in this first suite.
- Larger and harder datasets are needed before treating these as final rankings.

Full results:

- [Merged JSON summary](summaries/cheap_tier_starter_suite_100.json)
- [Compact markdown table](summaries/cheap_tier_starter_suite_100_table.md)

## Datasets

The starter suite contains 100 examples from each dataset:

| Dataset | Task |
|---|---|
| SST-2 | sentiment |
| AG News | news topic classification |
| TREC | question type classification |
| BoolQ | yes/no reading comprehension |
| ARC-Challenge | science multiple-choice QA |
| 20 Newsgroups | 20-way topic classification |

All datasets are converted to the same JSONL schema:

```json
{
  "id": "sst2_dev_00000",
  "question": "Classify the sentiment of this sentence: ...",
  "options": ["negative", "positive"],
  "answer": "B"
}
```

## Metrics

The benchmark reports:

- accuracy
- macro-F1 and weighted-F1
- p50 / p95 latency
- token usage
- estimated cost
- cost per correct answer

## Repo structure

```text
data/          Converted benchmark datasets
experiments/   Model and dataset manifests
src/jevbench/  Benchmark harness
summaries/     Result summaries
tests/         Unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[openai,dev]"
```

Create `.env`:

```bash
OPENROUTER_API_KEY=...
```

## Download datasets

```bash
PYTHONPATH=src python3 -m jevbench.cli datasets starter-suite \
  --out-dir data \
  --limit 100
```

## Run the cheap-tier suite

```bash
PYTHONPATH=src python3 -m jevbench.cli run-suite \
  --models-manifest experiments/cheap_tier_candidates.json \
  --datasets-manifest experiments/starter_datasets_100.json \
  --out-dir runs/cheap_tier_starter_suite_100 \
  --summaries-out summaries/cheap_tier_starter_suite_100.json \
  --concurrency 4
```

## Run one model on one dataset

```bash
PYTHONPATH=src python3 -m jevbench.cli run \
  --dataset data/sst2_dev_100.jsonl \
  --adapter openrouter-letter \
  --model qwen/qwen3.7-flash \
  --input-price-per-1m 0.03 \
  --output-price-per-1m 0.13 \
  --out runs/qwen_sst2_100.jsonl \
  --concurrency 4
```

## Compare runs

```bash
PYTHONPATH=src python3 -m jevbench.cli compare \
  --dataset data/sst2_dev_100.jsonl \
  runs/cheap_tier_starter_suite_100/sst2/jev_openrouter.jsonl \
  runs/cheap_tier_starter_suite_100/sst2/qwen_flash.jsonl
```

## Notes

This is an early benchmark harness, not a final leaderboard. The first suite is deliberately small so that model behavior, cost, and latency can be inspected quickly. The next step is to scale to more datasets and larger samples.
