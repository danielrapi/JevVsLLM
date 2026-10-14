# Jev vs. LLMs and Decision Models for Classification

This repo benchmarks Jev against low-cost LLMs and other OpenRouter decision models on language classification tasks.

The core question:

> For classification, can general-purpose LLMs or dedicated decision models match Jev on accuracy, latency, and cost?

All model calls in the main experiment run through OpenRouter. Each task is converted into a common multiple-choice format. Chat models are tested in two output modes:

- **Letter/code mode:** append the answer options and ask the model to return only the selected code, such as `A` or `3`.
- **Structured-label mode:** ask the model to return JSON containing the real category name, such as `{ "answer": "business" }`.

Jev and the other decision-model baselines are evaluated through OpenRouter’s Decisions API.

## Headline result

The first main run compares Jev against general-purpose LLMs. It uses 5 datasets × 2,000 examples each, for 130,000 total decisions.

| Configuration | Accuracy | Cost | p50 latency |
|---|---:|---:|---:|
| Jev | 89.0% | $0.27 | 288 ms |
| DeepSeek Flash, letter | 88.0% | $0.55 | 395 ms |
| OpenAI Luna, letter | 87.7% | $0.68 | 825 ms |
| Gemini Flash Lite, letter | 87.0% | $1.11 | 659 ms |
| Qwen Flash, letter | 85.8% | $0.11 | 751 ms |
| Claude Haiku, structured | 83.5% | $0.69 | 1131 ms |
| Llama Scout, structured | 68.9% | $0.61 | 3118 ms |

Main takeaways:

- Jev had the best aggregate accuracy, lowest latency, and no output/API failures.
- The single-letter/code method was usually better than normal structured-label JSON: similar or better accuracy, lower cost, and lower latency.
- Structured-label output increased output tokens by roughly 2.4× to 8.4× depending on the model.
- DeepSeek Flash was the strongest cheap LLM competitor by accuracy.
- Qwen Flash was the cheapest clean LLM setup, but a few points behind Jev, DeepSeek, Luna, and Gemini.
- Llama Scout was not reliable in this prompt/setup: many invalid outputs and very high tail latency.

Output/API failures are counted as incorrect in accuracy. In this run they mostly meant either upstream OpenRouter/provider rate limits, especially for Llama Scout, or a model ignoring the requested response schema and returning reasoning text instead of the answer.

## Decision-model comparison

The final run compares dedicated decision models through OpenRouter’s Decisions API on the same 5 datasets × 2,000 examples setup.

| Configuration | Accuracy | Cost | p50 latency |
|---|---:|---:|---:|
| Clef Flash | 92.0% | $0.11 | 372 ms |
| Mercury Decide free | 90.2% | $0.00 | 504 ms |
| Perplexity Decider | 89.5% | $0.07 | 311 ms |
| Jev | 89.0% | $0.27 | 272 ms |
| Microsoft Decision 1 | 87.5% | $0.10 | 314 ms |
| Drex v1.5 | 86.8% | $0.09 | 274 ms |
| OpenAI Luna Decisions | 86.1% | $0.34 | 292 ms |

Main takeaways:

- Dedicated decision models are much stronger than generic chat LLMs for this benchmark.
- Clef Flash is the aggregate winner, driven by very strong Banking77 and ARC results.
- Jev remains the best on MMLU and is still the fastest model in the top accuracy cluster.
- Perplexity Decider is the strongest cheap paid route: it beats Jev overall in this run while costing much less.
- Microsoft Decision 1 is strong on Banking77 and BoolQ, but weaker on MMLU.
- Mercury Decide free is surprisingly competitive, though slower and weaker on MMLU.

Per-dataset decision-model winners:

| Dataset | Winner | Accuracy |
|---|---|---:|
| Banking77 | Clef Flash | 94.5% |
| AG News | Mercury Decide free | 87.9% |
| BoolQ | Perplexity Decider | 94.5% |
| ARC Challenge | Clef Flash | 98.8% |
| MMLU | Jev | 93.1% |

## Accuracy breakdown

Average accuracy hides some important structure. The main pattern is:

- LLMs beat Jev on the broad text-classification datasets: Banking77 and AG News.
- Jev wins on the harder multiple-choice / reasoning-style datasets: BoolQ, ARC Challenge, and MMLU.
- The letter/code format is usually at least as accurate as structured-label JSON, while being cheaper and producing fewer tokens.

| Dataset | Best configuration | Best acc | Jev acc | Best LLM gap vs Jev |
|---|---|---:|---:|---:|
| Banking77 | Luna letter | 85.2% | 79.6% | +5.6 pp |
| AG News | DeepSeek Flash structured | 86.1% | 81.8% | +4.3 pp |
| BoolQ | Jev | 92.4% | 92.4% | -0.7 pp |
| ARC Challenge | Jev | 97.8% | 97.8% | -0.3 pp |
| MMLU | Jev | 93.2% | 93.2% | -4.7 pp |

Per-dataset accuracy for the strongest configurations:

| Configuration | Avg acc | Banking77 | AG News | BoolQ | ARC | MMLU |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 89.0% | 79.6% | 81.8% | 92.4% | 97.8% | 93.2% |
| DeepSeek Flash letter | 88.0% | 79.1% | 84.3% | 91.6% | 96.5% | 88.5% |
| Luna letter | 87.7% | 85.2% | 82.5% | 91.0% | 96.4% | 83.2% |
| Gemini Flash Lite letter | 87.0% | 79.2% | 82.3% | 90.3% | 97.5% | 85.7% |
| Qwen Flash letter | 85.8% | 76.0% | 82.0% | 90.8% | 96.3% | 83.9% |

Full report:

- [Full per-dataset report](results/output_modes_focused_2000.md)
- [Good-tier letter-only report](results/good_tier_core_no_terra_letter_2000.md)
- [Tiny-tier letter-only report](results/tiny_tier_letter_2000.md)
- [OpenRouter decision-model report](results/decision_models_openrouter_2000.md)
- [Raw combined JSON summary](summaries/output_modes_focused_2000.json)
- [Letter-vs-structured comparison table](summaries/output_modes_focused_2000_modes.md)

## Good-tier letter-only run

After the cheap-tier output-mode test, we ran a smaller “good model” comparison using only the letter/code output path. Models that required reasoning-first behavior or had unstable routing for this setup were omitted.

| Configuration | Accuracy | Failures | Cost | p50 latency |
|---|---:|---:|---:|---:|
| Jev | 89.1% | 1 | $0.27 | 283 ms |
| Qwen Plus letter | 88.3% | 0 | $1.13 | 1279 ms |
| OpenAI Sol letter | 86.8% | 392 | $6.59 | 1190 ms |
| DeepSeek Pro letter | 86.1% | 33 | $1.35 | 1016 ms |

Per-dataset winners:

| Dataset | Winner | Accuracy |
|---|---|---:|
| Banking77 | OpenAI Sol | 86.9% |
| AG News | OpenAI Sol | 83.8% |
| BoolQ | Jev | 92.5% |
| ARC Challenge | Qwen Plus | 97.9% |
| MMLU | Jev | 93.2% |

OpenAI Sol’s MMLU route returned many empty OpenRouter responses. We reran those invalid rows several times with a larger output-token cap and merged all successful repairs, but 392 Sol/MMLU rows still failed and are counted as incorrect.

The good-tier result reinforces the earlier pattern: stronger LLMs help on broad text classification, but Jev remains strongest overall and especially strong on the harder QA-style datasets. Qwen Plus is the closest clean LLM competitor in this setup.

One caveat: these are public benchmark datasets, so training-data contamination cannot be ruled out for Jev or any LLM. A fresh/private validation set should be the next check before making strong generalization claims.

## Datasets

The focused run uses 2,000 examples from each dataset:

| Dataset | Task |
|---|---|
| Banking77 | 77-way banking intent classification |
| AG News | news topic classification |
| BoolQ | yes/no reading comprehension |
| ARC Challenge | science multiple-choice QA |
| MMLU | broad knowledge multiple-choice QA |

All datasets are converted to the same JSONL schema:

```json
{
  "id": "ag_news_train_00000",
  "question": "Classify this news article: ...",
  "options": ["World", "Sports", "Business", "Sci/Tech"],
  "answer": "2"
}
```

For datasets with up to 26 options, answer codes are letters. For larger label spaces, such as Banking77, answer codes are numeric.

## Metrics

The benchmark reports:

- accuracy
- macro-F1 and weighted-F1
- invalid-output rate
- p50 / p95 latency
- token usage
- estimated cost
- cost per correct answer

Invalid outputs count against accuracy.

## Repo structure

```text
data/          Converted benchmark datasets
experiments/   Model and dataset manifests
results/       Human-readable experiment reports
src/jevbench/  Benchmark harness
summaries/     Machine-readable result summaries
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

## Download the focused datasets

```bash
PYTHONPATH=src python3 -m jevbench.cli datasets focused-suite \
  --out-dir data \
  --limit 2000
```

## Run the focused output-mode suite

```bash
PYTHONPATH=src python3 -m jevbench.cli run-suite \
  --models-manifest experiments/cheap_tier_output_modes_no_mistral.json \
  --datasets-manifest experiments/focused_datasets_2000.json \
  --out-dir runs/output_modes_focused_2000 \
  --summaries-out summaries/output_modes_focused_2000.json \
  --concurrency 16 \
  --skip-existing
```

## Compare letter vs. structured output modes

```bash
PYTHONPATH=src python3 -m jevbench.cli compare-output-modes \
  summaries/output_modes_focused_2000.json \
  --out summaries/output_modes_focused_2000_modes.md
```

## Notes

This is still a benchmark harness, not a final leaderboard. The current results are useful for comparing output formats and low-cost model behavior, but future runs should add stronger models, retry policies for invalid outputs, and more datasets.
