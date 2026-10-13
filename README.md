# Jev vs. LLMs for Classification

This repo benchmarks Jev against low-cost LLMs on language classification tasks.

The core question:

> For classification, can general-purpose LLMs match a decision model like Jev on accuracy, latency, and cost?

All model calls in the main experiment run through OpenRouter. Each task is converted into a common multiple-choice format. Chat models are tested in two output modes:

- **Letter/code mode:** append the answer options and ask the model to return only the selected code, such as `A` or `3`.
- **Structured-label mode:** ask the model to return JSON containing the real category name, such as `{ "answer": "business" }`.

Jev is evaluated through OpenRouter’s Decisions API.

## Headline result

Main run: 5 datasets × 2,000 examples each, for 130,000 total decisions.

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
- [Raw combined JSON summary](summaries/output_modes_focused_2000.json)
- [Letter-vs-structured comparison table](summaries/output_modes_focused_2000_modes.md)

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
