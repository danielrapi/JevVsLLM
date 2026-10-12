# Jev vs. Cheap Classification Models

Benchmark language classification tasks across decision models and cheap LLMs using the same multiple-choice interface.

The current experiment compares Jev on OpenRouter against low-cost OpenRouter chat models. Each example is converted to a question with options `A`, `B`, `C`, etc. Chat models must answer with one letter; Jev uses the OpenRouter Decisions API.

## Current takeaways

Starter suite: 6 datasets × 100 examples each.

| Model | Avg accuracy | Invalid rate | Avg p50 latency | Sum cost/1k across suite |
|---|---:|---:|---:|---:|
| Qwen 3.7 Flash | 89.0% | 0.0% | 750 ms | $0.0357 |
| Jev OpenRouter | 88.8% | 0.0% | 292 ms | $0.1161 |
| DeepSeek V4.1 Flash | 88.7% | 0.2% | 418 ms | $0.1473 |
| OpenAI Luna | 88.2% | 0.0% | 836 ms | $0.2396 |
| Claude Haiku | 87.7% | 0.3% | 1144 ms | $0.1589 |
| Gemini Flash Lite | 80.5% | 9.8% | 615 ms | $0.3204 |
| Llama Scout | 80.3% | 2.8% | 763 ms | $0.1564 |
| Mistral Small | 64.0% | 19.5% | 573 ms | $0.1262 |

High-level read:

- Jev is the fastest and most stable: zero invalids and ~292 ms average median latency.
- Qwen and DeepSeek are very competitive on accuracy once reasoning is disabled.
- Luna is close on accuracy, but BYOK means cost is paid through the OpenAI account, not OpenRouter.
- Mistral and Gemini need more prompt/provider tuning before they are fair cheap-tier candidates.

Full merged summary:

[summaries/cheap_tier_starter_suite_100.json](/Users/danielrapoport/.codex/.chatgpt-projects/g-p-6acbd04100648191899cdf9d909eb0f2/summaries/cheap_tier_starter_suite_100.json)

Compact result table:

[summaries/cheap_tier_starter_suite_100_table.md](/Users/danielrapoport/.codex/.chatgpt-projects/g-p-6acbd04100648191899cdf9d909eb0f2/summaries/cheap_tier_starter_suite_100_table.md)

## Datasets

All datasets are converted to a common JSONL schema.

| Dataset | Task | File |
|---|---|---|
| SST-2 | sentiment | `data/sst2_dev_100.jsonl` |
| AG News | topic classification | `data/ag_news_test_100.jsonl` |
| TREC | question type | `data/trec_test_100.jsonl` |
| BoolQ | yes/no reading comprehension | `data/boolq_val_100.jsonl` |
| ARC-Challenge | science multiple choice | `data/arc_challenge_dev_100.jsonl` |
| 20 Newsgroups | 20-way topic classification | `data/20_newsgroups_test_100.jsonl` |

Example row:

```json
{
  "id": "sst2_dev_00000",
  "question": "Classify the sentiment of this sentence: ...",
  "options": ["negative", "positive"],
  "answer": "B"
}
```

## Metrics

The harness tracks:

- accuracy
- macro-F1 and weighted-F1
- invalid-output rate
- error rate
- p50 / p95 latency
- token counts
- estimated cost
- cost per 1k examples
- cost per correct answer

Invalid outputs count as wrong. That is intentional: invalid output is a product failure for classification.

## Running

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[openai,dev]"
```

Create `.env`:

```bash
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
TYPESAFE_API_KEY=...
```

Download the starter dataset suite:

```bash
PYTHONPATH=src python3 -m jevbench.cli datasets starter-suite \
  --out-dir data \
  --limit 100
```

Run the cheap-tier suite:

```bash
PYTHONPATH=src python3 -m jevbench.cli run-suite \
  --models-manifest experiments/cheap_tier_candidates.json \
  --datasets-manifest experiments/starter_datasets_100.json \
  --out-dir runs/cheap_tier_starter_suite_100 \
  --summaries-out summaries/cheap_tier_starter_suite_100.json \
  --concurrency 4
```

Retry only invalid/error rows:

```bash
PYTHONPATH=src python3 -m jevbench.cli retry-invalid \
  --run runs/.../model.jsonl \
  --dataset data/sst2_dev_100.jsonl \
  --adapter openrouter-letter \
  --model qwen/qwen3.7-flash \
  --out runs/.../model_repaired.jsonl
```

## Notes

- OpenRouter BYOK can report zero cost even when the upstream provider charges separately. The current Luna summary is manually adjusted with OpenAI Luna pricing.
- Some models need provider-specific controls. DeepSeek and Qwen required `reasoning.enabled=false` to avoid spending the small output budget on hidden reasoning.
- Gemini rejected or mishandled the reasoning parameter, so Gemini was rerun with that parameter omitted.
