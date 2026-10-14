# Tiny-Tier Letter-Only Results

This run tests very small OpenRouter models against Jev using the same single-letter/code classification path as the main benchmark.

- Included: Jev, Qwen 2.5 7B, Command R7B, Gemma 3 4B, Ministral 3B, Llama 3.2 3B, Llama 3.2 1B
- Datasets: Banking77, AG News, BoolQ, ARC Challenge, MMLU
- Examples: 2,000 per dataset
- Total decisions: 70,000
- Total reported cost: $1.55

No general-purpose sub-1B / 500M-ish text classification route was available in the OpenRouter model list at the time of setup, so Llama 3.2 1B is the smallest included endpoint.

## Aggregate results

| Configuration | Accuracy | Macro F1 | Failures | Cost | p50 latency |
|---|---:|---:|---:|---:|---:|
| Jev | 89.1% | 0.850 | 0 | $0.27 | 251 ms |
| Qwen 2.5 7B letter | 76.0% | 0.734 | 14 | $0.34 | 357 ms |
| Ministral 3B letter | 74.2% | 0.719 | 0 | $0.35 | 428 ms |
| Command R7B letter | 74.0% | 0.714 | 16 | $0.14 | 396 ms |
| Gemma 3 4B letter | 70.4% | 0.638 | 2 | $0.19 | 433 ms |
| Llama 3.2 3B letter | 65.7% | 0.650 | 0 | $0.17 | 347 ms |
| Llama 3.2 1B letter | 31.9% | 0.281 | 128 | $0.09 | 296 ms |

## Per-dataset accuracy

| Configuration | Avg acc | Banking77 | AG News | BoolQ | ARC | MMLU |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 89.1% | 80.2% | 81.8% | 92.3% | 97.8% | 93.2% |
| Qwen 2.5 7B letter | 76.0% | 57.1% | 79.9% | 86.2% | 88.8% | 68.3% |
| Ministral 3B letter | 74.2% | 57.2% | 81.5% | 83.2% | 86.2% | 63.1% |
| Command R7B letter | 74.0% | 56.2% | 84.0% | 88.0% | 83.5% | 58.2% |
| Gemma 3 4B letter | 70.4% | 62.1% | 75.5% | 82.5% | 78.0% | 53.9% |
| Llama 3.2 3B letter | 65.7% | 45.4% | 74.2% | 81.0% | 75.2% | 52.6% |
| Llama 3.2 1B letter | 31.9% | 0.2% | 36.5% | 61.2% | 31.6% | 30.1% |

## Takeaways

- Jev is far ahead of every tiny general-purpose LLM in the aggregate.
- The tiny models are weakest on the 77-way Banking77 intent task and MMLU.
- Command R7B is the most cost-efficient tiny LLM in this setup and is the only tiny model to beat Jev on a dataset, winning AG News at 84.0%.
- Qwen 2.5 7B is the strongest tiny LLM overall by accuracy, but it is still 13.1 percentage points behind Jev.
- Llama 3.2 1B appears below the useful floor for this benchmark, especially on Banking77.

Failures are counted as incorrect. They were mostly invalid model outputs or upstream response issues from the routed model/provider.
