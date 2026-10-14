# OpenRouter Decision-Model Results

This run compares dedicated decision models through OpenRouter's Decisions API, using the same 5-dataset focused suite as the main benchmark.

- Included: Jev, OpenAI Luna Decisions, Microsoft Decision 1, Perplexity Decider, Drex v1.5, Clef Flash, Mercury Decide free
- Datasets: Banking77, AG News, BoolQ, ARC Challenge, MMLU
- Examples: 2,000 per dataset
- Total decisions: 70,000
- Total reported cost: $0.97

## Aggregate results

| Configuration | Accuracy | Macro F1 | Failures | Cost | p50 latency |
|---|---:|---:|---:|---:|---:|
| Clef Flash | 92.0% | 0.877 | 0 | $0.11 | 372 ms |
| Mercury Decide free | 90.2% | 0.850 | 0 | $0.00 | 504 ms |
| Perplexity Decider | 89.5% | 0.852 | 0 | $0.07 | 311 ms |
| Jev | 89.0% | 0.848 | 0 | $0.27 | 272 ms |
| Microsoft Decision 1 | 87.5% | 0.829 | 0 | $0.10 | 314 ms |
| Drex v1.5 | 86.8% | 0.828 | 0 | $0.09 | 274 ms |
| OpenAI Luna Decisions | 86.1% | 0.822 | 9 | $0.34 | 292 ms |

## Per-dataset accuracy

| Configuration | Avg acc | Banking77 | AG News | BoolQ | ARC | MMLU |
|---|---:|---:|---:|---:|---:|---:|
| Clef Flash | 92.0% | 94.5% | 85.0% | 91.3% | 98.8% | 90.1% |
| Mercury Decide free | 90.2% | 90.3% | 87.9% | 92.5% | 96.4% | 83.8% |
| Perplexity Decider | 89.5% | 82.9% | 85.9% | 94.5% | 96.9% | 87.3% |
| Jev | 89.0% | 79.8% | 81.9% | 92.7% | 97.8% | 93.1% |
| Microsoft Decision 1 | 87.5% | 86.4% | 80.5% | 92.5% | 96.2% | 82.0% |
| Drex v1.5 | 86.8% | 84.7% | 83.5% | 90.1% | 97.1% | 78.5% |
| OpenAI Luna Decisions | 86.1% | 79.9% | 81.3% | 90.5% | 96.0% | 82.9% |

## Per-dataset winners

| Dataset | Winner | Accuracy |
|---|---|---:|
| Banking77 | Clef Flash | 94.5% |
| AG News | Mercury Decide free | 87.9% |
| BoolQ | Perplexity Decider | 94.5% |
| ARC Challenge | Clef Flash | 98.8% |
| MMLU | Jev | 93.1% |

## Takeaways

- Dedicated decision models are a much stronger comparison set than generic chat LLMs for this task.
- Clef Flash is the aggregate winner, driven by very strong Banking77 and ARC results.
- Jev remains the strongest model on MMLU and the fastest/most stable paid route among the top cluster.
- Perplexity Decider is the best cheap paid route: close to Jev overall, stronger on BoolQ and AG News, and much cheaper in this run.
- Microsoft Decision 1 is strong on Banking77 and BoolQ, but weaker on MMLU.
- OpenAI Luna Decisions was not competitive with the best decision routes in this run and had the only invalid/API failures.
- Mercury Decide free is surprisingly strong and free, but slower and materially weaker on MMLU.

Failures are counted as incorrect.
