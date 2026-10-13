# Good-Tier Letter-Only Results

This run tests the stronger, clean letter/code setup after removing models that were a bad fit for the one-token classification path:

- Included: Jev, OpenAI Sol, DeepSeek Pro, Qwen Plus
- Excluded: OpenAI Terra, Anthropic Sonnet/Opus, Gemini Pro, Mistral, Kimi
- Datasets: Banking77, AG News, BoolQ, ARC Challenge, MMLU
- Examples: 2,000 per dataset
- Total decisions: 40,000

OpenAI Sol on MMLU had a route-specific issue through OpenRouter: many calls returned an empty assistant message. Those invalid rows were rerun several times with a larger token cap. The final merged result uses every successful repair, but 392 MMLU Sol rows still failed with empty responses and are counted as incorrect.

## Aggregate results

| Configuration | Accuracy | Failures | Cost | p50 latency |
|---|---:|---:|---:|---:|
| Jev | 89.1% | 1 | $0.27 | 283 ms |
| Qwen Plus letter | 88.3% | 0 | $1.13 | 1279 ms |
| OpenAI Sol letter | 86.8% | 392 | $6.59 | 1190 ms |
| DeepSeek Pro letter | 86.1% | 33 | $1.35 | 1016 ms |

## Per-dataset accuracy

| Dataset | Jev | OpenAI Sol | DeepSeek Pro | Qwen Plus | Winner |
|---|---:|---:|---:|---:|---|
| Banking77 | 80.0% | 86.9% | 77.2% | 80.8% | OpenAI Sol |
| AG News | 81.8% | 83.8% | 79.7% | 81.7% | OpenAI Sol |
| BoolQ | 92.5% | 91.7% | 89.5% | 90.7% | Jev |
| ARC Challenge | 97.8% | 97.7% | 96.9% | 97.9% | Qwen Plus |
| MMLU | 93.2% | 73.9% | 87.2% | 90.4% | Jev |

## Takeaways

- Jev remains the best aggregate performer and is much cheaper/faster than the stronger LLMs.
- Qwen Plus is the closest clean LLM in this run: no failures, low cost, and only 0.8 percentage points behind Jev overall.
- OpenAI Sol is strongest on Banking77 and AG News, but the OpenRouter route was unreliable on MMLU even after repeated invalid reruns.
- DeepSeek Pro was not an improvement over the cheaper DeepSeek Flash result from the previous run.
- The strongest-model story is dataset-dependent: Sol wins broad classification, while Jev remains strongest on BoolQ and MMLU, and Qwen edges ARC.

## Interpretation caveat

These are public benchmark datasets. Training-data contamination cannot be ruled out for Jev or any general-purpose LLM. Treat these results as a useful public benchmark comparison, not as proof of real-world generalization. A private/fresh validation set is the next important check.
