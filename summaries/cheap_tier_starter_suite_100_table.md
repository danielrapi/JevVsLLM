| Model | Avg accuracy | Avg macro-F1 | Invalid rate | Avg p50 latency | Avg p95 latency | Sum cost/1k across suite |
|---|---:|---:|---:|---:|---:|---:|
| qwen_flash | 89.0% | 0.858 | 0.0% | 750 ms | 1028 ms | $0.0357 |
| jev_openrouter | 88.8% | 0.854 | 0.0% | 292 ms | 399 ms | $0.1161 |
| deepseek_flash_latest | 88.7% | 0.853 | 0.2% | 418 ms | 1335 ms | $0.1473 |
| openai_luna | 88.2% | 0.847 | 0.0% | 836 ms | 1259 ms | $0.2396 |
| claude_haiku | 87.7% | 0.848 | 0.3% | 1144 ms | 1490 ms | $0.1589 |
| gemini_flash_lite | 80.5% | 0.858 | 9.8% | 615 ms | 751 ms | $0.3204 |
| llama_scout | 80.3% | 0.803 | 2.8% | 763 ms | 1397 ms | $0.1564 |
| mistral_small | 64.0% | 0.779 | 19.5% | 573 ms | 5222 ms | $0.1262 |

| Dataset | Best accuracy | Jev accuracy | Notes |
|---|---:|---:|---|
| sst2 | deepseek_flash_latest 98.0% | 95.0% | deepseek_flash_latest led on this 100-example slice. |
| ag_news | jev_openrouter 84.0% | 84.0% | jev_openrouter led on this 100-example slice. |
| trec | qwen_flash 93.0% | 86.0% | qwen_flash led on this 100-example slice. |
| boolq | jev_openrouter 95.0% | 95.0% | jev_openrouter led on this 100-example slice. |
| arc_challenge | jev_openrouter 99.0% | 99.0% | jev_openrouter led on this 100-example slice. |
| 20_newsgroups | openai_luna 76.0% | 74.0% | openai_luna led on this 100-example slice. |
