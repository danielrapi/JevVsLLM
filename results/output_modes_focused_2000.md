# Full Results: Letter vs Structured-Label Output Modes

This report summarizes the 5-dataset, 2,000-example-per-dataset run comparing Jev with low-cost chat models on OpenRouter. Chat models were evaluated in two modes:

- `letter`: the model returns only the option code, such as `A` or `3`.
- `structured`: the model returns JSON containing the real category label, such as `{ "answer": "business" }`.

## Scope

| Item | Value |
|---|---:|
| Datasets | 5 |
| Examples per dataset | 2,000 |
| Total evaluated decisions | 130,000 |
| Total measured API cost | $8.20 |
| Total output/API failures | 2,846 |

## Aggregate leaderboard

| Configuration | Accuracy | Failures | p50 latency | p95 latency | Cost | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 89.0% | 0 | 288 ms | 375 ms | $0.27 | $0.0270 |
| DeepSeek Flash letter | 88.0% | 10 | 395 ms | 1334 ms | $0.55 | $0.0552 |
| Luna letter | 87.7% | 0 | 825 ms | 1279 ms | $0.68 | $0.0683 |
| DeepSeek Flash structured | 87.2% | 6 | 444 ms | 1420 ms | $0.70 | $0.0696 |
| Gemini Flash Lite letter | 87.0% | 27 | 659 ms | 820 ms | $1.11 | $0.1112 |
| Gemini Flash Lite structured | 86.5% | 7 | 696 ms | 895 ms | $1.49 | $0.1488 |
| Luna structured | 86.4% | 3 | 898 ms | 1746 ms | $0.89 | $0.0888 |
| Qwen Flash letter | 85.8% | 0 | 751 ms | 1035 ms | $0.11 | $0.0113 |
| Qwen Flash structured | 84.6% | 10 | 827 ms | 1074 ms | $0.13 | $0.0134 |
| Claude Haiku structured | 83.5% | 272 | 1131 ms | 1482 ms | $0.69 | $0.0695 |
| Claude Haiku letter | 83.1% | 198 | 1153 ms | 1515 ms | $0.47 | $0.0471 |
| Llama Scout letter | 74.5% | 914 | 1631 ms | 7055 ms | $0.49 | $0.0491 |
| Llama Scout structured | 68.9% | 1,399 | 3118 ms | 11942 ms | $0.61 | $0.0606 |

Failures are counted as incorrect in accuracy. They fall into two main buckets: upstream provider/API failures, mostly OpenRouter shared-pool rate limits for Llama Scout, and schema failures where a model returned free-form reasoning text instead of the required letter/code or JSON label.

| Configuration | Upstream rate limits | Schema/format failures | Other errors | Total failures |
|---|---:|---:|---:|---:|
| Llama Scout structured | 1,371 | 12 | 16 | 1,399 |
| Llama Scout letter | 906 | 0 | 8 | 914 |
| Claude Haiku structured | 0 | 268 | 6 | 274 |
| Claude Haiku letter | 0 | 0 | 198 | 198 |
| Gemini Flash Lite letter | 0 | 0 | 27 | 27 |
| DeepSeek Flash letter | 0 | 0 | 10 | 10 |
| Qwen Flash structured | 0 | 1 | 9 | 10 |
| Gemini Flash Lite structured | 0 | 0 | 7 | 7 |
| DeepSeek Flash structured | 0 | 5 | 1 | 6 |
| Luna structured | 0 | 0 | 3 | 3 |

## Output-mode comparison

| Model | Letter acc | Structured acc | Δ acc | Letter cost / 1k | Structured cost / 1k | Cost × | p50 latency × | Output tokens × |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen_flash | 85.8% | 84.6% | -1.2 pp | $0.0113 | $0.0134 | 1.19× | 1.10× | 8.41× |
| llama_scout | 74.5% | 68.9% | -5.6 pp | $0.0491 | $0.0606 | 1.23× | 2.39× | 4.78× |
| claude_haiku | 83.1% | 83.5% | +0.3 pp | $0.0471 | $0.0695 | 1.48× | 0.98× | 5.83× |
| deepseek_flash | 88.0% | 87.2% | -0.8 pp | $0.0552 | $0.0696 | 1.26× | 1.12× | 5.11× |
| openai_luna | 87.7% | 86.4% | -1.2 pp | $0.0683 | $0.0888 | 1.30× | 1.09× | 2.40× |
| gemini_flash_lite | 87.0% | 86.5% | -0.5 pp | $0.1112 | $0.1488 | 1.34× | 1.06× | 8.17× |

## Accuracy distribution by dataset

Aggregate accuracy is useful, but it hides whether a model is consistently good or only helped by easier datasets. This table shows each dataset separately plus the spread between a configuration’s best and worst dataset.

| Configuration | Avg acc | Min | Max | Spread | Banking77 | AG News | BoolQ | ARC | MMLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | 89.0% | 79.6% | 97.8% | 18.2 pp | 79.6% | 81.8% | 92.4% | 97.8% | 93.2% |
| DeepSeek Flash letter | 88.0% | 79.1% | 96.5% | 17.4 pp | 79.1% | 84.3% | 91.6% | 96.5% | 88.5% |
| Luna letter | 87.7% | 82.5% | 96.4% | 13.9 pp | 85.2% | 82.5% | 91.0% | 96.4% | 83.2% |
| DeepSeek Flash structured | 87.2% | 79.4% | 92.4% | 13.0 pp | 79.4% | 86.1% | 91.7% | 92.4% | 86.2% |
| Gemini Flash Lite letter | 87.0% | 79.2% | 97.5% | 18.2 pp | 79.2% | 82.3% | 90.3% | 97.5% | 85.7% |
| Gemini Flash Lite structured | 86.5% | 79.0% | 92.8% | 13.8 pp | 79.0% | 83.5% | 90.8% | 92.8% | 86.3% |
| Luna structured | 86.4% | 79.4% | 92.5% | 13.1 pp | 85.0% | 79.4% | 90.6% | 92.5% | 84.5% |
| Qwen Flash letter | 85.8% | 76.0% | 96.3% | 20.3 pp | 76.0% | 82.0% | 90.8% | 96.3% | 83.9% |
| Qwen Flash structured | 84.6% | 76.6% | 91.7% | 15.1 pp | 76.6% | 82.1% | 90.2% | 91.7% | 82.5% |
| Claude Haiku structured | 83.5% | 74.8% | 91.5% | 16.6 pp | 78.8% | 82.1% | 90.1% | 91.5% | 74.8% |
| Claude Haiku letter | 83.1% | 75.1% | 94.3% | 19.2 pp | 75.1% | 81.7% | 89.4% | 94.3% | 75.1% |
| Llama Scout letter | 74.5% | 65.1% | 84.0% | 18.9 pp | 65.1% | 72.9% | 76.3% | 84.0% | 74.0% |
| Llama Scout structured | 68.9% | 56.4% | 79.6% | 23.2 pp | 62.9% | 70.6% | 75.0% | 79.6% | 56.4% |

## Per-dataset winners and gaps

| Dataset | Best configuration | Best acc | Jev acc | Best LLM gap vs Jev | Notes |
|---|---|---:|---:|---:|---|
| Banking77 | Luna letter | 85.2% | 79.6% | +5.6 pp | Luna letter is best LLM |
| AG News | DeepSeek Flash structured | 86.1% | 81.8% | +4.3 pp | DeepSeek Flash structured is best LLM |
| BoolQ | Jev | 92.4% | 92.4% | -0.7 pp | Jev wins |
| ARC Challenge | Jev | 97.8% | 97.8% | -0.3 pp | Jev wins |
| MMLU | Jev | 93.2% | 93.2% | -4.7 pp | Jev wins |

## Letter vs structured accuracy by dataset

Positive values mean structured-label JSON was more accurate than letter/code mode for that dataset.

| Model | Banking77 | AG News | BoolQ | ARC | MMLU | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Luna structured − letter | -0.2 pp | -3.0 pp | -0.4 pp | -3.8 pp | +1.2 pp | -1.2 pp |
| Claude Haiku structured − letter | +3.7 pp | +0.4 pp | +0.7 pp | -2.8 pp | -0.3 pp | +0.3 pp |
| Gemini Flash Lite structured − letter | -0.2 pp | +1.3 pp | +0.4 pp | -4.6 pp | +0.6 pp | -0.5 pp |
| DeepSeek Flash structured − letter | +0.3 pp | +1.8 pp | +0.1 pp | -4.1 pp | -2.3 pp | -0.8 pp |
| Qwen Flash structured − letter | +0.6 pp | +0.0 pp | -0.6 pp | -4.6 pp | -1.3 pp | -1.2 pp |
| Llama Scout structured − letter | -2.2 pp | -2.3 pp | -1.3 pp | -4.4 pp | -17.6 pp | -5.6 pp |

## Per-dataset results

### Banking77

| Configuration | Accuracy | Macro F1 | Invalids | p50 latency | p95 latency | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|
| Luna letter | 85.2% | 0.638 | 0 | 803 ms | 1253 ms | $0.1797 |
| Luna structured | 85.0% | 0.627 | 0 | 895 ms | 1354 ms | $0.2253 |
| Jev | 79.6% | 0.584 | 0 | 299 ms | 387 ms | $0.0659 |
| DeepSeek Flash structured | 79.4% | 0.581 | 1 | 509 ms | 1635 ms | $0.1998 |
| Gemini Flash Lite letter | 79.2% | 0.606 | 0 | 674 ms | 827 ms | $0.3439 |
| DeepSeek Flash letter | 79.1% | 0.594 | 2 | 434 ms | 1271 ms | $0.1658 |
| Gemini Flash Lite structured | 79.0% | 0.594 | 0 | 703 ms | 875 ms | $0.4194 |
| Claude Haiku structured | 78.8% | 0.604 | 0 | 1124 ms | 1480 ms | $0.1878 |
| Qwen Flash structured | 76.6% | 0.570 | 1 | 822 ms | 1044 ms | $0.0343 |
| Qwen Flash letter | 76.0% | 0.589 | 0 | 769 ms | 1097 ms | $0.0321 |
| Claude Haiku letter | 75.1% | 0.509 | 0 | 1158 ms | 1485 ms | $0.1270 |
| Llama Scout letter | 65.1% | 0.502 | 196 | 1065 ms | 5907 ms | $0.1368 |
| Llama Scout structured | 62.9% | 0.514 | 253 | 2840 ms | 6257 ms | $0.1653 |

### AG News

| Configuration | Accuracy | Macro F1 | Invalids | p50 latency | p95 latency | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek Flash structured | 86.1% | 0.860 | 0 | 403 ms | 1506 ms | $0.0329 |
| DeepSeek Flash letter | 84.3% | 0.847 | 1 | 409 ms | 1708 ms | $0.0266 |
| Gemini Flash Lite structured | 83.5% | 0.839 | 0 | 659 ms | 827 ms | $0.0705 |
| Luna letter | 82.5% | 0.831 | 0 | 807 ms | 1206 ms | $0.0388 |
| Gemini Flash Lite letter | 82.3% | 0.829 | 0 | 644 ms | 799 ms | $0.0508 |
| Qwen Flash structured | 82.1% | 0.829 | 0 | 793 ms | 996 ms | $0.0072 |
| Claude Haiku structured | 82.1% | 0.828 | 0 | 1094 ms | 1423 ms | $0.0356 |
| Qwen Flash letter | 82.0% | 0.828 | 0 | 731 ms | 929 ms | $0.0058 |
| Jev | 81.8% | 0.826 | 0 | 299 ms | 405 ms | $0.0170 |
| Claude Haiku letter | 81.7% | 0.823 | 0 | 1144 ms | 1450 ms | $0.0263 |
| Luna structured | 79.4% | 0.805 | 0 | 877 ms | 1352 ms | $0.0485 |
| Llama Scout letter | 72.9% | 0.806 | 161 | 1159 ms | 6219 ms | $0.0258 |
| Llama Scout structured | 70.6% | 0.772 | 132 | 1351 ms | 6216 ms | $0.0316 |

### BoolQ

| Configuration | Accuracy | Macro F1 | Invalids | p50 latency | p95 latency | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 92.4% | 0.920 | 0 | 286 ms | 369 ms | $0.0195 |
| DeepSeek Flash structured | 91.7% | 0.914 | 2 | 436 ms | 1500 ms | $0.0423 |
| DeepSeek Flash letter | 91.6% | 0.912 | 0 | 375 ms | 1280 ms | $0.0363 |
| Luna letter | 91.0% | 0.905 | 0 | 816 ms | 1301 ms | $0.0525 |
| Qwen Flash letter | 90.8% | 0.904 | 0 | 744 ms | 1054 ms | $0.0081 |
| Gemini Flash Lite structured | 90.8% | 0.903 | 0 | 678 ms | 882 ms | $0.0896 |
| Luna structured | 90.6% | 0.902 | 0 | 864 ms | 1308 ms | $0.0611 |
| Gemini Flash Lite letter | 90.3% | 0.900 | 2 | 644 ms | 802 ms | $0.0720 |
| Qwen Flash structured | 90.2% | 0.898 | 0 | 813 ms | 1072 ms | $0.0094 |
| Claude Haiku structured | 90.1% | 0.896 | 0 | 1087 ms | 1405 ms | $0.0441 |
| Claude Haiku letter | 89.4% | 0.888 | 0 | 1123 ms | 1570 ms | $0.0366 |
| Llama Scout letter | 76.3% | 0.893 | 299 | 1938 ms | 11140 ms | $0.0344 |
| Llama Scout structured | 75.0% | 0.880 | 307 | 3944 ms | 20532 ms | $0.0416 |

### ARC Challenge

| Configuration | Accuracy | Macro F1 | Invalids | p50 latency | p95 latency | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 97.8% | 0.983 | 0 | 272 ms | 353 ms | $0.0159 |
| Gemini Flash Lite letter | 97.5% | 0.980 | 1 | 673 ms | 849 ms | $0.0426 |
| DeepSeek Flash letter | 96.5% | 0.973 | 2 | 355 ms | 1188 ms | $0.0206 |
| Luna letter | 96.4% | 0.971 | 0 | 852 ms | 1377 ms | $0.0337 |
| Qwen Flash letter | 96.3% | 0.970 | 0 | 754 ms | 1103 ms | $0.0050 |
| Claude Haiku letter | 94.3% | 0.957 | 7 | 1167 ms | 1560 ms | $0.0216 |
| Gemini Flash Lite structured | 92.8% | 0.942 | 0 | 765 ms | 1072 ms | $0.0782 |
| Luna structured | 92.5% | 0.940 | 0 | 923 ms | 1457 ms | $0.0523 |
| DeepSeek Flash structured | 92.4% | 0.939 | 0 | 419 ms | 1268 ms | $0.0330 |
| Qwen Flash structured | 91.7% | 0.934 | 2 | 846 ms | 1119 ms | $0.0076 |
| Claude Haiku structured | 91.5% | 0.937 | 16 | 1156 ms | 1503 ms | $0.0364 |
| Llama Scout letter | 84.0% | 0.938 | 180 | 2993 ms | 6226 ms | $0.0242 |
| Llama Scout structured | 79.6% | 0.909 | 204 | 2086 ms | 6051 ms | $0.0342 |

### MMLU

| Configuration | Accuracy | Macro F1 | Invalids | p50 latency | p95 latency | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 93.2% | 0.932 | 0 | 282 ms | 362 ms | $0.0165 |
| DeepSeek Flash letter | 88.5% | 0.888 | 5 | 403 ms | 1221 ms | $0.0267 |
| Gemini Flash Lite structured | 86.3% | 0.865 | 7 | 673 ms | 816 ms | $0.0863 |
| DeepSeek Flash structured | 86.2% | 0.863 | 3 | 451 ms | 1191 ms | $0.0399 |
| Gemini Flash Lite letter | 85.7% | 0.866 | 24 | 658 ms | 824 ms | $0.0469 |
| Luna structured | 84.5% | 0.846 | 3 | 932 ms | 3257 ms | $0.0571 |
| Qwen Flash letter | 83.9% | 0.838 | 0 | 755 ms | 991 ms | $0.0054 |
| Luna letter | 83.2% | 0.832 | 0 | 845 ms | 1257 ms | $0.0366 |
| Qwen Flash structured | 82.5% | 0.828 | 7 | 859 ms | 1141 ms | $0.0084 |
| Claude Haiku letter | 75.1% | 0.830 | 191 | 1171 ms | 1512 ms | $0.0239 |
| Claude Haiku structured | 74.8% | 0.858 | 256 | 1194 ms | 1599 ms | $0.0436 |
| Llama Scout letter | 74.0% | 0.770 | 78 | 997 ms | 5783 ms | $0.0244 |
| Llama Scout structured | 56.4% | 0.753 | 503 | 5371 ms | 20655 ms | $0.0302 |

## Takeaways

- Jev had the best aggregate accuracy in this run while producing zero invalid outputs.
- The letter/code output mode was generally better than normal structured-label JSON: similar or better accuracy, lower cost, and lower latency.
- Structured-label JSON increased output tokens by roughly 2.4× to 8.4× depending on the model.
- DeepSeek letter was the strongest cheap LLM competitor by aggregate accuracy.
- Qwen letter was the cheapest clean LLM configuration and had zero invalid outputs, but trailed Jev, DeepSeek, Luna, and Gemini on accuracy.
- Llama Scout was not reliable in this setup: it produced many invalid outputs and dominated runtime due to high tail latency.
- Banking77 was the hardest dataset for all systems, which is expected because it has 77 fine-grained intent labels.
