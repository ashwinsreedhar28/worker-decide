50 questions compared (129 skipped: errored or missing in a run)

| per-question max \|Δp\| | max | p99 | p90 | median | mean |
|---|---|---|---|---|---|
| SGLang-h100-80gb-hbm3 vs truth | 0.0082 | 0.0082 | 0.0061 | 0.00236 | 0.00301 |
| WorkersAI vs truth | 0.1435 | 0.1435 | 0.0917 | 0.04292 | 0.05220 |
| SGLang-h100-80gb-hbm3 vs WorkersAI | 0.1374 | 0.1374 | 0.0936 | 0.04457 | 0.05230 |

Argmax flips (clear = truth top-2 margin > 0.04):
- SGLang-h100-80gb-hbm3 vs truth: 0 total, 0 clear
- WorkersAI vs truth: 5 total, 2 clear
  - CLEAR: pulse-10619114/urgency (margin 0.085)
  - CLEAR: pulse-11711631/urgency (margin 0.077)
- SGLang-h100-80gb-hbm3 vs WorkersAI: 5 total, 2 clear
  - CLEAR: pulse-10619114/urgency (margin 0.085)
  - CLEAR: pulse-11711631/urgency (margin 0.077)

SGLang-h100-80gb-hbm3 closer to truth on 50 questions, WorkersAI closer on 0.

Largest SGLang-h100-80gb-hbm3-vs-WorkersAI gaps:
- pulse-10545020/urgency: SGLang-h100-80gb-hbm3-WorkersAI 0.1374, SGLang-h100-80gb-hbm3-truth 0.0061, WorkersAI-truth 0.1435, truth margin 0.001
- pulse-9710164/urgency: SGLang-h100-80gb-hbm3-WorkersAI 0.1334, SGLang-h100-80gb-hbm3-truth 0.0033, WorkersAI-truth 0.1316, truth margin 0.061
- pulse-10702774/urgency: SGLang-h100-80gb-hbm3-WorkersAI 0.1102, SGLang-h100-80gb-hbm3-truth 0.0082, WorkersAI-truth 0.1184, truth margin 0.040
- pulse-11716347/urgency: SGLang-h100-80gb-hbm3-WorkersAI 0.1094, SGLang-h100-80gb-hbm3-truth 0.0023, WorkersAI-truth 0.1117, truth margin 0.112
- pulse-11714938/urgency: SGLang-h100-80gb-hbm3-WorkersAI 0.0936, SGLang-h100-80gb-hbm3-truth 0.0046, WorkersAI-truth 0.0898, truth margin 0.035
- pulse-11711957/urgency: SGLang-h100-80gb-hbm3-WorkersAI 0.0935, SGLang-h100-80gb-hbm3-truth 0.0036, WorkersAI-truth 0.0917, truth margin 0.006
