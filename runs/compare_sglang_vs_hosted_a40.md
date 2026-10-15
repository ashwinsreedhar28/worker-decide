50 questions compared (129 skipped: errored or missing in a run)

| per-question max \|Δp\| | max | p99 | p90 | median | mean |
|---|---|---|---|---|---|
| SGLang-a40 vs truth | 0.0112 | 0.0112 | 0.0051 | 0.00265 | 0.00296 |
| WorkersAI vs truth | 0.1435 | 0.1435 | 0.0917 | 0.04292 | 0.05220 |
| SGLang-a40 vs WorkersAI | 0.1505 | 0.1505 | 0.0942 | 0.04292 | 0.05237 |

Argmax flips (clear = truth top-2 margin > 0.04):
- SGLang-a40 vs truth: 1 total, 0 clear
- WorkersAI vs truth: 5 total, 2 clear
  - CLEAR: pulse-10619114/urgency (margin 0.085)
  - CLEAR: pulse-11711631/urgency (margin 0.077)
- SGLang-a40 vs WorkersAI: 6 total, 2 clear
  - CLEAR: pulse-10619114/urgency (margin 0.085)
  - CLEAR: pulse-11711631/urgency (margin 0.077)

SGLang-a40 closer to truth on 50 questions, WorkersAI closer on 0.

Largest SGLang-a40-vs-WorkersAI gaps:
- pulse-10545020/urgency: SGLang-a40-WorkersAI 0.1505, SGLang-a40-truth 0.0070, WorkersAI-truth 0.1435, truth margin 0.001
- pulse-9710164/urgency: SGLang-a40-WorkersAI 0.1331, SGLang-a40-truth 0.0036, WorkersAI-truth 0.1316, truth margin 0.061
- pulse-11716347/urgency: SGLang-a40-WorkersAI 0.1132, SGLang-a40-truth 0.0024, WorkersAI-truth 0.1117, truth margin 0.112
- pulse-10702774/urgency: SGLang-a40-WorkersAI 0.1072, SGLang-a40-truth 0.0112, WorkersAI-truth 0.1184, truth margin 0.040
- pulse-9709747/urgency: SGLang-a40-WorkersAI 0.0942, SGLang-a40-truth 0.0040, WorkersAI-truth 0.0902, truth margin 0.075
- pulse-11711957/urgency: SGLang-a40-WorkersAI 0.0889, SGLang-a40-truth 0.0031, WorkersAI-truth 0.0917, truth margin 0.006
