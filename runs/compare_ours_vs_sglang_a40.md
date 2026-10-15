178 questions compared (1 skipped: errored or missing in a run)

| per-question max \|Δp\| | max | p99 | p90 | median | mean |
|---|---|---|---|---|---|
| ours-A40 vs truth | 0.0163 | 0.0115 | 0.0040 | 0.00100 | 0.00172 |
| SGLang-a40 vs truth | 0.0130 | 0.0120 | 0.0044 | 0.00093 | 0.00181 |
| ours-A40 vs SGLang-a40 | 0.0160 | 0.0118 | 0.0057 | 0.00118 | 0.00215 |

Argmax flips (clear = truth top-2 margin > 0.04):
- ours-A40 vs truth: 2 total, 0 clear
- SGLang-a40 vs truth: 1 total, 0 clear
- ours-A40 vs SGLang-a40: 3 total, 0 clear

ours-A40 closer to truth on 90 questions, SGLang-a40 closer on 88.

Largest ours-A40-vs-SGLang-a40 gaps:
- img_board2048/gate: ours-A40-SGLang-a40 0.0160, ours-A40-truth 0.0053, SGLang-a40-truth 0.0107, truth margin 0.878
- img_sign448/word: ours-A40-SGLang-a40 0.0118, ours-A40-truth 0.0115, SGLang-a40-truth 0.0120, truth margin 0.292
- pulse-9707413/urgency: ours-A40-SGLang-a40 0.0102, ours-A40-truth 0.0061, SGLang-a40-truth 0.0041, truth margin 0.263
- pulse-10702774/urgency: ours-A40-SGLang-a40 0.0088, ours-A40-truth 0.0025, SGLang-a40-truth 0.0112, truth margin 0.040
- pulse-11711123/urgency: ours-A40-SGLang-a40 0.0079, ours-A40-truth 0.0045, SGLang-a40-truth 0.0034, truth margin 0.119
- pulse-10426369/urgency: ours-A40-SGLang-a40 0.0079, ours-A40-truth 0.0072, SGLang-a40-truth 0.0015, truth margin 0.000
