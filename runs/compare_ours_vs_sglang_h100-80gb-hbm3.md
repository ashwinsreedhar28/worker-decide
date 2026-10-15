178 questions compared (1 skipped: errored or missing in a run)

| per-question max \|Δp\| | max | p99 | p90 | median | mean |
|---|---|---|---|---|---|
| ours-h100-80gb-hbm3 vs truth | 0.0211 | 0.0127 | 0.0050 | 0.00108 | 0.00202 |
| SGLang-h100-80gb-hbm3 vs truth | 0.0159 | 0.0141 | 0.0047 | 0.00091 | 0.00182 |
| ours-h100-80gb-hbm3 vs SGLang-h100-80gb-hbm3 | 0.0286 | 0.0222 | 0.0067 | 0.00103 | 0.00250 |

Argmax flips (clear = truth top-2 margin > 0.04):
- ours-h100-80gb-hbm3 vs truth: 4 total, 0 clear
- SGLang-h100-80gb-hbm3 vs truth: 1 total, 0 clear
- ours-h100-80gb-hbm3 vs SGLang-h100-80gb-hbm3: 3 total, 0 clear

ours-h100-80gb-hbm3 closer to truth on 70 questions, SGLang-h100-80gb-hbm3 closer on 108.

Largest ours-h100-80gb-hbm3-vs-SGLang-h100-80gb-hbm3 gaps:
- invoice_extract/overdue: ours-h100-80gb-hbm3-SGLang-h100-80gb-hbm3 0.0286, ours-h100-80gb-hbm3-truth 0.0127, SGLang-h100-80gb-hbm3-truth 0.0159, truth margin 0.204
- img_sign448/word: ours-h100-80gb-hbm3-SGLang-h100-80gb-hbm3 0.0222, ours-h100-80gb-hbm3-truth 0.0211, SGLang-h100-80gb-hbm3-truth 0.0141, truth margin 0.292
- pulse-10901596/urgency: ours-h100-80gb-hbm3-SGLang-h100-80gb-hbm3 0.0164, ours-h100-80gb-hbm3-truth 0.0084, SGLang-h100-80gb-hbm3-truth 0.0080, truth margin 0.275
- pulse-9707528/urgency: ours-h100-80gb-hbm3-SGLang-h100-80gb-hbm3 0.0149, ours-h100-80gb-hbm3-truth 0.0113, SGLang-h100-80gb-hbm3-truth 0.0042, truth margin 0.041
- img_board2048/boarding_before_1430: ours-h100-80gb-hbm3-SGLang-h100-80gb-hbm3 0.0119, ours-h100-80gb-hbm3-truth 0.0026, SGLang-h100-80gb-hbm3-truth 0.0093, truth margin 0.748
- pulse-10426369/urgency: ours-h100-80gb-hbm3-SGLang-h100-80gb-hbm3 0.0102, ours-h100-80gb-hbm3-truth 0.0094, SGLang-h100-80gb-hbm3-truth 0.0012, truth margin 0.000
