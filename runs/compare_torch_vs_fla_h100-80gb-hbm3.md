179 questions compared (0 skipped: errored or missing in a run)

| per-question max \|Δp\| | max | p99 | p90 | median | mean |
|---|---|---|---|---|---|
| torch-gdn vs truth | 0.0211 | 0.0127 | 0.0050 | 0.00108 | 0.00202 |
| fla-gdn vs truth | 0.0189 | 0.0102 | 0.0051 | 0.00083 | 0.00177 |
| torch-gdn vs fla-gdn | 0.0229 | 0.0156 | 0.0063 | 0.00113 | 0.00235 |

Argmax flips (clear = truth top-2 margin > 0.04):
- torch-gdn vs truth: 4 total, 0 clear
- fla-gdn vs truth: 1 total, 0 clear
- torch-gdn vs fla-gdn: 3 total, 0 clear

torch-gdn closer to truth on 72 questions, fla-gdn closer on 95.

Largest torch-gdn-vs-fla-gdn gaps:
- invoice_extract/overdue: torch-gdn-fla-gdn 0.0229, torch-gdn-truth 0.0127, fla-gdn-truth 0.0102, truth margin 0.204
- pulse-9707528/urgency: torch-gdn-fla-gdn 0.0156, torch-gdn-truth 0.0113, fla-gdn-truth 0.0042, truth margin 0.041
- pulse-11711957/urgency: torch-gdn-fla-gdn 0.0114, torch-gdn-truth 0.0084, fla-gdn-truth 0.0030, truth margin 0.006
- pulse-10426369/urgency: torch-gdn-fla-gdn 0.0098, torch-gdn-truth 0.0094, fla-gdn-truth 0.0017, truth margin 0.000
- pulse-11716232/urgency: torch-gdn-fla-gdn 0.0098, torch-gdn-truth 0.0075, fla-gdn-truth 0.0023, truth margin 0.017
- near_dup_a/day: torch-gdn-fla-gdn 0.0093, torch-gdn-truth 0.0039, fla-gdn-truth 0.0062, truth margin 0.621
