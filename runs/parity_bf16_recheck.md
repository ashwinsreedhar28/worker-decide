# Parity: runs/bf16-recheck vs fixtures/clef-flash/bfloat16

| metric | value |
|---|---|
| records | 87 |
| failed | 0 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0000 (bar 0.005) |
| mean of record means | 0.00000 (bar 0.001) |
| mean of question max \|Δp\| (info) | 0.00000 |
| near-ties (not judged) | 5 |
| result | PASS |

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q4, run Q4, max |Δp| 0.0000
- pulse-10545020/urgency: oracle 0, run 0, max |Δp| 0.0000
- pulse-11711957/urgency: oracle 0, run 0, max |Δp| 0.0000
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0000
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0000

