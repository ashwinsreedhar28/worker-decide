# Parity: runs/bfloat16 vs fixtures/clef-flash/float32

| metric | value |
|---|---|
| records | 87 |
| failed | 1 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0211 (bar 0.02) |
| mean of record means | 0.00155 (bar 0.002) |
| near-ties (not judged) | 7 |
| result | FAIL |

## Failures

- **img_sign448** (span ok): word: max |Δp| 0.0211 > 0.02

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q3, run Q4, max |Δp| 0.0011
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0050
- pulse-10426369/urgency: oracle 0, run 1, max |Δp| 0.0094
- pulse-10545020/urgency: oracle 2, run 0, max |Δp| 0.0023
- pulse-11711957/urgency: oracle 1, run 0, max |Δp| 0.0084
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0075
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0017

