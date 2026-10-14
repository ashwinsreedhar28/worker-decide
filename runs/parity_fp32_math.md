# Parity: runs/fp32-math vs fixtures/clef-flash/float32

| metric | value |
|---|---|
| records | 86 (+1 skipped, --subset) |
| failed | 0 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0000 (bar 0.001) |
| mean of record means | 0.00000 (bar 0.0001) |
| mean of question max \|Δp\| (info) | 0.00000 |
| near-ties (not judged) | 3 |
| result | PASS |

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q3, run Q3, max |Δp| 0.0000
- pulse-10426369/urgency: oracle 0, run 0, max |Δp| 0.0000
- pulse-10545020/urgency: oracle 2, run 2, max |Δp| 0.0000

