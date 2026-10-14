# Parity: runs/http-bfloat16 vs fixtures/clef-flash/bfloat16

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


--- informational: vs fixtures/clef-flash/float32 (not gated) ---

# Parity: runs/http-bfloat16 vs fixtures/clef-flash/float32

| metric | value |
|---|---|
| records | 87 |
| failed | 1 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0211 (bar 0.02) |
| mean of record means | 0.00155 (bar 0.002) |
| mean of question max \|Δp\| (info) | 0.00202 |
| near-ties (not judged) | 15 |
| result | FAIL |

## Failures

- **img_sign448** (request ok, span ok): word: max |Δp| 0.0211 > 0.02

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q3, run Q4, max |Δp| 0.0011
- img_receipt_textonly/total_bucket: oracle 10_to_50, run 10_to_50, max |Δp| 0.0021
- moderation/severity: oracle 2, run 2, max |Δp| 0.0036
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0050
- pulse-10426369/urgency: oracle 0, run 1, max |Δp| 0.0094
- pulse-10545020/urgency: oracle 2, run 0, max |Δp| 0.0023
- pulse-10702774/urgency: oracle 0, run 0, max |Δp| 0.0042
- pulse-10720478/urgency: oracle 1, run 1, max |Δp| 0.0037
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0018
- pulse-11711957/urgency: oracle 1, run 0, max |Δp| 0.0084
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0028
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0075
- pulse-9707552/urgency: oracle 1, run 1, max |Δp| 0.0020
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0067
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0017

