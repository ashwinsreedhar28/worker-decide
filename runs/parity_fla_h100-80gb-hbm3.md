## Correctness gate (any GPU)

# Parity: runs/http-fla-h100-80gb-hbm3 vs /root/worker-decide/fixtures/clef-flash/float32

| metric | value |
|---|---|
| records | 87 |
| failed | 0 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0189 (bar 0.025) |
| mean of record means | 0.00137 (bar 0.0025) |
| mean of question max \|Δp\| (info) | 0.00177 |
| near-ties (not judged) | 19 |
| result | PASS |

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q3, run Q4, max |Δp| 0.0019
- img_receipt_textonly/total_bucket: oracle 10_to_50, run 10_to_50, max |Δp| 0.0011
- img_shapes_textonly/red_shape: oracle circle, run circle, max |Δp| 0.0020
- moderation/severity: oracle 2, run 2, max |Δp| 0.0021
- opts255/sku: oracle sku_120, run sku_120, max |Δp| 0.0026
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0026
- pulse-10426369/urgency: oracle 0, run 0, max |Δp| 0.0017
- pulse-10545020/urgency: oracle 2, run 2, max |Δp| 0.0047
- pulse-10702774/urgency: oracle 0, run 0, max |Δp| 0.0051
- pulse-10720478/urgency: oracle 1, run 1, max |Δp| 0.0029
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0031
- pulse-11711957/urgency: oracle 1, run 1, max |Δp| 0.0030
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0033
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0023
- pulse-9707528/urgency: oracle 0, run 0, max |Δp| 0.0042
- pulse-9707552/urgency: oracle 1, run 1, max |Δp| 0.0062
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0057
- score10/frustration: oracle 6, run 6, max |Δp| 0.0013
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0040


--- informational: vs fixtures/clef-flash/bfloat16 (not gated) ---

# Parity: runs/http-fla-h100-80gb-hbm3 vs fixtures/clef-flash/bfloat16

| metric | value |
|---|---|
| records | 87 |
| failed | 1 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0229 (bar 0.02) |
| mean of record means | 0.00174 (bar 0.002) |
| mean of question max \|Δp\| (info) | 0.00235 |
| near-ties (not judged) | 13 |
| result | FAIL |

## Failures

- **invoice_extract** (request ok, span ok): overdue: max |Δp| 0.0229 > 0.02

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q4, run Q4, max |Δp| 0.0021
- moderation/severity: oracle 2, run 2, max |Δp| 0.0058
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0034
- pulse-10426369/urgency: oracle 1, run 0, max |Δp| 0.0098
- pulse-10545020/urgency: oracle 0, run 2, max |Δp| 0.0071
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0049
- pulse-11711957/urgency: oracle 0, run 1, max |Δp| 0.0114
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0038
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0098
- pulse-9707528/urgency: oracle 0, run 0, max |Δp| 0.0156
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0045
- score10/frustration: oracle 6, run 6, max |Δp| 0.0021
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0051

