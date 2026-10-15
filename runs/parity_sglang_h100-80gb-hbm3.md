## Correctness gate (vs fp32 truth)

# Parity: runs/sglang-h100-80gb-hbm3 vs /root/worker-decide/fixtures/clef-flash/float32

| metric | value |
|---|---|
| records | 87 |
| failed | 0 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0159 (bar 0.025) |
| mean of record means | 0.00138 (bar 0.0025) |
| mean of question max \|Δp\| (info) | 0.00182 |
| near-ties (not judged) | 19 |
| result | PASS |

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q3, run Q4, max |Δp| 0.0011
- img_receipt_textonly/total_bucket: oracle 10_to_50, run 10_to_50, max |Δp| 0.0012
- img_shapes_textonly/red_shape: oracle circle, run circle, max |Δp| 0.0017
- moderation/severity: oracle 2, run 2, max |Δp| 0.0088
- opts255/sku: oracle sku_120, run sku_120, max |Δp| 0.0015
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0036
- pulse-10426369/urgency: oracle 0, run 0, max |Δp| 0.0012
- pulse-10545020/urgency: oracle 2, run 2, max |Δp| 0.0061
- pulse-10702774/urgency: oracle 0, run 0, max |Δp| 0.0082
- pulse-10720478/urgency: oracle 1, run 1, max |Δp| 0.0047
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0024
- pulse-11711957/urgency: oracle 1, run 1, max |Δp| 0.0036
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0046
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0027
- pulse-9707528/urgency: oracle 0, run 0, max |Δp| 0.0042
- pulse-9707552/urgency: oracle 1, run 1, max |Δp| 0.0005
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0019
- score10/frustration: oracle 6, run 6, max |Δp| 0.0035
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0013


Known API differences (SKIP, not judged): empty_instructions

## Prompt length (stand-in for the span gate)

| token counts vs oracle | value |
|---|---|
| equal | 85 |
| different | 0 |
| no count (errored or absent) | 1 |

--- informational: vs fixtures/clef-flash/bfloat16 (not gated) ---

# Parity: runs/sglang-h100-80gb-hbm3 vs fixtures/clef-flash/bfloat16

| metric | value |
|---|---|
| records | 87 |
| failed | 2 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0286 (bar 0.025) |
| mean of record means | 0.00189 (bar 0.0025) |
| mean of question max \|Δp\| (info) | 0.00250 |
| near-ties (not judged) | 19 |
| result | FAIL |

## Failures

- **empty_instructions** (request ok, span n/a): run errored: http 422: [{"type": "value_error", "loc": ["body", "questions", "needs_shipping_
- **invoice_extract** (request ok, span n/a): overdue: max |Δp| 0.0286 > 0.025

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q4, run Q4, max |Δp| 0.0012
- img_receipt_textonly/total_bucket: oracle 10_to_50, run 10_to_50, max |Δp| 0.0024
- img_shapes_textonly/red_shape: oracle circle, run circle, max |Δp| 0.0018
- moderation/severity: oracle 2, run 2, max |Δp| 0.0052
- opts255/sku: oracle sku_120, run sku_120, max |Δp| 0.0017
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0086
- pulse-10426369/urgency: oracle 1, run 0, max |Δp| 0.0102
- pulse-10545020/urgency: oracle 0, run 2, max |Δp| 0.0084
- pulse-10702774/urgency: oracle 0, run 0, max |Δp| 0.0041
- pulse-10720478/urgency: oracle 1, run 1, max |Δp| 0.0084
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0042
- pulse-11711957/urgency: oracle 0, run 1, max |Δp| 0.0078
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0066
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0101
- pulse-9707528/urgency: oracle 0, run 0, max |Δp| 0.0149
- pulse-9707552/urgency: oracle 1, run 1, max |Δp| 0.0015
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0084
- score10/frustration: oracle 6, run 6, max |Δp| 0.0033
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0022

