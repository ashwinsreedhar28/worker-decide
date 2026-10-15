## Correctness gate (vs fp32 truth)

# Parity: runs/sglang-a40 vs /root/worker-decide/fixtures/clef-flash/float32

| metric | value |
|---|---|
| records | 87 |
| failed | 0 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0130 (bar 0.025) |
| mean of record means | 0.00137 (bar 0.0025) |
| mean of question max \|Δp\| (info) | 0.00181 |
| near-ties (not judged) | 19 |
| result | PASS |

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q3, run Q3, max |Δp| 0.0009
- img_receipt_textonly/total_bucket: oracle 10_to_50, run 10_to_50, max |Δp| 0.0009
- img_shapes_textonly/red_shape: oracle circle, run circle, max |Δp| 0.0009
- moderation/severity: oracle 2, run 2, max |Δp| 0.0032
- opts255/sku: oracle sku_120, run sku_120, max |Δp| 0.0020
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0019
- pulse-10426369/urgency: oracle 0, run 0, max |Δp| 0.0015
- pulse-10545020/urgency: oracle 2, run 0, max |Δp| 0.0070
- pulse-10702774/urgency: oracle 0, run 0, max |Δp| 0.0112
- pulse-10720478/urgency: oracle 1, run 1, max |Δp| 0.0013
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0019
- pulse-11711957/urgency: oracle 1, run 1, max |Δp| 0.0031
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0015
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0032
- pulse-9707528/urgency: oracle 0, run 0, max |Δp| 0.0021
- pulse-9707552/urgency: oracle 1, run 1, max |Δp| 0.0012
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0027
- score10/frustration: oracle 6, run 6, max |Δp| 0.0024
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0022


Known API differences (SKIP, not judged): empty_instructions

## Prompt length (stand-in for the span gate)

| token counts vs oracle | value |
|---|---|
| equal | 85 |
| different | 0 |
| no count (errored or absent) | 1 |

--- informational: vs runs/http-a40 (not gated) ---

# Parity: runs/sglang-a40 vs runs/http-a40

| metric | value |
|---|---|
| records | 87 |
| failed | 1 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0160 (bar 0.025) |
| mean of record means | 0.00160 (bar 0.0025) |
| mean of question max \|Δp\| (info) | 0.00215 |
| near-ties (not judged) | 19 |
| result | FAIL |

## Failures

- **empty_instructions** (request ok, span n/a): run errored: http 422: [{"type": "value_error", "loc": ["body", "questions", "needs_shipping_

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q4, run Q3, max |Δp| 0.0022
- img_receipt_textonly/total_bucket: oracle 10_to_50, run 10_to_50, max |Δp| 0.0052
- img_shapes_textonly/red_shape: oracle circle, run circle, max |Δp| 0.0030
- moderation/severity: oracle 2, run 2, max |Δp| 0.0028
- opts255/sku: oracle sku_120, run sku_120, max |Δp| 0.0012
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0018
- pulse-10426369/urgency: oracle 1, run 0, max |Δp| 0.0079
- pulse-10545020/urgency: oracle 2, run 0, max |Δp| 0.0068
- pulse-10702774/urgency: oracle 0, run 0, max |Δp| 0.0088
- pulse-10720478/urgency: oracle 1, run 1, max |Δp| 0.0008
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0027
- pulse-11711957/urgency: oracle 1, run 1, max |Δp| 0.0028
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0012
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0069
- pulse-9707528/urgency: oracle 0, run 0, max |Δp| 0.0040
- pulse-9707552/urgency: oracle 1, run 1, max |Δp| 0.0026
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0035
- score10/frustration: oracle 6, run 6, max |Δp| 0.0030
- triage_support/severity: oracle 3, run 3, max |Δp| 0.0022

