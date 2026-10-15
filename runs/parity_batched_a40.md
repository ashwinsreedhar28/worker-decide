## Correctness gate (any GPU)

# Parity: runs/http-batched-a40 vs /root/worker-decide/fixtures/clef-flash/float32

| metric | value |
|---|---|
| records | 87 |
| failed | 0 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0154 (bar 0.025) |
| mean of record means | 0.00121 (bar 0.0025) |
| mean of question max \|Δp\| (info) | 0.00165 |
| near-ties (not judged) | 19 |
| result | PASS |

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q3, run Q4, max |Δp| 0.0009
- img_receipt_textonly/total_bucket: oracle 10_to_50, run 10_to_50, max |Δp| 0.0014
- img_shapes_textonly/red_shape: oracle circle, run circle, max |Δp| 0.0005
- moderation/severity: oracle 2, run 2, max |Δp| 0.0011
- opts255/sku: oracle sku_120, run sku_120, max |Δp| 0.0025
- pulse-10221259/urgency: oracle 4, run 4, max |Δp| 0.0045
- pulse-10426369/urgency: oracle 0, run 0, max |Δp| 0.0009
- pulse-10545020/urgency: oracle 2, run 2, max |Δp| 0.0071
- pulse-10702774/urgency: oracle 0, run 0, max |Δp| 0.0088
- pulse-10720478/urgency: oracle 1, run 1, max |Δp| 0.0022
- pulse-10729536/urgency: oracle 2, run 2, max |Δp| 0.0019
- pulse-11711957/urgency: oracle 1, run 1, max |Δp| 0.0022
- pulse-11714938/urgency: oracle 1, run 1, max |Δp| 0.0010
- pulse-11716232/urgency: oracle 4, run 4, max |Δp| 0.0046
- pulse-9707528/urgency: oracle 0, run 0, max |Δp| 0.0035
- pulse-9707552/urgency: oracle 1, run 1, max |Δp| 0.0008
- pulse-9853056/urgency: oracle 3, run 3, max |Δp| 0.0066
- score10/frustration: oracle 6, run 6, max |Δp| 0.0025
- triage_support/severity: oracle 3, run 2, max |Δp| 0.0024

