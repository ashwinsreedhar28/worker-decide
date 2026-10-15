# Parity: runs/http-batched-a40 vs runs/http-single-a40

| metric | value |
|---|---|
| records | 87 |
| failed | 11 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0233 (bar 0.005) |
| mean of record means | 0.00114 (bar 0.001) |
| mean of question max \|Δp\| (info) | 0.00166 |
| near-ties (not judged) | 4 |
| result | FAIL |

## Failures

- **img_board2048_textonly** (request ok, span ok): boarding_before_1430: max |Δp| 0.0054 > 0.005
- **invoice_extract** (request ok, span ok): lines_sum_matches_total: max |Δp| 0.0145 > 0.005; overdue: max |Δp| 0.0233 > 0.005
- **list_state** (request ok, span ok): cause: max |Δp| 0.0054 > 0.005
- **moderation** (request ok, span ok): category: max |Δp| 0.0055 > 0.005
- **near_dup_c** (request ok, span ok): day: max |Δp| 0.0059 > 0.005
- **pulse-10702774** (request ok, span ok): urgency: max |Δp| 0.0100 > 0.005
- **pulse-10720478** (request ok, span ok): urgency: max |Δp| 0.0056 > 0.005
- **pulse-11617432** (request ok, span ok): urgency: max |Δp| 0.0051 > 0.005
- **pulse-11714938** (request ok, span ok): urgency: max |Δp| 0.0058 > 0.005
- **pulse-9707528** (request ok, span ok): urgency: max |Δp| 0.0082 > 0.005
- **unicode_options** (request ok, span ok): stimmung: max |Δp| 0.0059 > 0.005

## Near-ties (oracle top-2 margin ≤ near-tie; argmax not judged)

- img_chart_textonly/best_quarter: oracle Q4, run Q4, max |Δp| 0.0009
- pulse-10426369/urgency: oracle 0, run 0, max |Δp| 0.0012
- pulse-11711957/urgency: oracle 0, run 1, max |Δp| 0.0036
- triage_support/severity: oracle 2, run 2, max |Δp| 0.0041

**Mean-of-means 0.00114 exceeds bar 0.001.**
