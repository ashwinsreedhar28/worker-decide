## Correctness gate (vs fp32 truth)

# Parity: runs/sglang-order-a40 vs fixtures/clef-flash/float32-order

| metric | value |
|---|---|
| records | 5 |
| failed | 0 |
| request mismatches | 0 |
| span mismatches | 0 |
| max \|Δp\| (worst question) | 0.0106 (bar 0.025) |
| mean of record means | 0.00114 (bar 0.0025) |
| mean of question max \|Δp\| (info) | 0.00173 |
| near-ties (not judged) | 0 |
| result | PASS |


## Prompt length (stand-in for the span gate)

| token counts vs oracle | value |
|---|---|
| equal | 5 |
| different | 0 |
| no count (errored or absent) | 0 |
