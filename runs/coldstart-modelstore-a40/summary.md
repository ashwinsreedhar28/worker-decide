## Cold start: modelstore-a40

10/10 cold starts, 0 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 0 s; GPU NVIDIA A40; snapshot model_store, not model_store (v1: download vs cache unknown); 3 distinct workers; idle slots before each submit: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 25.517 | 42.017 | 18.278 | 73.834 |
|   platform: submit -> process start (est.) | 5.737 | 27.675 | 3.297 | 57.375 |
|   worker: process start -> ready | 16.166 | 22.173 | 14.046 | 25.056 |
|     startup (Python, worker imports) | 1.77 | 2 | 1.72 | 2.11 |
|     snapshot fetch | 0.0 | 10.001 | 0 | 12.225 |
|     import torch / transformers / model code | 3.87 | 4.277 | 3.508 | 5.441 |
|     weight load | 9.647 | 10.446 | 6.84 | 11.852 |
| cold: first execution | 1.01 | 1.077 | 0.847 | 1.097 |
| warm: next execution, same worker | 0.429 | 0.453 | 0.343 | 0.475 |

Fetch source and bytes here come from the v1 metric (blob-directory growth), which saw ~130 KB for a 19 GB snapshot: they are not a transfer measurement.
