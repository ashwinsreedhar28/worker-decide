## Cold start: flashboot-gap-1800

3/3 cold starts, 0 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 1800 s; GPU NVIDIA A40; snapshot model_store; 1 distinct workers; idle slots before each submit: [1, 1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 16.943 | 19.046 | 15.965 | 19.046 |
|   platform: submit -> process start (est.) | 3.109 | 5.895 | 3.107 | 5.895 |
|   worker: process start -> ready | 13.151 | 13.836 | 12.856 | 13.836 |
|     startup (Python, worker imports) | 1.75 | 1.77 | 1.73 | 1.77 |
|     snapshot fetch | 0.001 | 0.001 | 0.001 | 0.001 |
|     import torch / transformers / model code | 3.595 | 3.834 | 3.334 | 3.834 |
|     weight load | 7.805 | 8.231 | 7.791 | 8.231 |
| fetch: GB written during it | 0.0 | 0.0 | 0.0 | 0.0 |
| cold: first execution | 0.77 | 0.841 | 0.684 | 0.841 |
| warm: next execution, same worker | 0.315 | 0.381 | 0.278 | 0.381 |
