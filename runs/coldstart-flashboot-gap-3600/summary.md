## Cold start: flashboot-gap-3600

3/3 cold starts, 0 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 3600 s; GPU NVIDIA A40; snapshot model_store; 1 distinct workers; idle slots before each submit: [1, 1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 17.802 | 17.955 | 16.387 | 17.955 |
|   platform: submit -> process start (est.) | 3.784 | 4.018 | 3.199 | 4.018 |
|   worker: process start -> ready | 13.937 | 14.018 | 13.188 | 14.018 |
|     startup (Python, worker imports) | 1.75 | 1.81 | 1.73 | 1.81 |
|     snapshot fetch | 0.001 | 0.001 | 0.001 | 0.001 |
|     import torch / transformers / model code | 3.81 | 3.935 | 3.713 | 3.935 |
|     weight load | 8.251 | 8.494 | 7.647 | 8.494 |
| fetch: GB written during it | 0.0 | 0.0 | 0.0 | 0.0 |
| cold: first execution | 0.946 | 0.989 | 0.868 | 0.989 |
| warm: next execution, same worker | 0.406 | 0.424 | 0.377 | 0.424 |
