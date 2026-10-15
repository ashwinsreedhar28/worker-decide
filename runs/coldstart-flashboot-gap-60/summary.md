## Cold start: flashboot-gap-60

1/3 cold starts, 2 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 60 s; GPU NVIDIA A40; snapshot model_store; 1 distinct workers; idle slots before each submit: [1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 24.072 | 24.072 | 24.072 | 24.072 |
|   platform: submit -> process start (est.) | 4.127 | 4.127 | 4.127 | 4.127 |
|   worker: process start -> ready | 19.945 | 19.945 | 19.945 | 19.945 |
|     startup (Python, worker imports) | 1.74 | 1.74 | 1.74 | 1.74 |
|     snapshot fetch | 0.001 | 0.001 | 0.001 | 0.001 |
|     import torch / transformers / model code | 3.88 | 3.88 | 3.88 | 3.88 |
|     weight load | 14.324 | 14.324 | 14.324 | 14.324 |
| fetch: GB written during it | 0.0 | 0.0 | 0.0 | 0.0 |
| cold: first execution | 0.936 | 0.936 | 0.936 | 0.936 |
| warm: next execution, same worker | 0.373 | 0.373 | 0.373 | 0.373 |

Resumed samples (2, 1 distinct workers): the job ran in a process started before the submit.

| resumed (seconds) | median | p90 | min | max |
|---|---|---|---|---|
| queue delay (Runpod delayTime) | 0.923 | 1.311 | 0.536 | 1.311 |
| first execution | 0.313 | 0.326 | 0.3 | 0.326 |
| process up before submit | 144.4 | 189.2 | 99.6 | 189.2 |
