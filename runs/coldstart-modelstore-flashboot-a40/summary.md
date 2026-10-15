## Cold start: modelstore-flashboot-a40

2/10 cold starts, 8 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 0 s; GPU NVIDIA A40; snapshot model_store; 2 distinct workers; idle slots before each submit: [1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 20.072 | 20.355 | 19.79 | 20.355 |
|   platform: submit -> process start (est.) | 5.913 | 6.004 | 5.823 | 6.004 |
|   worker: process start -> ready | 14.159 | 14.351 | 13.967 | 14.351 |
|     startup (Python, worker imports) | 1.73 | 1.74 | 1.72 | 1.74 |
|     snapshot fetch | 0.002 | 0.002 | 0.001 | 0.002 |
|     import torch / transformers / model code | 3.606 | 3.634 | 3.578 | 3.634 |
|     weight load | 8.82 | 9.051 | 8.589 | 9.051 |
| fetch: GB written during it | 0.0 | 0.0 | 0.0 | 0.0 |
| cold: first execution | 0.883 | 0.899 | 0.867 | 0.899 |
| warm: next execution, same worker | 0.335 | 0.337 | 0.332 | 0.337 |

Resumed samples (8, 2 distinct workers): the job ran in a process started before the submit.

| resumed (seconds) | median | p90 | min | max |
|---|---|---|---|---|
| queue delay (Runpod delayTime) | 0.54 | 1.182 | 0.485 | 1.182 |
| first execution | 0.452 | 1.329 | 0.324 | 1.329 |
