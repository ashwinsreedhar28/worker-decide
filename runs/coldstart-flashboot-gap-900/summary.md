## Cold start: flashboot-gap-900

2/3 cold starts, 1 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 900 s; GPU NVIDIA A40; snapshot model_store; 1 distinct workers; idle slots before each submit: [1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 20.072 | 23.27 | 16.873 | 23.27 |
|   platform: submit -> process start (est.) | 3.617 | 4.253 | 2.98 | 4.253 |
|   worker: process start -> ready | 16.455 | 19.017 | 13.893 | 19.017 |
|     startup (Python, worker imports) | 1.735 | 1.74 | 1.73 | 1.74 |
|     snapshot fetch | 0.001 | 0.001 | 0.001 | 0.001 |
|     import torch / transformers / model code | 3.739 | 3.788 | 3.69 | 3.788 |
|     weight load | 10.98 | 13.596 | 8.364 | 13.596 |
| fetch: GB written during it | 0.0 | 0.0 | 0.0 | 0.0 |
| cold: first execution | 0.847 | 0.883 | 0.811 | 0.883 |
| warm: next execution, same worker | 0.363 | 0.39 | 0.336 | 0.39 |

Resumed samples (1, 1 distinct workers): the job ran in a process started before the submit.

| resumed (seconds) | median | p90 | min | max |
|---|---|---|---|---|
| queue delay (Runpod delayTime) | 0.527 | 0.527 | 0.527 | 0.527 |
| first execution | 0.403 | 0.403 | 0.403 | 0.403 |
| process up before submit | 938.0 | 938.0 | 938.0 | 938.0 |
