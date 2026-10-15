## Cold start: flashboot-gap-300

1/3 cold starts, 2 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 300 s; GPU NVIDIA A40; snapshot model_store; 1 distinct workers; idle slots before each submit: [1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 18.254 | 18.254 | 18.254 | 18.254 |
|   platform: submit -> process start (est.) | 4.993 | 4.993 | 4.993 | 4.993 |
|   worker: process start -> ready | 13.261 | 13.261 | 13.261 | 13.261 |
|     startup (Python, worker imports) | 1.74 | 1.74 | 1.74 | 1.74 |
|     snapshot fetch | 0.001 | 0.001 | 0.001 | 0.001 |
|     import torch / transformers / model code | 4.024 | 4.024 | 4.024 | 4.024 |
|     weight load | 7.496 | 7.496 | 7.496 | 7.496 |
| fetch: GB written during it | 0.0 | 0.0 | 0.0 | 0.0 |
| cold: first execution | 0.905 | 0.905 | 0.905 | 0.905 |
| warm: next execution, same worker | 0.374 | 0.374 | 0.374 | 0.374 |

Resumed samples (2, 1 distinct workers): the job ran in a process started before the submit.

| resumed (seconds) | median | p90 | min | max |
|---|---|---|---|---|
| queue delay (Runpod delayTime) | 0.542 | 0.542 | 0.541 | 0.542 |
| first execution | 0.378 | 0.388 | 0.367 | 0.388 |
| process up before submit | 422.9 | 497.4 | 348.4 | 497.4 |
