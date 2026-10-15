## Cold start: disk-a40-v2

10/10 cold starts, 0 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 0 s; GPU NVIDIA A40; snapshot hf_download; 3 distinct workers; idle slots before each submit: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 33.77 | 37.86 | 27.393 | 42.252 |
|   platform: submit -> process start (est.) | 4.329 | 11.063 | 3.518 | 13.289 |
|   worker: process start -> ready | 27.785 | 33.363 | 23.776 | 33.593 |
|     startup (Python, worker imports) | 1.8 | 1.85 | 1.71 | 1.88 |
|     snapshot fetch | 13.655 | 16.207 | 10.38 | 19.296 |
|     import torch / transformers / model code | 4.034 | 5.31 | 3.767 | 5.484 |
|     weight load | 7.729 | 10.512 | 7.467 | 10.735 |
| fetch: GB written during it | 19.084 | 19.084 | 19.084480512 | 19.084480512 |
| fetch: GB/s | 1.4 | 1.796 | 0.9890381691542288 | 1.838581937572254 |
| cold: first execution | 0.977 | 0.989 | 0.825 | 1.101 |
| warm: next execution, same worker | 0.354 | 0.413 | 0.314 | 0.416 |
