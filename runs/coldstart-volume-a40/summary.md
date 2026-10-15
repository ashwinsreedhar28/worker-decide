## Cold start: volume-a40

10/10 cold starts, 0 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 0 s; GPU NVIDIA A40; snapshot hf_cache, hf_download; 3 distinct workers; idle slots before each submit: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 27.617 | 48.67 | 17.797 | 74.657 |
|   platform: submit -> process start (est.) | 5.34 | 11.069 | 2.784 | 11.794 |
|   worker: process start -> ready | 19.878 | 44.946 | 14.99 | 69.875 |
|     startup (Python, worker imports) | 1.79 | 1.84 | 1.71 | 1.87 |
|     snapshot fetch | 0.315 | 1.216 | 0.063 | 17.878 |
|     import torch / transformers / model code | 3.974 | 4.615 | 3.297 | 4.723 |
|     weight load | 13.354 | 38.602 | 8.694 | 45.492 |
| fetch: GB written during it | 0.0 | 0.0 | 0.0 | 19.0839467 |
| fetch: GB/s | 1.067 | 1.067 | 1.0674542286609239 | 1.0674542286609239 |
| cold: first execution | 1.008 | 1.137 | 0.832 | 1.206 |
| warm: next execution, same worker | 0.374 | 0.466 | 0.316 | 0.558 |
