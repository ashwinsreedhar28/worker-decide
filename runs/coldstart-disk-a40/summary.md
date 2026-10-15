## Cold start: disk-a40

10/10 cold starts, 0 resumed (process already up at submit), 0 excluded as suspected warm; gap after cold 0 s; GPU NVIDIA A40; snapshot not measured (v1); 2 distinct workers; idle slots before each submit: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

| segment (seconds unless noted) | median | p90 | min | max |
|---|---|---|---|---|
| cold: queue delay (Runpod delayTime) | 29.215 | 41.294 | 26.33 | 69.43 |
|   platform: submit -> process start (est.) | 5.177 | 15.156 | 3.7 | 26.533 |
|   worker: process start -> ready | 24.579 | 27.246 | 21.873 | 42.897 |
|     startup (Python, worker imports) | 1.735 | 1.78 | 1.72 | 1.87 |
|     snapshot fetch | 11.155 | 14.173 | 9.777 | 22.871 |
|     import torch / transformers / model code | 3.809 | 3.908 | 3.344 | 5.968 |
|     weight load | 7.63 | 7.999 | 6.877 | 12.188 |
| cold: first execution | 0.938 | 0.958 | 0.831 | 1.181 |
| warm: next execution, same worker | 0.323 | 0.372 | 0.272 | 0.449 |

Fetch source and bytes here come from the v1 metric (blob-directory growth), which saw ~130 KB for a 19 GB snapshot: they are not a transfer measurement.
