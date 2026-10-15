## Shape probe: long, long_2k, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 8 | 5.2 | 0.5 | 0.1 | 458.9 | 5.7 | 0.1 | 470.6 | 0.0 |
| repeat | 8 | 5.2 | 0.5 | 0.1 | 459.3 | 5.5 | 0.1 | 471.2 | 0.0 |
| sibling | 8 | 5.2 | 0.5 | 0.1 | 459.5 | 5.5 | 0.1 | 471.0 | 0.0 |
| new - repeat | | +0.0 | +0.0 | +0.0 | -0.4 | +0.2 | +0.0 | -0.6 | |
