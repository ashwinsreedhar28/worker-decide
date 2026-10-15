## Shape probe: long, long_2k, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 8 | 5.4 | 0.6 | 0.1 | 459.0 | 6.1 | 0.1 | 471.1 | 0.0 |
| repeat | 8 | 5.5 | 0.6 | 0.1 | 459.1 | 6.1 | 0.1 | 472.0 | 0.0 |
| sibling | 8 | 5.7 | 0.6 | 0.1 | 459.8 | 5.9 | 0.1 | 472.2 | 0.0 |
| new - repeat | | -0.2 | +0.0 | +0.0 | -0.1 | -0.1 | -0.0 | -0.8 | |
