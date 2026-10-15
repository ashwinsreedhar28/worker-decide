## Shape probe: eager, pulse, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, CUDA_MODULE_LOADING=EAGER, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 8 | 2.3 | 0.3 | 0.1 | 168.8 | 4.2 | 0.1 | 176.1 | 0.0 |
| repeat | 8 | 2.3 | 0.3 | 0.1 | 168.9 | 4.0 | 0.1 | 175.7 | 0.0 |
| sibling | 8 | 2.3 | 0.3 | 0.1 | 169.3 | 4.3 | 0.1 | 176.4 | 0.0 |
| new - repeat | | +0.0 | +0.0 | +0.0 | -0.1 | +0.1 | +0.0 | +0.4 | |
