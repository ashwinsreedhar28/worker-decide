## Shape probe: eager, pulse, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, CUDA_MODULE_LOADING=EAGER, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 8 | 2.5 | 0.3 | 0.1 | 169.8 | 4.4 | 0.1 | 177.3 | 0.0 |
| repeat | 8 | 2.4 | 0.3 | 0.1 | 169.6 | 4.3 | 0.1 | 177.1 | 0.0 |
| sibling | 8 | 2.5 | 0.3 | 0.1 | 169.8 | 4.5 | 0.1 | 177.3 | 0.0 |
| new - repeat | | +0.1 | +0.0 | +0.0 | +0.3 | +0.1 | +0.0 | +0.2 | |
