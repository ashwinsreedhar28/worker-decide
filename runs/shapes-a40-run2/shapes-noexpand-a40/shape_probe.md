## Shape probe: noexpand, pulse, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 8 | 2.3 | 0.3 | 0.1 | 169.7 | 4.8 | 0.1 | 177.2 | 0.0 |
| repeat | 8 | 2.4 | 0.3 | 0.1 | 169.6 | 4.4 | 0.1 | 177.0 | 0.0 |
| sibling | 8 | 2.4 | 0.3 | 0.1 | 170.3 | 4.6 | 0.1 | 178.0 | 0.0 |
| new - repeat | | -0.0 | +0.0 | +0.0 | +0.1 | +0.5 | +0.0 | +0.2 | |
