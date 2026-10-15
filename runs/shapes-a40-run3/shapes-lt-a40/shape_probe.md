## Shape probe: lt, pulse, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, TORCH_BLAS_PREFER_CUBLASLT=1, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublaslt

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 8 | 2.3 | 0.3 | 0.1 | 162.8 | 4.9 | 0.1 | 170.8 | 0.0 |
| repeat | 8 | 2.4 | 0.3 | 0.1 | 162.8 | 4.8 | 0.1 | 170.6 | 0.0 |
| sibling | 8 | 2.4 | 0.3 | 0.1 | 163.2 | 4.7 | 0.1 | 171.6 | 0.0 |
| new - repeat | | -0.1 | -0.0 | +0.0 | +0.0 | +0.1 | +0.0 | +0.2 | |
