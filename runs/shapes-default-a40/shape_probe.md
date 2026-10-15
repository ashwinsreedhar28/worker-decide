## Shape probe: default, pulse, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 7 | 2.8 | 0.3 | 0.1 | 168.1 | 6.0 | 0.1 | 176.9 | 0 |
| repeat | 7 | 2.6 | 0.3 | 0.1 | 167.8 | 5.2 | 0.1 | 176.7 | 0 |
| sibling | 8 | 2.8 | 0.3 | 0.1 | 167.6 | 5.5 | 0.1 | 176.8 | 0.0 |
| new - repeat | | +0.2 | +0.0 | +0.0 | +0.2 | +0.9 | +0.0 | +0.2 | |

Ops that grew most on a new token count (new pass minus repeat pass):

| op | Δ self CPU ms | Δ self CUDA ms | calls new / repeat |
|---|---|---|---|
| `ChunkGatedDeltaRuleFunction` | +0.403 | +0.006 | 24 / 24 |
| `aten::mm` | +0.154 | +0.074 | 254 / 254 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_12` | +0.000 | +0.109 | 32 / 32 |
| `aten::copy_` | +0.021 | +0.050 | 436 / 436 |
| `aten::transpose` | +0.061 | +0.000 | 519 / 519 |
| `aten::addmm` | +0.063 | -0.002 | 42 / 42 |
| `aten::_conv_depthwise2d` | +0.034 | +0.026 | 24 / 24 |
| `aten::empty` | +0.059 | +0.000 | 467 / 467 |
| `aten::matmul` | +0.059 | +0.000 | 257 / 257 |
| `aten::clone` | +0.041 | +0.000 | 134 / 134 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_imp` | +0.000 | +0.040 | 133 / 133 |
| `aten::pow` | +0.033 | +0.003 | 81 / 81 |
| `aten::sigmoid` | +0.031 | +0.001 | 33 / 33 |
| `void at::native::(anonymous namespace)::conv_depthwise2d_forward_kerne` | +0.000 | +0.026 | 24 / 24 |
| `aten::reshape` | +0.023 | +0.000 | 532 / 532 |

Batch preview, lengths [843, 300, 2486]: batched 1398.08 ms vs 724.55 ms one at a time; outputs not comparable one to one
