## Shape probe: default, pulse, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 7 | 2.2 | 0.3 | 0.1 | 168.5 | 4.8 | 0.1 | 175.9 | 0 |
| repeat | 7 | 2.2 | 0.3 | 0.1 | 168.2 | 4.6 | 0.1 | 175.5 | 0 |
| sibling | 8 | 2.3 | 0.3 | 0.1 | 168.3 | 4.8 | 0.1 | 176.1 | 0.0 |
| new - repeat | | -0.0 | +0.0 | +0.0 | +0.2 | +0.2 | +0.0 | +0.4 | |

Ops that grew most on a new token count (new pass minus repeat pass):

| op | Δ self CPU ms | Δ self CUDA ms | calls new / repeat |
|---|---|---|---|
| `aten::mm` | +0.000 | +0.556 | 254 / 254 |
| `cudaStreamSynchronize` | +0.502 | +0.000 | 6 / 6 |
| `Command Buffer Full` | +0.216 | +0.098 | 46 / 46 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_25` | +0.000 | +0.283 | 113 / 113 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_12` | +0.000 | +0.194 | 32 / 32 |
| `ChunkGatedDeltaRuleFunction` | +0.118 | +0.027 | 24 / 24 |
| `Activity Buffer Request` | +0.107 | +0.000 | 1 / 1 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn` | +0.000 | +0.079 | 56 / 56 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_imp` | +0.000 | +0.055 | 133 / 133 |
| `LayerNormGatedFunction` | +0.051 | +0.001 | 24 / 24 |
| `aten::mean` | +0.047 | -0.001 | 92 / 92 |
| `void at::native::(anonymous namespace)::conv_depthwise2d_forward_kerne` | +0.000 | +0.041 | 24 / 24 |
| `aten::_conv_depthwise2d` | -0.004 | +0.041 | 24 / 24 |
| `aten::copy_` | -0.020 | +0.048 | 436 / 436 |
| `cuLaunchKernelEx` | +0.019 | +0.000 | 192 / 192 |

Batch preview, lengths [843, 300, 2486]: batched 1400.42 ms vs 730.56 ms one at a time; outputs not comparable one to one
