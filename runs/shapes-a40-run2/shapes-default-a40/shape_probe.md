## Shape probe: default, pulse, a40

Environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, TRITON_CACHE_DIR=/root/triton-cache-shapes, preferred_blas=_BlasBackend.Cublas

| kind | n | encode | collate | pre | backbone | head | post | total | device allocs |
|---|---|---|---|---|---|---|---|---|---|
| new | 7 | 2.3 | 0.3 | 0.1 | 169.3 | 5.1 | 0.1 | 177.6 | 0 |
| repeat | 7 | 2.2 | 0.3 | 0.1 | 169.8 | 4.5 | 0.1 | 177.4 | 0 |
| sibling | 8 | 2.2 | 0.3 | 0.1 | 169.9 | 4.9 | 0.1 | 177.2 | 0.0 |
| new - repeat | | +0.1 | +0.0 | +0.0 | -0.4 | +0.6 | -0.0 | +0.2 | |

Ops that grew most on a new token count (new pass minus repeat pass):

| op | Δ self CPU ms | Δ self CUDA ms | calls new / repeat |
|---|---|---|---|
| `Command Buffer Full` | +1.281 | +0.287 | 66 / 62 |
| `aten::mm` | -0.097 | +0.686 | 254 / 254 |
| `cudaStreamSynchronize` | +0.516 | +0.000 | 6 / 6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_25` | +0.000 | +0.357 | 113 / 113 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_12` | +0.000 | +0.227 | 32 / 32 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn` | +0.000 | +0.102 | 56 / 56 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_imp` | +0.000 | +0.042 | 133 / 133 |
| `void at::native::(anonymous namespace)::conv_depthwise2d_forward_kerne` | +0.000 | +0.030 | 24 / 24 |
| `aten::_conv_depthwise2d` | -0.002 | +0.030 | 24 / 24 |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | +0.000 | +0.011 | 24 / 24 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_imp` | +0.000 | +0.009 | 186 / 186 |
| `aten::sigmoid` | +0.006 | -0.000 | 33 / 33 |
| `aten::neg` | +0.006 | -0.000 | 40 / 40 |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` | +0.005 | +0.000 | 2 / 0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunc` | +0.000 | +0.005 | 87 / 87 |

Batch preview, lengths [843, 300, 2486]: batched 1413.37 ms vs 732.01 ms one at a time; outputs not comparable one to one
