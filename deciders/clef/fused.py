"""Optimization tier 2: fuse the glue around Qwen3.5's GEMMs.

Tier 0's traces put a quarter of the A40's kernel time (39 of 167 ms on an 843-token request) and
a third of the H100's into small torch kernels around the matrix multiplies: RMSNorm as about six
kernels per call, SiLU and the multiply as two, the Gated DeltaNet gate math, transposes and copies
around the short convolution, and q/k copied to twice their head count before the recurrence.
SGLang fuses these. Each fusion here replaces one of them with a fla-core Triton kernel (fla is
already the tier 0 dependency), on the loaded model's own module instances, after loading:

    rmsnorm   Qwen3_5RMSNorm (64 decoder norms + the final norm + attention q/k norms): one fused
              kernel, with (1 + weight) precomputed in fp32 (the module computes in fp32 too)
    mlp       Qwen3_5MLP: gate_proj and up_proj as one GEMM over a concatenated weight (the two
              original weights become views of it, so no extra memory), SiLU·mul as one kernel
    gdn       Qwen3_5GatedDeltaNet: q and k go to fla's recurrence at their own 16 heads (fla
              supports more value heads than key heads, mapping value head i to key head
              i // (HV / H), which is what repeat_interleave builds), and the decay
              -exp(A_log) · softplus(a + dt_bias) is computed inside the kernel
    conv      the GDN short convolution as fla's causal conv1d on [B, T, C] with SiLU fused,
              instead of transpose → torch depthwise conv → slice → SiLU → transpose
    nocache   use_cache=False in the model's configs: transformers otherwise builds a cache on
              every forward (each GDN layer's conv and recurrent state, each attention layer's
              K/V) that a single-pass classifier never reads
    cublaslt  torch's GEMMs through cuBLASLt (3.1–3.4% faster on the A40 backbone, Oct 15)

All of it is off unless FUSED names it ("all" for everything); each fusion goes through the
correctness gate on a GPU before it can become a default. `gdn` and `conv` replace the GDN layer's
forward with `gdn_forward` below, a copy of transformers 5.10.2's for the no-cache prefill path
only; a call with a cache or packed sequences goes to the original forward.

`Kernels` is the seam the CPU tests use: they substitute exact torch references for fla's kernels
and check that the patched model computes what the unpatched one does.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

FUSIONS = ("rmsnorm", "mlp", "gdn", "conv", "nocache", "cublaslt")


def parse(spec: str | None) -> tuple[str, ...]:
    """FUSED's value: "" or None → nothing, "all" → every fusion, else a comma list."""
    if not spec or not spec.strip():
        return ()
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if names == ["all"]:
        return FUSIONS
    unknown = [n for n in names if n not in FUSIONS]
    if unknown:
        raise ValueError(f"unknown FUSED value(s) {unknown}; known: {', '.join(FUSIONS)} or all")
    return tuple(n for n in FUSIONS if n in names)  # canonical order, no duplicates


@dataclass
class Kernels:
    rms_norm: Callable[..., Any]  # (x, weight, eps) -> normalized x * weight, in x's dtype
    swiglu: Callable[..., Any]  # (gate, up) -> silu(gate) * up
    causal_conv1d: Callable[..., Any]  # (x [B,T,D], weight [D,W], bias, activation) -> [B,T,D]
    chunk_gated_delta_rule: Callable[..., Any]  # fla's signature


def fla_kernels() -> Kernels:
    from fla.modules.activations import swiglu
    from fla.modules.conv import causal_conv1d
    from fla.modules.layernorm import rms_norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    def conv(x, weight, bias, activation):
        y, _ = causal_conv1d(x=x, weight=weight, bias=bias, activation=activation, backend="triton")
        return y

    return Kernels(
        rms_norm=lambda x, w, eps: rms_norm(x, w, None, eps=eps),
        swiglu=swiglu,
        causal_conv1d=conv,
        chunk_gated_delta_rule=chunk_gated_delta_rule,
    )


def _class(module: Any) -> str:
    return type(module).__name__


def patch_rmsnorm(module: Any, k: Kernels, torch: Any) -> None:
    # Qwen3_5RMSNorm: _norm(x.float()) * (1 + weight.float()), cast back to x's dtype.
    module._fused_weight = (1.0 + module.weight.detach().float()).contiguous()

    def forward(self, x):
        return k.rms_norm(x, self._fused_weight, self.eps)

    module._fused_orig_forward = module.forward
    module.forward = types.MethodType(forward, module)


def patch_mlp(module: Any, k: Kernels, torch: Any) -> None:
    F = torch.nn.functional
    n = module.intermediate_size
    merged = torch.cat([module.gate_proj.weight.data, module.up_proj.weight.data], dim=0)
    module.gate_proj.weight.data = merged[:n]  # views: the merge costs no extra memory
    module.up_proj.weight.data = merged[n:]
    module._fused_gate_up = merged

    def forward(self, x):
        gate, up = F.linear(x, self._fused_gate_up).split(self.intermediate_size, dim=-1)
        return self.down_proj(k.swiglu(gate, up))

    module._fused_orig_forward = module.forward
    module.forward = types.MethodType(forward, module)


def gdn_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """transformers 5.10.2 Qwen3_5GatedDeltaNet.forward for a prefill (no cache, or a cache with
    no state for this layer yet, which transformers creates by default), with this layer's
    fusions (self._fused: {"gdn", "conv"} subset). A cache is filled exactly as the original
    fills it (conv state from the pre-convolution input, the recurrence's final state)."""
    continuing = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
    if continuing or kwargs.get("cu_seq_lens_q") is not None:
        return self._fused_orig_forward(
            hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs
        )
    torch, k, modeling = self._fused_torch, self._fused_kernels, self._fused_modeling
    F = torch.nn.functional
    hidden_states = modeling.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    if cache_params is not None:  # as the original: the last conv_kernel_size inputs, [B, C, W]
        x = mixed_qkv.transpose(1, 2)
        cache_params.update_conv_state(F.pad(x, (self.conv_kernel_size - x.shape[-1], 0)),
                                       self.layer_idx)  # fmt: skip

    if "conv" in self._fused:
        mixed_qkv = k.causal_conv1d(
            mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
        )
    elif self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv.transpose(1, 2),
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=kwargs.get("seq_idx"),
        ).transpose(1, 2)
    else:
        x = mixed_qkv.transpose(1, 2)
        mixed_qkv = F.silu(self.conv1d(x)[:, :, : x.shape[-1]]).transpose(1, 2)

    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    extra: dict[str, Any] = {}
    if "gdn" in self._fused:
        g = a  # raw: the kernel applies -exp(A_log) * softplus(a + dt_bias)
        extra = {"use_gate_in_kernel": True, "A_log": self.A_log, "dt_bias": self.dt_bias}
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=None,
        output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True, **extra,
    )  # fmt: skip
    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def patch_gdn(module: Any, which: set[str], k: Kernels, torch: Any, modeling: Any) -> None:
    module._fused = set(which)
    module._fused_kernels, module._fused_torch, module._fused_modeling = k, torch, modeling
    if "gdn" in which:  # fla's kernel takes the 16 key heads and computes the gate itself
        module.chunk_gated_delta_rule = k.chunk_gated_delta_rule
    module._fused_orig_forward = module.forward
    module.forward = types.MethodType(gdn_forward, module)


def apply(
    model: Any,
    fusions: tuple[str, ...] | list[str],
    kernels: Kernels | None = None,
    torch: Any = None,
    modeling: Any = None,
) -> dict[str, Any]:
    """Patch `model` in place. Returns {fusion: modules patched} (plus "skipped" reasons).

    `kernels`, `torch` and `modeling` default to fla's kernels, the real torch and
    transformers' modeling_qwen3_5; the tests pass references. Nothing is patched twice: an
    RMSNorm or MLP that already has _fused_orig_forward is left alone, and a GDN layer already
    patched for some of gdn/conv gets only the ones it lacks (so fusions can be added one at a
    time, as scripts/fusion_check.py's latency ladder does)."""
    report: dict[str, Any] = {f: 0 for f in fusions}
    if not fusions:
        return report
    if torch is None:
        import torch as _torch

        torch = _torch
    if modeling is None and ({"gdn", "conv"} & set(fusions)):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    skipped: dict[str, str] = {}
    if "cublaslt" in fusions:
        try:
            torch.backends.cuda.preferred_blas_library("cublaslt")
            report["cublaslt"] = 1
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            skipped["cublaslt"] = f"{type(e).__name__}: {e}"
    if "nocache" in fusions:
        configs = [c for c in (getattr(model, "config", None),) if c is not None]
        configs += [getattr(c, "text_config", None) for c in list(configs)]
        configs += [getattr(m, "config", None) for m in model.modules()]
        seen = set()
        for c in configs:
            if c is not None and id(c) not in seen and hasattr(c, "use_cache"):
                seen.add(id(c))
                c.use_cache = False
        report["nocache"] = len(seen)
    module_fusions = [f for f in fusions if f not in ("cublaslt", "nocache")]
    if module_fusions and kernels is None:
        try:
            kernels = fla_kernels()
        except Exception as e:  # noqa: BLE001 - no fla: nothing to fuse with
            for f in module_fusions:
                skipped[f] = f"fla unavailable: {type(e).__name__}: {e}"
            module_fusions = []
    gdn_set = {f for f in module_fusions if f in ("gdn", "conv")}
    for module in list(model.modules()):
        name = _class(module)
        if name == "Qwen3_5GatedDeltaNet" and gdn_set:
            new = gdn_set - getattr(module, "_fused", set())
            if not new:
                continue
            if module.activation not in ("silu", "swish") and "conv" in new:
                skipped["conv"] = f"activation {module.activation!r} is not SiLU"
                continue
            if hasattr(module, "_fused"):  # patched for part of the set: add the rest
                module._fused |= new
                if "gdn" in new:
                    module.chunk_gated_delta_rule = kernels.chunk_gated_delta_rule
            else:
                patch_gdn(module, new, kernels, torch, modeling)
            for f in new:
                report[f] += 1
            continue
        if hasattr(module, "_fused_orig_forward"):
            continue
        if name == "Qwen3_5RMSNorm" and "rmsnorm" in module_fusions:
            patch_rmsnorm(module, kernels, torch)
            report["rmsnorm"] += 1
        elif name == "Qwen3_5MLP" and "mlp" in module_fusions:
            if type(module.act_fn).__name__ not in ("SiLU", "SiLUActivation"):
                skipped["mlp"] = f"activation {type(module.act_fn).__name__} is not SiLU"
                continue
            patch_mlp(module, kernels, torch)
            report["mlp"] += 1
    if skipped:
        report["skipped"] = skipped
    return report
