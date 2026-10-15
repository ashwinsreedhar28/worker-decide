"""deciders/clef/fused.py on a tiny Qwen3.5 text model, on the CPU, with exact torch references
standing in for fla's kernels: the patched model must compute what the unpatched one does.

Needs torch and transformers 5.10.2 (the CI job `fusions` installs CPU torch for this); skipped
without them. The fla kernels themselves are checked against the originals on a GPU
(scripts/fusion_check.py, scripts/pod_tier2.sh)."""

import copy

import pytest

from deciders.clef import fused

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
F = torch.nn.functional


def ref_kernels(calls=None):
    calls = calls if calls is not None else []

    def rms_norm(x, w, eps):
        calls.append("rms_norm")
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)

    def swiglu(g, u):
        calls.append("swiglu")
        return F.silu(g) * u

    def causal_conv1d(x, weight, bias, activation):
        calls.append("conv")
        assert activation in ("silu", "swish")
        width = weight.shape[-1]
        y = F.conv1d(x.transpose(1, 2), weight.unsqueeze(1), bias, padding=width - 1,
                     groups=weight.shape[0])[:, :, : x.shape[1]]  # fmt: skip
        return F.silu(y).transpose(1, 2).contiguous()

    def chunk(q, k, v, g, beta, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, A_log=None,
              dt_bias=None, **kw):  # fmt: skip
        calls.append(("chunk", q.shape[2], v.shape[2], use_gate_in_kernel))
        if use_gate_in_kernel:  # fla's documented formula: -exp(A_log) * softplus(g + dt_bias)
            g = -A_log.float().exp() * F.softplus(g.float() + dt_bias)
        if q.shape[2] < v.shape[2]:  # fla's GVA: value head i reads key head i // (HV // H)
            rep = v.shape[2] // q.shape[2]
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        return modeling.torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )  # fmt: skip

    return fused.Kernels(rms_norm=rms_norm, swiglu=swiglu, causal_conv1d=causal_conv1d,
                         chunk_gated_delta_rule=chunk), calls  # fmt: skip


def tiny_model(seed=0):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )  # fmt: skip
    model = modeling.Qwen3_5TextModel(cfg).eval()
    with torch.no_grad():  # the init leaves these at 0 / 1: randomize so a wrong formula shows
        for m in model.modules():
            if type(m).__name__ == "Qwen3_5RMSNorm":
                m.weight.normal_(0, 0.3)
            if type(m).__name__ == "Qwen3_5GatedDeltaNet":
                m.dt_bias.normal_(0, 0.5)
                m.A_log.uniform_(-1, 1)
    return model


def run(model, ids, mask=None, use_cache=None):
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=mask, use_cache=use_cache).last_hidden_state


IDS = None


def ids():
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, 37))


@pytest.mark.parametrize("use_cache", [None, False])  # None: the config's default (True)
@pytest.mark.parametrize("which", [("rmsnorm",), ("mlp",), ("gdn",), ("conv",), ("gdn", "conv"),
                                   ("rmsnorm", "mlp", "gdn", "conv")])  # fmt: skip
def test_each_fusion_computes_what_the_unpatched_model_does(which, use_cache):
    base = tiny_model()
    patched = copy.deepcopy(base)
    k, calls = ref_kernels()
    report = fused.apply(patched, which, kernels=k, torch=torch, modeling=modeling)
    want, got = run(base, ids(), use_cache=use_cache), run(patched, ids(), use_cache=use_cache)
    assert (want - got).abs().max().item() < 2e-5, which
    for f in which:
        assert report[f] > 0, report
    if "gdn" in which:  # q/k reach the kernel at their own 2 heads, the gate computed inside
        assert ("chunk", 2, 4, True) in calls and not any(
            c[0] == "chunk" and c[1] == 4 for c in calls if isinstance(c, tuple)
        )
    if "conv" in which:
        assert "conv" in calls


def test_right_padded_batches_match_too():
    base = tiny_model()
    patched = copy.deepcopy(base)
    k, _ = ref_kernels()
    fused.apply(
        patched, ("rmsnorm", "mlp", "gdn", "conv"), kernels=k, torch=torch, modeling=modeling
    )
    mask = torch.ones(2, 37, dtype=torch.long)
    mask[1, 25:] = 0  # the second record is 25 tokens, padded to 37
    want, got = run(base, ids(), mask), run(patched, ids(), mask)
    assert (want[0] - got[0]).abs().max().item() < 2e-5
    assert (want[1, :25] - got[1, :25]).abs().max().item() < 2e-5


def test_report_counts_every_module_of_each_kind():
    model = tiny_model()
    k, _ = ref_kernels()
    report = fused.apply(model, fused.FUSIONS[:-1], kernels=k, torch=torch, modeling=modeling)
    names = [type(m).__name__ for m in model.modules()]
    assert report["rmsnorm"] == names.count("Qwen3_5RMSNorm") == 11  # 8 decoder, final, q/k norm
    assert report["mlp"] == names.count("Qwen3_5MLP") == 4
    assert report["gdn"] == report["conv"] == names.count("Qwen3_5GatedDeltaNet") == 3
    assert "skipped" not in report


def test_the_mlp_merge_shares_storage_with_the_original_weights():
    model = tiny_model()
    k, _ = ref_kernels()
    fused.apply(model, ("mlp",), kernels=k, torch=torch, modeling=modeling)
    mlp = next(m for m in model.modules() if type(m).__name__ == "Qwen3_5MLP")
    merged = mlp._fused_gate_up
    assert mlp.gate_proj.weight.data_ptr() == merged.data_ptr()
    assert mlp.up_proj.weight.data_ptr() == merged[mlp.intermediate_size :].data_ptr()


def test_applying_twice_does_not_patch_twice_and_parse_validates():
    model = tiny_model()
    k, calls = ref_kernels()
    fused.apply(model, ("rmsnorm",), kernels=k, torch=torch, modeling=modeling)
    again = fused.apply(model, ("rmsnorm",), kernels=k, torch=torch, modeling=modeling)
    assert again["rmsnorm"] == 0
    run(model, ids())
    norms = calls.count("rms_norm")
    assert norms == 11  # once per module per forward, not twice
    assert fused.parse("") == () and fused.parse(None) == ()
    assert fused.parse("all") == fused.FUSIONS
    assert fused.parse("conv, RMSNORM,conv") == ("rmsnorm", "conv")
    with pytest.raises(ValueError, match="unknown FUSED"):
        fused.parse("rmsnorm,flash")


def test_a_cached_call_goes_to_the_original_forward():
    model = tiny_model()
    k, calls = ref_kernels()
    fused.apply(model, ("gdn",), kernels=k, torch=torch, modeling=modeling)
    gdn = next(m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet")
    seen = []
    gdn._fused_orig_forward = lambda *a, **kw: seen.append(kw.get("cache_params")) or "orig"

    class Continuing:  # a cache that already holds this layer's state: decode, not prefill
        def has_previous_state(self, layer_idx):
            return True

    cache = Continuing()
    assert gdn(torch.zeros(1, 3, 64), cache_params=cache) == "orig" and seen == [cache]
    assert gdn(torch.zeros(1, 3, 64), cu_seq_lens_q=torch.tensor([0, 3])) == "orig"
    assert not calls  # neither reached the fused path


def test_without_fla_module_fusions_are_skipped_and_reported(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "fla", None)  # import fails
    model = tiny_model()
    report = fused.apply(model, ("rmsnorm", "mlp"), torch=torch, modeling=modeling)
    assert report["rmsnorm"] == report["mlp"] == 0
    assert "fla unavailable" in report["skipped"]["rmsnorm"]
    assert not any(hasattr(m, "_fused_orig_forward") for m in model.modules())


def test_a_prefill_with_a_cache_fills_it_as_the_original_does():
    base = tiny_model()
    patched = copy.deepcopy(base)
    k, _ = ref_kernels()
    fused.apply(patched, ("gdn", "conv"), kernels=k, torch=torch, modeling=modeling)
    with torch.inference_mode():
        a = base(input_ids=ids(), use_cache=True)
        b = patched(input_ids=ids(), use_cache=True)
    for i, layer in enumerate(a.past_key_values.layers):
        lb = b.past_key_values.layers[i]
        for name in ("conv_states", "recurrent_states"):
            x, y = getattr(layer, name, None), getattr(lb, name, None)
            if x is not None:
                assert y is not None and x.shape == y.shape, (i, name)
                assert (x - y).abs().max().item() < 2e-5, (i, name)


def test_nocache_turns_the_cache_off_and_changes_no_output():
    base = tiny_model()
    patched = copy.deepcopy(base)
    report = fused.apply(patched, ("nocache",), torch=torch, modeling=modeling)
    assert report["nocache"] >= 1 and patched.config.use_cache is False
    with torch.inference_mode():
        out = patched(input_ids=ids())
    assert out.past_key_values is None
    assert (run(base, ids()) - out.last_hidden_state).abs().max().item() < 1e-6
