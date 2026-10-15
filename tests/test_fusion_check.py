"""scripts/fusion_check.py: the pure parts without torch; the module check itself on the tiny
Qwen3.5 from test_fused.py (CPU, exact torch stand-ins for fla), including in bf16 and with a
deliberately wrong kernel, which it must fail."""

import fusion_check as fc  # scripts/ is on sys.path (conftest.py)
import pytest

E = {"max_rel": 4e-3, "mean_rel": 5e-4}


def test_pick_takes_first_and_last_per_class_and_the_qk_norms():
    named = [("embed", "Embedding")]
    for i in range(4):
        named += [(f"layers.{i}.input_layernorm", "Qwen3_5RMSNorm"),
                  (f"layers.{i}.mlp", "Qwen3_5MLP")]  # fmt: skip
        if i < 3:
            named.append((f"layers.{i}.linear_attn", "Qwen3_5GatedDeltaNet"))
    named += [("layers.3.self_attn.q_norm", "Qwen3_5RMSNorm"),
              ("layers.3.self_attn.k_norm", "Qwen3_5RMSNorm"),
              ("norm", "Qwen3_5RMSNorm")]  # fmt: skip
    got = fc.pick(named, per_class=1)
    assert set(got) == {"layers.0.input_layernorm", "norm", "layers.3.self_attn.q_norm",
                        "layers.3.self_attn.k_norm", "layers.0.mlp", "layers.3.mlp",
                        "layers.0.linear_attn", "layers.2.linear_attn"}  # fmt: skip
    assert "embed" not in got


def test_verdict_allows_bf16_noise_and_refuses_worse_or_unpatched():
    assert fc.verdict(E, {"max_rel": 8e-3, "mean_rel": 1e-3}, True, True) == (True, "ok")
    ok, why = fc.verdict(E, {"max_rel": 1e-2, "mean_rel": 1e-3}, True, True)
    assert not ok and "max_rel" in why
    ok, why = fc.verdict(E, {"max_rel": 1e-3, "mean_rel": 2e-3}, True, True)
    assert not ok and "mean_rel" in why
    assert fc.verdict(E, E, False, True)[0] is False  # a no-op patch cannot pass
    assert fc.verdict(E, E, True, False)[0] is False
    zero = {"max_rel": 0.0, "mean_rel": 0.0}  # fp32 original: the floors apply
    assert fc.verdict(zero, {"max_rel": 9e-4, "mean_rel": 9e-5}, True, True)[0]
    assert not fc.verdict(zero, {"max_rel": 2e-3, "mean_rel": 0.0}, True, True)[0]


def test_ladder_adds_one_fusion_per_step_and_reports_drift():
    applied, state = [], {"ms": 100.0}

    def apply(f):
        applied.append(f)
        state["ms"] -= 10
        return {f[0]: 1}

    def run_pass(rec):
        return {"total": state["ms"], "backbone": state["ms"] - 5, "input_tokens": 7}

    steps = fc.ladder(apply, run_pass, lambda r: [0.5 + 0.01 * len(applied), 0.5], [("a", {})],
                      reps=3, warmup=1, steps=("rmsnorm", "mlp"), log=lambda s: None)  # fmt: skip
    assert applied == [("rmsnorm",), ("mlp",)]
    assert [s["step"] for s in steps] == ["none", "+rmsnorm", "+mlp"]
    assert [s["records"]["a"]["total_ms"] for s in steps] == [100.0, 90.0, 80.0]
    assert steps[2]["records"]["a"]["max_dp_vs_none"] == pytest.approx(0.02)
    md = fc.render("GPU", [], steps)
    assert "| +mlp | 80.0 (-10.0)" in md and "a 7" in md


# ---- the module check on a real (tiny) Qwen3.5 ------------------------------------------------


@pytest.fixture
def tf():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from tests import test_fused

    return torch, test_fused


def captured(torch, tf_mod, model):
    names = fc.pick(((n, type(m).__name__) for n, m in model.named_modules()), per_class=1)
    mask = torch.ones(2, 37, dtype=torch.long)
    mask[1, 25:] = 0
    with torch.inference_mode(), fc.Capture(model, names, torch) as cap:
        model(input_ids=tf_mod.ids(), attention_mask=mask)  # default use_cache: a cache exists
    return names, cap.inputs


def test_capture_keeps_inputs_drops_the_cache_and_removes_its_hooks(tf):
    torch, t = tf
    model = t.tiny_model()
    names, inputs = captured(torch, t, model)
    assert set(inputs) == set(names)
    args, kw = inputs["layers.0.linear_attn"]
    assert kw.get("cache_params") is None and kw["attention_mask"].shape == (2, 37)
    assert all(not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_every_check_passes_with_exact_kernels(tf, dtype):
    torch, t = tf
    model = t.tiny_model().to(getattr(torch, dtype))
    _, inputs = captured(torch, t, model)
    k, _ = t.ref_kernels()
    rows = fc.module_checks(model, {"batch": inputs}, torch, t.modeling, kernels=k)
    assert {r["fusions"] for r in rows} == {"rmsnorm", "mlp", "gdn", "conv", "gdn+conv"}
    assert all(r["ok"] and r["patched"] for r in rows), [r for r in rows if not r["ok"]]
    gdn = next(r for r in rows if r["fusions"] == "gdn+conv")
    if dtype == "bfloat16":  # the bf16 original really is measured against an fp32 copy
        assert gdn["orig_vs_fp32"]["max_rel"] > 0
    assert not any(hasattr(m, "_fused_orig_forward") for m in model.modules())  # copies only


def test_a_wrong_kernel_fails_and_a_check_on_the_wrong_class_is_not_patched(tf):
    torch, t = tf
    model = t.tiny_model().to(torch.bfloat16)
    _, inputs = captured(torch, t, model)
    k, _ = t.ref_kernels()
    good = k.swiglu
    k.swiglu = lambda g, u: good(g, u) * 1.02  # 2% off
    rows = fc.module_checks(model, {"batch": inputs}, torch, t.modeling, kernels=k)
    assert all(not r["ok"] for r in rows if r["fusions"] == "mlp")
    assert all(r["ok"] for r in rows if r["fusions"] != "mlp")
    modules = dict(model.named_modules())
    args, kw = inputs["layers.0.mlp"]
    row = fc.check(modules["layers.0.mlp"], args, kw, ("rmsnorm",), torch, t.modeling, kernels=k)
    assert not row["patched"] and not row["ok"] and "not patched" in row["why"]


def test_errors_ignore_padded_positions_and_the_reference_never_runs_fla(tf):
    torch, t = tf
    ref = torch.ones(2, 4, 3)
    out = ref.clone()
    out[1, 3] += 100.0  # garbage at a padded position
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    assert fc.errors(out, ref, mask) == {"max_rel": 0.0, "mean_rel": 0.0}
    assert fc.errors(out, ref)["max_rel"] == 100.0
    gdn = next(m for m in t.tiny_model().modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet")
    gdn.chunk_gated_delta_rule = lambda *a, **k: None  # stands in for fla's binding
    gdn.causal_conv1d_fn = lambda *a, **k: None
    r = fc.fp32_copy(gdn, t.modeling)
    assert r.chunk_gated_delta_rule is t.modeling.torch_chunk_gated_delta_rule
    assert r.causal_conv1d_fn is None and r.in_proj_qkv.weight.dtype == torch.float32
