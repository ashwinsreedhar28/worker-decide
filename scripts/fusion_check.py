#!/usr/bin/env python3
"""Tier 2 on a GPU: each fusion's real fla kernel against the module it replaces, then what each
fusion buys, added one at a time to one loaded model.

tests/test_fused.py proves the patching logic on the CPU with exact torch stand-ins for fla's
kernels. This checks the kernels themselves, on the real model with real activations:

1. Module checks. One unfused bf16 engine runs a fixture record and a right-padded batch of two;
   forward pre-hooks keep the inputs of a few modules of each patched class (the first
   --per-class of each plus the last, and the attention q/k norms). For each kept module and
   each fusion that applies to its class, three outputs from the same input:

       orig    the module as loaded (bf16)
       ref     an fp32 copy (GDN: transformers' torch recurrence and torch conv, so no fla)
       fused   a bf16 copy patched with fused.apply and the real fla kernels

   Verdict: fused must actually be patched (apply reports 1), finite, and no further from ref
   than twice orig's distance plus a floor (max relative 1e-3, mean relative 1e-4). bf16 is
   allowed its own rounding; a fusion is not allowed to be worse than the path it replaces.
   GDN outputs are compared on valid (unpadded) positions only.

2. Latency ladder (skip with --no-ladder). On the same engine, fusions are added one at a time
   (none, +rmsnorm, +mlp, +gdn, +conv, +nocache, +cublaslt). After each, per workload record:
   the first pass (Triton compiles the new kernels then, so it is the cold-start cost the fusion
   adds), then --warmup passes, then --reps timed passes split like profile_engine.py, and the
   probabilities' max |Δp| against the unfused step (an end-to-end drift check on 3 records;
   the 87-record gate runs separately over HTTP in scripts/pod_tier2.sh).

    python scripts/fusion_check.py runs/fusion-check-a40 [--records card_trio pulse long_2k]
        [--batch pulse card_trio] [--per-class 2] [--reps 10] [--warmup 3] [--no-ladder]

Writes fusion_check.json and fusion_check.md (module checks first, so a ladder crash keeps
them). Exits 1 if a module check fails, 2 if fused.apply skipped a fusion.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deciders.clef import fused  # noqa: E402

CLASS_OF = {"rmsnorm": "Qwen3_5RMSNorm", "mlp": "Qwen3_5MLP", "gdn": "Qwen3_5GatedDeltaNet",
            "conv": "Qwen3_5GatedDeltaNet"}  # fmt: skip
CHECKS = (("rmsnorm",), ("mlp",), ("gdn",), ("conv",), ("gdn", "conv"))
LADDER = ("rmsnorm", "mlp", "gdn", "conv", "nocache", "cublaslt")
MAX_FLOOR, MEAN_FLOOR, FACTOR = 1e-3, 1e-4, 2.0
CACHE_KEYS = ("cache_params", "past_key_values", "past_key_value")


def pick(named: Iterable[tuple[str, str]], per_class: int = 2) -> dict[str, str]:
    """{module name: class} to check: per patched class, the first `per_class` and the last;
    for RMSNorm also the first attention q_norm and k_norm (a different input shape)."""
    by_class: dict[str, list[str]] = {}
    for name, cls in named:
        if cls in CLASS_OF.values():
            by_class.setdefault(cls, []).append(name)
    out: dict[str, str] = {}
    for cls, names in by_class.items():
        chosen = names[:per_class] + names[-1:]
        if cls == "Qwen3_5RMSNorm":
            for suffix in ("q_norm", "k_norm"):
                chosen += [n for n in names if n.endswith(suffix)][:1]
        for n in chosen:
            out[n] = cls
    return out


def verdict(e_orig: dict[str, float], e_fused: dict[str, float], patched: bool,
            finite: bool) -> tuple[bool, str]:  # fmt: skip
    if not patched:
        return False, "not patched (fused.apply reported 0)"
    if not finite:
        return False, "non-finite output"
    lim_max = FACTOR * e_orig["max_rel"] + MAX_FLOOR
    lim_mean = FACTOR * e_orig["mean_rel"] + MEAN_FLOOR
    bad = []
    if e_fused["max_rel"] > lim_max:
        bad.append(f"max_rel {e_fused['max_rel']:.2e} > {lim_max:.2e}")
    if e_fused["mean_rel"] > lim_mean:
        bad.append(f"mean_rel {e_fused['mean_rel']:.2e} > {lim_mean:.2e}")
    return (not bad), ("; ".join(bad) or "ok")


class Capture:
    """Forward pre-hooks on the picked modules; keeps the first call's inputs per name, cloned,
    with any cache argument dropped (the fused path is checked without a cache, as it runs
    with nocache)."""

    def __init__(self, model: Any, names: Iterable[str], torch: Any):
        self.torch, self.inputs, self.handles = torch, {}, []
        wanted = set(names)
        for name, module in model.named_modules():
            if name in wanted:
                self.handles.append(
                    module.register_forward_pre_hook(self._hook(name), with_kwargs=True)
                )

    def _clone(self, x: Any) -> Any:
        if isinstance(x, self.torch.Tensor):
            return x.detach().clone()
        if isinstance(x, tuple | list):
            return type(x)(self._clone(v) for v in x)
        return x

    def _hook(self, name: str) -> Callable:
        def hook(module, args, kwargs):
            if name not in self.inputs:
                kw = {k: (None if k in CACHE_KEYS else self._clone(v)) for k, v in kwargs.items()}
                self.inputs[name] = (self._clone(args), kw)

        return hook

    def __enter__(self) -> Capture:
        return self

    def __exit__(self, *exc) -> None:
        for h in self.handles:
            h.remove()


def _fp32(x: Any, torch: Any) -> Any:
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.float()
    if isinstance(x, tuple | list):
        return type(x)(_fp32(v, torch) for v in x)
    if isinstance(x, dict):
        return {k: _fp32(v, torch) for k, v in x.items()}
    return x


def fp32_copy(module: Any, modeling: Any) -> Any:
    ref = copy.deepcopy(module).float()
    if type(module).__name__ == "Qwen3_5GatedDeltaNet":  # no fla in the reference
        ref.chunk_gated_delta_rule = modeling.torch_chunk_gated_delta_rule
        ref.causal_conv1d_fn = None
    return ref


def errors(out: Any, ref: Any, mask: Any = None) -> dict[str, float]:
    """Relative distance of `out` from `ref`: max |d| / max |ref| and mean |d| / mean |ref|,
    over the positions `mask` ([B, T], nonzero = valid) keeps."""
    o, r = out.float(), ref.float()
    if mask is not None and o.dim() >= 2 and tuple(o.shape[:2]) == tuple(mask.shape):
        keep = mask.bool()
        o, r = o[keep], r[keep]
    d = (o - r).abs()
    return {"max_rel": (d.max() / r.abs().max().clamp_min(1e-30)).item(),
            "mean_rel": (d.mean() / r.abs().mean().clamp_min(1e-30)).item()}  # fmt: skip


def check(module: Any, args: tuple, kwargs: dict, fusions: tuple[str, ...], torch: Any,
          modeling: Any, kernels: Any = None) -> dict[str, Any]:  # fmt: skip
    mask = kwargs.get("attention_mask") if type(module).__name__ == "Qwen3_5GatedDeltaNet" else None
    if mask is not None and mask.dim() != 2:
        mask = None
    with torch.inference_mode():
        out_orig = module(*args, **kwargs)
        ref = fp32_copy(module, modeling)
        out_ref = ref(*_fp32(args, torch), **_fp32(kwargs, torch))
        del ref
        patched_mod = copy.deepcopy(module)
        report = fused.apply(patched_mod, fusions, kernels=kernels, torch=torch, modeling=modeling)
        patched = all(report.get(f, 0) == 1 for f in fusions)
        out_fused = patched_mod(*args, **kwargs)
        del patched_mod
    e_orig, e_fused = errors(out_orig, out_ref, mask), errors(out_fused, out_ref, mask)
    finite = bool(torch.isfinite(out_fused).all().item())
    ok, why = verdict(e_orig, e_fused, patched, finite)
    return {"fusions": "+".join(fusions), "shape": list(out_orig.shape), "patched": patched,
            "report": report, "orig_vs_fp32": e_orig, "fused_vs_fp32": e_fused,
            "fused_vs_orig": errors(out_fused, out_orig, mask), "ok": ok, "why": why}  # fmt: skip


def module_checks(model: Any, captured: dict[str, dict[str, tuple]], torch: Any, modeling: Any,
                  kernels: Any = None) -> list[dict[str, Any]]:  # fmt: skip
    """captured: {tag: {module name: (args, kwargs)}}."""
    modules = dict(model.named_modules())
    rows = []
    for tag, inputs in captured.items():
        for name, (args, kwargs) in inputs.items():
            cls = type(modules[name]).__name__
            for fusions in CHECKS:
                if all(CLASS_OF[f] == cls for f in fusions):
                    row = check(modules[name], args, kwargs, fusions, torch, modeling, kernels)
                    rows.append({"tag": tag, "module": name, **row})
    return rows


def ladder(apply: Callable[[tuple[str, ...]], dict], run_pass: Callable[[Any], dict],
           probs: Callable[[Any], list[float]], records: list[tuple[str, Any]], reps: int,
           warmup: int, steps: Iterable[str] = LADDER,
           log: Callable[[str], Any] = print) -> list[dict[str, Any]]:  # fmt: skip
    """Fusions added one at a time. run_pass(rec) -> {"total", "backbone", "input_tokens"} in ms;
    probs(rec) -> the record's probabilities, flattened."""
    base: dict[str, list[float]] = {}
    out = []
    for label, add in [("none", ())] + [(f"+{f}", (f,)) for f in steps]:
        report = apply(add) if add else {}
        per: dict[str, Any] = {}
        for rid, rec in records:
            first = run_pass(rec)
            for _ in range(warmup):
                run_pass(rec)
            samples = [run_pass(rec) for _ in range(reps)]
            p = probs(rec)
            base.setdefault(rid, p)
            per[rid] = {
                "input_tokens": first.get("input_tokens"),
                "first_ms": round(first["total"], 2),
                "total_ms": round(statistics.median(s["total"] for s in samples), 2),
                "backbone_ms": round(statistics.median(s["backbone"] for s in samples), 2),
                "max_dp_vs_none": max(
                    (abs(a - b) for a, b in zip(p, base[rid], strict=True)), default=0.0
                ),  # fmt: skip
            }
        out.append({"step": label, "report": report, "records": per})
        log(f"[ladder] {label}: " + ", ".join(f"{r} {v['total_ms']} ms" for r, v in per.items()))
    return out


def render(gpu: str | None, checks: list[dict], steps: list[dict] | None) -> str:
    n_bad = sum(not c["ok"] for c in checks)
    lines = [f"# Tier 2 fusion check ({gpu or 'unknown GPU'})", "",
             f"Module checks: {len(checks) - n_bad}/{len(checks)} ok. Distances are relative to "
             "an fp32 copy of the module; a fusion passes if it is no further than "
             f"{FACTOR:g}x the bf16 original's distance plus a floor ({MAX_FLOOR:g} max, "
             f"{MEAN_FLOOR:g} mean).", "",
             "| input | module | fusions | orig max / mean | fused max / mean | fused vs orig max "
             "| verdict |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for c in checks:
        o, f, v = c["orig_vs_fp32"], c["fused_vs_fp32"], c["fused_vs_orig"]
        state = "ok" if c["ok"] else "FAIL: " + c["why"]
        lines.append(f"| {c['tag']} | `{c['module']}` | {c['fusions']} | {o['max_rel']:.2e} / "
                     f"{o['mean_rel']:.2e} | {f['max_rel']:.2e} / {f['mean_rel']:.2e} | "
                     f"{v['max_rel']:.2e} | {state} |")  # fmt: skip
    if steps:
        rids = list(steps[0]["records"])
        lines += ["", "## Latency ladder (fusions added one at a time; ms, medians)", "",
                  "| step | " + " | ".join(f"{r} total (Δ) / first / max Δp" for r in rids) + " |",
                  "|---|" + "---|" * len(rids)]  # fmt: skip
        prev = None
        for s in steps:
            cells = []
            for r in rids:
                v = s["records"][r]
                delta = (
                    ""
                    if prev is None
                    else f" ({v['total_ms'] - prev['records'][r]['total_ms']:+.1f})"
                )
                cells.append(f"{v['total_ms']:.1f}{delta} / {v['first_ms']:.0f} / "
                             f"{v['max_dp_vs_none']:.4f}")  # fmt: skip
            lines.append(f"| {s['step']} | " + " | ".join(cells) + " |")
            prev = s
        toks = ", ".join(f"{r} {steps[0]['records'][r]['input_tokens']}" for r in rids)
        lines += ["", f"Tokens: {toks}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--records", nargs="+", default=["card_trio", "pulse", "long_2k"])
    ap.add_argument("--batch", nargs="+", default=["pulse", "card_trio"],
                    help="records for the padded-batch capture")  # fmt: skip
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-ladder", action="store_true")
    a = ap.parse_args(argv)

    import dataclasses

    import torch
    from loadtest import load_line
    from profile_engine import Splitter, backbone_chain, record_for
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    from deciders.clef.engine import ClefEngine
    from worker.core import config_from_env

    cfg = dataclasses.replace(config_from_env(), fused="", warmup=False)  # start unfused
    engine = ClefEngine(cfg)
    if engine.device.type != "cuda":
        raise SystemExit("fusion_check needs a GPU")
    gpu = torch.cuda.get_device_name(0)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    def rec(rid: str) -> tuple[str, dict]:
        line = load_line(None if rid == "pulse" else rid, "fixtures/requests/*.jsonl")
        return line["id"], record_for(line["id"], line)

    def collate(encoded: list) -> Any:
        return engine.jsm.collate_records(encoded, engine.processor.tokenizer.pad_token_id,
                                          engine.device)  # fmt: skip

    names = pick(((n, type(m).__name__) for n, m in engine.model.named_modules()), a.per_class)
    captured: dict[str, dict[str, tuple]] = {}
    with torch.inference_mode():
        _, single = rec(a.batch[0])
        with Capture(engine.model, names, torch) as cap:
            engine._forward(collate([engine.encode(single)]))
        captured["single"] = cap.inputs
        with Capture(engine.model, names, torch) as cap:  # right-padded, as the batcher pads
            engine._forward_records(collate([engine.encode(r) for _, r in map(rec, a.batch)]))
        captured["batch"] = cap.inputs
    missing = sorted(set(names) - set(captured["single"]))
    checks = module_checks(engine.model, captured, torch, modeling)
    out: dict[str, Any] = {"engine": engine.manifest(), "gpu": gpu, "picked": names,
                           "not_called": missing, "module_checks": checks}  # fmt: skip

    def write(steps: list | None) -> None:
        out["ladder"] = steps
        (a.out_dir / "fusion_check.json").write_text(json.dumps(out, indent=1))
        (a.out_dir / "fusion_check.md").write_text(render(gpu, checks, steps))

    write(None)
    print(render(gpu, checks, None), flush=True)

    steps = None
    skipped: dict[str, Any] = {}
    if not a.no_ladder:
        split = Splitter(torch, backbone_chain(engine.model))

        def apply(fusions: tuple[str, ...]) -> dict:
            report = fused.apply(engine.model, fusions)
            skipped.update(report.get("skipped", {}))
            return report

        def probs(r: dict) -> list[float]:
            with torch.inference_mode():
                logits = engine._forward(collate([engine.encode(r)]))
            return [p for q in logits for p in q.float().softmax(-1).flatten().tolist()]

        steps = ladder(apply, lambda r: split.one_pass(engine, r), probs,
                       [rec(r) for r in a.records], a.reps, a.warmup,
                       log=lambda s: print(s, file=sys.stderr, flush=True))  # fmt: skip
        out["ladder_skipped"] = skipped
        write(steps)
        print(render(gpu, [], steps).split("## ", 1)[-1], flush=True)
    if any(not c["ok"] for c in checks):
        return 1
    return 2 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
