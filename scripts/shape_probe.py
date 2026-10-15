#!/usr/bin/env python3
"""Per-shape cost: does a request whose exact token count the worker has not seen run slower?

The tier 0 H100 run (Oct 15) suggested so: in the parity pass, text requests under 2,048 tokens
with a token count not seen before took a median 166 ms, while ones whose exact count had
already been seen took 54 ms; the pure-torch run of Oct 14 shows the same split (225 vs 130 ms).
The extra time was in the forward, not the encode. The load test repeats one request, so it
never sees this cost; real traffic, where nearly every request has a new length, would.

This loads ClefEngine in-process (same config as the worker), takes one fixture record with a
string `state`, and makes variants with distinct token counts by appending " x" k times. For
each variant it runs, in order:

    new      first pass at this token count
    repeat   the same record again (same count, same content)
    sibling  same count, different content (" y" instead of " x"), when the tokenizer agrees

and splits each pass like profile_engine.py (encode, collate, pre, backbone, head, post; CUDA
events), plus the cudaMalloc count from torch.cuda.memory_stats. If new is slow and repeat and
sibling are fast, the cost belongs to the shape, not the content. --trace profiles one new pass
and its repeat under torch.profiler and writes the ops whose self CPU or CUDA time grew most,
which names the cause (GEMM heuristics inside aten::mm, lazy kernel loading inside launches,
allocator calls, or the head).

Groundwork for batching (tier 1) and length buckets: the structure collate_records builds for
one record, and a batch preview. The preview collates several records of different lengths and
question shapes into one batch, describes the batch and the model's output, times the batched
forward against the records one at a time, and compares each question's probabilities from
the batch with the single-record run (pairing outputs in order; "comparable" is false when the
flattened outputs do not line up one to one in shape).

    python scripts/shape_probe.py runs/shapes-h100 [--record pulse] [--variants 8] [--trace]
        [--batch-records pulse card_trio long_2k]

Run each environment variant in its own process (the pod script does): the allocator, module
loading and BLAS choices are fixed at CUDA initialization.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from profile_engine import SEGMENTS  # noqa: E402

KINDS = ("new", "repeat", "sibling")
ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_MODULE_LOADING", "TORCH_BLAS_PREFER_CUBLASLT",
            "TRITON_CACHE_DIR")  # fmt: skip


def make_variants(state: str, n: int, count, max_tries: int = 64) -> list[dict[str, Any]]:
    """Up to n variants with token counts that differ from each other and from the base state.

    `count(state) -> int` is the encoded length of the record with that state. Each variant is
    {"k", "state", "tokens", "sibling"}; sibling is the same-length text with " y" (or " z")
    in place of " x", or None when no replacement keeps the length.
    """
    seen = {count(state)}
    out: list[dict[str, Any]] = []
    k = 0
    while len(out) < n and k < max_tries:
        k += 1
        s = state + " x" * k
        t = count(s)
        if t in seen:
            continue
        seen.add(t)
        sib = next((c for c in (state + " y" * k, state + " z" * k) if count(c) == t), None)
        out.append({"k": k, "state": s, "tokens": t, "sibling": sib})
    return out


def summarize(passes: list[dict[str, Any]]) -> dict[str, Any]:
    """Median of each segment (and total) per kind, and new minus repeat."""
    out: dict[str, Any] = {}
    for kind in KINDS:
        rows = [p for p in passes if p["kind"] == kind]
        if rows:
            med = {k: round(statistics.median(r[k] for r in rows), 2) for k in (*SEGMENTS, "total")}
            allocs = statistics.median(r.get("device_allocs", 0) for r in rows)
            out[kind] = {"n": len(rows), **med, "device_allocs": round(allocs, 1)}
    if "new" in out and "repeat" in out:
        out["new_minus_repeat"] = {k: round(out["new"][k] - out["repeat"][k], 2)
                                   for k in (*SEGMENTS, "total")}  # fmt: skip
    return out


def op_rows(key_averages) -> dict[str, dict[str, float]]:
    """{op name: {"cpu_us", "cuda_us", "count"}} from torch.profiler key_averages()."""
    rows: dict[str, dict[str, float]] = {}
    for e in key_averages:
        cuda = getattr(e, "self_device_time_total", None)
        if cuda is None:
            cuda = getattr(e, "self_cuda_time_total", 0.0)
        r = rows.setdefault(e.key, {"cpu_us": 0.0, "cuda_us": 0.0, "count": 0})
        r["cpu_us"] += float(e.self_cpu_time_total)
        r["cuda_us"] += float(cuda or 0.0)
        r["count"] += int(e.count)
    return rows


def diff_ops(new: dict, repeat: dict, top: int = 15) -> list[dict[str, Any]]:
    """Ops ranked by how much more self time (CPU + CUDA) the new pass spent than the repeat."""
    names = set(new) | set(repeat)
    zero = {"cpu_us": 0.0, "cuda_us": 0.0, "count": 0}
    rows = []
    for n in names:
        a, b = new.get(n, zero), repeat.get(n, zero)
        rows.append({
            "op": n,
            "d_cpu_ms": round((a["cpu_us"] - b["cpu_us"]) / 1000, 3),
            "d_cuda_ms": round((a["cuda_us"] - b["cuda_us"]) / 1000, 3),
            "count_new": a["count"], "count_repeat": b["count"],
        })  # fmt: skip
    rows.sort(key=lambda r: r["d_cpu_ms"] + r["d_cuda_ms"], reverse=True)
    return rows[:top]


def describe(obj: Any, depth: int = 0) -> Any:
    """Types and shapes of a (nested) batch, without values: what padding would have to touch."""
    if depth > 4:
        return type(obj).__name__
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {"tensor": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): describe(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        head = [describe(v, depth + 1) for v in obj[:3]]
        return {"type": type(obj).__name__, "len": len(obj), "first": head}
    if hasattr(obj, "__dict__"):
        return {"type": type(obj).__name__,
                "fields": {k: describe(v, depth + 1) for k, v in vars(obj).items()}}  # fmt: skip
    return type(obj).__name__


def flatten_tensors(obj: Any) -> list[Any]:
    """Every tensor in a nested output, depth first, in order (lists, tuples, dict values)."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return [obj]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list | tuple):
        return [t for v in obj for t in flatten_tensors(v)]
    return []


def compare_outputs(batched: Any, singles: list[Any]) -> dict[str, Any]:
    """Per-question softmax of the batched output against the single-record outputs, in order."""
    fb = flatten_tensors(batched)
    fs = [t for s in singles for t in flatten_tensors(s)]
    same = len(fb) == len(fs) and all(
        tuple(a.shape) == tuple(b.shape) for a, b in zip(fb, fs, strict=False)
    )
    out: dict[str, Any] = {
        "comparable": same,
        "tensors_batched": len(fb),
        "tensors_single": len(fs),
    }
    if same and fb:
        d = [float((a.float().softmax(-1) - b.float().softmax(-1)).abs().max()) for a, b in
             zip(fb, fs, strict=True)]  # fmt: skip
        out["max_abs_dprob"] = round(max(d), 6)
        out["per_tensor_max_abs_dprob"] = [round(x, 6) for x in d]
    return out


def batch_preview(engine, torch, encoded: list[Any], clock=time.perf_counter) -> dict[str, Any]:
    """One batch of several records against the same records one at a time: structure, time
    (second run of each, so neither pays a first-shape cost), and agreement."""
    pad = engine.processor.tokenizer.pad_token_id

    def run(recs):
        t0 = clock()
        with torch.inference_mode():
            out = engine._forward(engine.jsm.collate_records(recs, pad, engine.device))
        torch.cuda.synchronize()
        return out, (clock() - t0) * 1000

    res: dict[str, Any] = {"lengths": [len(e.input_ids) for e in encoded]}
    try:
        with torch.inference_mode():
            res["batch"] = describe(engine.jsm.collate_records(encoded, pad, engine.device))
        run(encoded)
        batched, ms_b = run(encoded)
        singles, ms_s = [], []
        for e in encoded:
            run([e])
            o, ms = run([e])
            singles.append(o)
            ms_s.append(ms)
        res |= {"output": describe(batched), "single_output": describe(singles[0]),
                "batched_ms": round(ms_b, 2), "singles_ms": [round(x, 2) for x in ms_s],
                "singles_ms_sum": round(sum(ms_s), 2),
                **compare_outputs(batched, singles)}  # fmt: skip
    except Exception as e:  # the probe's main job does not depend on this
        res["error"] = f"{type(e).__name__}: {e}"[:500]
    return res


def blas_library(torch) -> str | None:
    """What torch will use for GEMMs (did TORCH_BLAS_PREFER_CUBLASLT take?), None if unknown."""
    try:
        return str(torch.backends.cuda.preferred_blas_library())
    except Exception:
        return None


def render(label: str, summary: dict[str, Any], diff: list[dict[str, Any]] | None,
           preview: dict[str, Any] | None = None) -> str:  # fmt: skip
    cols = (*SEGMENTS, "total")
    lines = [f"## Shape probe: {label}", "",
             "| kind | n | " + " | ".join(cols) + " | cudaMalloc |",
             "|---|---|" + "---|" * len(cols) + "---|"]  # fmt: skip
    for kind in KINDS:
        if kind in summary:
            s = summary[kind]
            cells = " | ".join(f"{s[k]:.1f}" for k in cols)
            lines.append(f"| {kind} | {s['n']} | {cells} | {s['device_allocs']} |")
    if "new_minus_repeat" in summary:
        d = summary["new_minus_repeat"]
        lines.append("| new - repeat | | " + " | ".join(f"{d[k]:+.1f}" for k in cols) + " | |")
    if diff:
        lines += ["", "Ops that grew most on a new token count (new pass minus repeat pass):", "",
                  "| op | Δ self CPU ms | Δ self CUDA ms | calls new / repeat |",
                  "|---|---|---|---|"]  # fmt: skip
        for r in diff:
            lines.append(f"| `{r['op'][:70]}` | {r['d_cpu_ms']:+.3f} | {r['d_cuda_ms']:+.3f} | "
                         f"{r['count_new']} / {r['count_repeat']} |")  # fmt: skip
    if preview:
        lines += ["", f"Batch preview, lengths {preview.get('lengths')}: "]
        if "error" in preview:
            lines[-1] += f"error: {preview['error']}"
        else:
            agree = "outputs not comparable one to one"
            if preview.get("comparable"):
                agree = f"max |Δp| batched vs single {preview['max_abs_dprob']}"
            lines[-1] += (f"batched {preview['batched_ms']} ms vs {preview['singles_ms_sum']} ms "
                          f"one at a time; {agree}")  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--record", default="pulse", help="fixture id; pulse = first Pulse record")
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--batch-records", nargs="*", default=["pulse", "card_trio", "long_2k"],
                    help="records for the batch preview; none to skip it")  # fmt: skip
    a = ap.parse_args(argv)

    import torch
    from loadtest import load_line
    from profile_engine import Splitter, backbone_chain, record_for

    from deciders.clef.engine import ClefEngine
    from worker.core import config_from_env

    engine = ClefEngine(config_from_env())
    if engine.device.type != "cuda":
        raise SystemExit("shape_probe needs a GPU")
    split = Splitter(torch, backbone_chain(engine.model))
    line = load_line(None if a.record == "pulse" else a.record, "fixtures/requests/*.jsonl")
    base = record_for(line["id"], line)
    if not isinstance(base.get("state"), str):
        raise SystemExit(f"{line['id']}: state is not a string; pick another --record")

    def with_state(s: str) -> dict:
        return {**base, "state": s}

    def count(s: str) -> int:
        return len(engine.encode(with_state(s)).input_ids)

    def allocs() -> int:
        return int(torch.cuda.memory_stats().get("num_device_alloc", 0))

    def one(kind: str, rec: dict, k: int) -> dict:
        before = allocs()
        p = split.one_pass(engine, rec)
        return {"kind": kind, "k": k, **p, "device_allocs": allocs() - before}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        batch = engine.jsm.collate_records(
            [engine.encode(base)], engine.processor.tokenizer.pad_token_id, engine.device
        )
    (a.out_dir / "batch_structure.json").write_text(json.dumps(describe(batch), indent=1))

    warm = [one("warmup", base, 0) for _ in range(a.warmup)]
    variants = make_variants(base["state"], a.variants, count)
    passes: list[dict] = []
    traced: dict[str, Any] = {}
    for i, v in enumerate(variants):
        rec = with_state(v["state"])
        if a.trace and i == len(variants) // 2:  # mid-run: past first-use costs
            from torch.profiler import ProfilerActivity, profile

            acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            for kind in ("new", "repeat"):  # profiler overhead: kept out of the medians
                with profile(activities=acts) as prof:
                    passes.append(one(f"{kind}-traced", rec, v["k"]))
                traced[kind] = op_rows(prof.key_averages())
                prof.export_chrome_trace(str(a.out_dir / f"trace-{kind}.json.gz"))
        else:
            passes.append(one("new", rec, v["k"]))
            passes.append(one("repeat", rec, v["k"]))
        if v["sibling"] is not None:
            passes.append(one("sibling", with_state(v["sibling"]), v["k"]))

    summary = summarize(passes)
    diff = diff_ops(traced["new"], traced["repeat"]) if traced else None
    preview = None
    if a.batch_records:  # last, so its new shapes cannot warm anything measured above
        recs = []
        for rid in a.batch_records:
            ln = load_line(None if rid == "pulse" else rid, "fixtures/requests/*.jsonl")
            recs.append(engine.encode(record_for(ln["id"], ln)))
        preview = batch_preview(engine, torch, recs) | {"records": a.batch_records}
        (a.out_dir / "batch_preview.json").write_text(json.dumps(preview, indent=1))
    out = {
        "engine": engine.manifest(),
        "record": line["id"],
        "base_tokens": count(base["state"]),
        "variants": [
            {k: v[k] for k in ("k", "tokens")} | {"sibling": v["sibling"] is not None}
            for v in variants
        ],  # fmt: skip
        "env": {k: os.environ.get(k) for k in ENV_KEYS} | {"preferred_blas": blas_library(torch)},
        "warmup": warm,
        "passes": passes,
        "summary": summary,
        "op_diff": diff,
        "batch_preview": preview,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (a.out_dir / "shape_probe.json").write_text(json.dumps(out, indent=1, default=str))
    text = render(a.label or a.out_dir.name, summary, diff, preview)
    (a.out_dir / "shape_probe.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
