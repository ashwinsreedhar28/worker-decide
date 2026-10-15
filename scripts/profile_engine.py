#!/usr/bin/env python3
"""Where a Clef-flash request's time goes, in-process: encode, backbone, head, the rest.

Milestone 5a. Loads ClefEngine (same config as the worker), then for each fixture record runs
warm-up passes and N timed passes of the worker's own request path, split by CUDA events:

    encode    tokenizer + encode_record (CPU)
    collate   collate_records onto the device
    pre       model call start -> backbone start (vision tower / embedding prep, if outside it)
    backbone  the backbone module's forward (Qwen3.5-9B prefill)
    head      backbone end -> model call end (joint schema head, its Python loops included)
    post      logits -> float -> CPU -> softmax -> lists (what forward() does after the call)

The backbone is found at runtime (the model code is downloaded with the weights, so its
attribute names are not hard-coded): every module on the largest-parameter chain from the root
is hooked, and the outermost one that actually ran is timed. The chain, the module used and the
top-level module tree with parameter counts are written out so a wrong pick is visible, and
--backbone overrides it. Events are recorded on the current stream around Python calls, so
"head" includes GPU idle time while Python launches the head's small kernels, where that time
is not hidden behind queued backbone work.

What it does not time: image decoding (done once per record, before the passes), the worker's
encoding_block / assemble / systemone_answer, and HTTP (loadtest.py measures end to end).
Shares depend on the backbone path: our image runs Qwen3.5's Gated DeltaNet layers in pure
torch (gdn_fast_path=False, printed in the header), SGLang uses Triton kernels, so the
backbone share here is higher, and head + overhead lower, than on a faster backbone.

    python scripts/profile_engine.py runs/profile-a40 [--records card_trio pulse long_2k]
        [--reps 20] [--warmup 3] [--trace]

--trace also runs one pass per record under torch.profiler and writes a gzipped Chrome trace and
the top kernels by CUDA time. Needs a GPU and the model environment (requirements-oracle.txt).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SEGMENTS = ("encode", "collate", "pre", "backbone", "head", "post")


def param_count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def module_tree(model, depth: int = 2, prefix: str = "") -> list[dict[str, Any]]:
    """Children down to `depth`, with parameter counts: the record of what was split."""
    out = []
    for name, child in model.named_children():
        path = f"{prefix}{name}"
        out.append({"path": path, "type": type(child).__name__, "params": param_count(child)})
        if depth > 1:
            out.extend(module_tree(child, depth - 1, path + "."))
    return out


def _layer_span(mod) -> tuple[str, Any, Any] | None:
    """A decoder-layer list under `mod` (depth <= 2): its name, first and last layer."""
    for name, child in mod.named_children():
        if type(child).__name__ in ("ModuleList", "Sequential"):
            layers = [m for _, m in child.named_children()]
            if layers:
                return name, layers[0], layers[-1]
    for name, child in mod.named_children():
        for gname, grand in child.named_children():
            if type(grand).__name__ in ("ModuleList", "Sequential"):
                layers = [m for _, m in grand.named_children()]
                if layers:
                    return f"{name}.{gname}", layers[0], layers[-1]
    return None


def backbone_chain(model, override: str | None = None) -> list[tuple[str, Any, Any]]:
    """Candidate backbone spans (path, start module, end module), outermost first: from the
    root, repeatedly the child with the most parameters (the 9B backbone next to a 122M head),
    while it still holds more than half of all parameters, stopping above a layer list. Which
    of them actually runs depends on how the model code calls its backbone (a wrapper may call
    the inner language model directly, never the outer module's forward), so all are hooked
    and the outermost one that ran is used. Last, a span from the first to the last decoder
    layer, which runs however the backbone is entered (it leaves out the embedding and the
    final norm, so it slightly under-counts). --backbone replaces the chain with one path."""
    if override:
        mod = model
        for part in override.split("."):
            mod = getattr(mod, part)
        return [(override, mod, mod)]
    chain, path, mod = [], "", model
    total = param_count(model)
    for _ in range(5):
        kids = list(mod.named_children())
        if not kids:
            break
        name, best = max(kids, key=lambda kv: param_count(kv[1]))
        if param_count(best) <= 0.5 * total or type(best).__name__ in ("ModuleList", "Sequential"):
            break
        path, mod = f"{path}.{name}" if path else name, best
        chain.append((path, mod, mod))
    if not chain:
        raise SystemExit("could not find a backbone child; pass --backbone")
    span = _layer_span(mod)
    if span:
        name, first, last = span
        chain.append((f"{path}.{name}[first..last]", first, last))
    return chain


def aggregate(samples: list[dict[str, float]]) -> dict[str, Any]:
    """Median per segment over passes, and each segment's share of the median total."""
    med = {k: round(statistics.median(s[k] for s in samples), 3) for k in (*SEGMENTS, "total")}
    total = med["total"] or 1.0
    share = {k: round(med[k] / total, 3) for k in SEGMENTS}
    return {"passes": len(samples), "median_ms": med, "share_of_total": share,
            "head_plus_overhead_share": round(1 - share["backbone"], 3)}  # fmt: skip


def outermost_fired(chain_paths: list[str], fired: set[str]) -> str | None:
    """The outermost hooked module that ran both its pre- and post-hook in this pass."""
    return next((p for p in chain_paths if p in fired), None)


class Splitter:
    """CUDA events at model-call start/end and around each candidate backbone's forward."""

    def __init__(self, torch, chain: list[tuple[str, Any, Any]]):
        self.torch = torch
        self.paths = [p for p, _, _ in chain]
        self.events: dict[str, Any] = {}
        for path, start, end in chain:
            # Default args bind each path at definition time.
            start.register_forward_pre_hook(lambda m, a, p=path: self._mark(f"{p}:start"))
            end.register_forward_hook(lambda m, a, o, p=path: self._mark(f"{p}:end"))

    def _mark(self, name: str) -> None:
        if name in self.events:  # a module called twice per pass: keep the first span
            return
        ev = self.torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events[name] = ev

    def one_pass(self, engine, record: dict) -> dict[str, Any]:
        torch = self.torch
        self.events = {}
        t0 = time.perf_counter()
        encoded = engine.encode(record)
        t1 = time.perf_counter()
        with torch.inference_mode():
            batch = engine.jsm.collate_records(
                [encoded], engine.processor.tokenizer.pad_token_id, engine.device
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            self._mark("call:start")
            logits = engine._forward(batch)
            self._mark("call:end")
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        for ql in logits:
            lg = ql.float().cpu()
            lg.tolist(), lg.softmax(-1).tolist()
        t4 = time.perf_counter()
        ev = self.events
        fired = {p for p in self.paths if f"{p}:start" in ev and f"{p}:end" in ev}
        bb = outermost_fired(self.paths, fired)
        if bb is None:
            raise SystemExit(f"no candidate backbone ran ({self.paths}); pass --backbone")
        return {
            "encode": (t1 - t0) * 1000,
            "collate": (t2 - t1) * 1000,
            "pre": ev["call:start"].elapsed_time(ev[f"{bb}:start"]),
            "backbone": ev[f"{bb}:start"].elapsed_time(ev[f"{bb}:end"]),
            "head": ev[f"{bb}:end"].elapsed_time(ev["call:end"]),
            "post": (t4 - t3) * 1000,
            "total": (t4 - t0) * 1000,
            "input_tokens": len(encoded.input_ids),
            "backbone_path": bb,
        }


def record_for(engine_rid: str, line: dict) -> dict:
    """The record ClefEngine.decide builds. Images go through the worker's own data-URL decoder
    (once, outside the timed passes), so the model sees what it sees over HTTP."""
    from parity_http import data_url

    from deciders.clef.engine import images_from_data_urls

    rec = {k: v for k, v in line["request"].items() if k != "images"}
    rec["id"] = engine_rid
    if line.get("images"):
        rec["images"] = images_from_data_urls([data_url(ROOT / p) for p in line["images"]])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--records", nargs="+", default=["card_trio", "pulse", "long_2k"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--backbone", default=None, help="dotted module path, overrides the pick")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()

    import torch
    from loadtest import load_line

    from deciders.clef.engine import ClefEngine
    from worker.core import config_from_env

    engine = ClefEngine(config_from_env())
    if engine.device.type != "cuda":
        raise SystemExit("profile_engine needs a GPU")
    chain = backbone_chain(engine.model, a.backbone)
    split = Splitter(torch, chain)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for rid in a.records:
        line = load_line(None if rid == "pulse" else rid, "fixtures/requests/*.jsonl")
        rec = record_for(line["id"], line)
        for _ in range(a.warmup):
            split.one_pass(engine, rec)
        samples = [split.one_pass(engine, rec) for _ in range(a.reps)]
        results[line["id"]] = {"input_tokens": samples[0]["input_tokens"],
                               "backbone_path": samples[0]["backbone_path"],
                               **aggregate(samples)}  # fmt: skip
        print(f"[profile] {line['id']}: {results[line['id']]}", file=sys.stderr, flush=True)
        if a.trace:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                split.one_pass(engine, rec)
            # .gz: torch gzips by suffix; an uncompressed trace of the 2k record can approach
            # GitHub's 100 MB file limit and fail the whole push (Oct 15 audit).
            prof.export_chrome_trace(str(a.out_dir / f"trace-{line['id']}.json.gz"))
            (a.out_dir / f"kernels-{line['id']}.txt").write_text(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
            )
    used = {r["backbone_path"] for r in results.values()}
    path = used.pop() if len(used) == 1 else ", ".join(sorted(used))
    # The layer-span candidate's start module is one decoder layer; count the module holding
    # the layers instead, i.e. the chain entry before it (Oct 15 re-audit).
    plain = [(p, m) for p, m, e in chain if m is e]
    backbone = next((m for p, m in plain if p == path), plain[-1][1] if plain else chain[0][1])
    out = {
        "engine": engine.manifest(),
        "backbone_chain": [p for p, _, _ in chain],
        "backbone_path": path,
        "backbone_params": param_count(backbone),
        "model_params": param_count(engine.model),
        "module_tree": module_tree(engine.model),
        "reps": a.reps,
        "warmup": a.warmup,
        "records": results,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (a.out_dir / "profile.json").write_text(json.dumps(out, indent=1, default=str))
    rows = ["| record | tokens | " + " | ".join(SEGMENTS) + " | total ms | backbone share |",
            "|---|---|" + "---|" * len(SEGMENTS) + "---|---|"]  # fmt: skip
    for rid, r in results.items():
        m = r["median_ms"]
        cells = " | ".join(f"{m[k]:.1f}" for k in SEGMENTS)
        share = r["share_of_total"]["backbone"]
        rows.append(f"| {rid} | {r['input_tokens']} | {cells} | {m['total']:.1f} | {share:.0%} |")
    params = f"{out['backbone_params'] / 1e9:.2f}B of {out['model_params'] / 1e9:.2f}B params"
    text = (
        f"## Profile ({out['engine'].get('gpu')}, backbone = `{path}`, {params}, "
        f"gdn_fast_path={out['engine'].get('gdn_fast_path')})\n\n" + "\n".join(rows)
    )
    (a.out_dir / "profile.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
