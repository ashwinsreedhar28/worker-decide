#!/usr/bin/env python3
"""Reference runner for Cloudflare/clef-flash: fixtures in, per-record JSON out.

Drives ``deciders.clef.engine.ClefEngine`` — the same object the worker serves from — over the
fixture requests, so the oracle and the worker share one code path. Batch size is 1 on
purpose: the oracle should not depend on padding.

Output, one file per record id in --out:

    {
      "id": ..., "request_sha": sha256 of the canonical request JSON,
      "encoding": {"input_tokens": T, "ids_sha": ..., "spans_sha": ...,
                   "questions": [{"id","type","question_span","option_spans","option_ids"}]},
      "questions": [{"id","type","option_ids","logits": [...], "probs": [...]}],
      "response": <SystemOne response body>,
      "error": null | str,
      "timing_ms": {"encode": .., "forward": .., "total": ..}
    }

plus --out/manifest.json (model revision, file shas, library versions, device, dtype, counts).

    python -m deciders.clef.oracle --requests fixtures/requests/*.jsonl \\
        --out fixtures/clef-flash/float32 --dtype float32 [--dry-run] [--revision SHA] [--limit N]

--dry-run encodes only (no weights): it writes the encoding block and input_tokens, which is
how token-length targets and the span gate are checked before paying for a forward pass.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deciders.clef.engine import (  # noqa: F401  (re-exported for tests and scripts)
    DEFAULT_MODEL,
    ROOT,
    ClefEngine,
    EngineConfig,
    canonical,
    encoding_block,
    request_sha,
    sha,
)


def load_requests(patterns: list[str]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                lines += [json.loads(ln) for ln in f if ln.strip()]
    ids = [ln["id"] for ln in lines]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"duplicate request ids: {sorted(dupes)}")
    return lines


def record_for_model(line: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """The dict joint_schema_model.encode_record wants: request fields plus loaded images."""
    rec = dict(line["request"])
    rec["id"] = line["id"]
    if line.get("images"):
        rec["images"] = load_images(line, root)
    return rec


def load_images(line: dict[str, Any], root: Path = ROOT) -> list[Any]:
    from PIL import Image

    return [Image.open(root / p).convert("RGB") for p in line.get("images", [])]


@dataclass
class Result:
    id: str
    request_sha: str
    encoding: dict[str, Any] | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] | None = None
    error: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--requests", nargs="+", required=True, help="jsonl files or globs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--revision", default=None, help="HF commit sha to pin; default = main, recorded"
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="encode only; no weights loaded")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument(
        "--sdp",
        default="efficient",
        choices=["efficient", "math", "auto"],
        help="SDPA backend on CUDA: efficient = memory-efficient kernel (O(T) memory, fp32 ok), "
        "math = materialize T×T (OOMs at 16k in fp32), auto = torch's choice",
    )
    ap.add_argument(
        "--exclude", default="", help="comma-separated record ids to skip (e.g. the 16k records)"
    )
    a = ap.parse_args()

    lines = load_requests(a.requests)
    skip = {s for s in a.exclude.split(",") if s}
    lines = [ln for ln in lines if ln["id"] not in skip]
    if a.limit:
        lines = lines[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)

    eng = ClefEngine(
        EngineConfig(
            model=a.model,
            revision=a.revision,
            dtype=a.dtype,
            device=a.device,
            max_length=a.max_length,
            sdp=a.sdp,
            dry_run=a.dry_run,
        )
    )
    torch = eng.torch

    counts = {"ok": 0, "error": 0}
    for i, line in enumerate(lines, 1):
        res = Result(id=line["id"], request_sha=request_sha(line["request"]))
        t0 = time.perf_counter()
        try:
            d = eng.decide(line["request"], load_images(line), rid=line["id"])
            res.encoding, res.questions, res.response = d.encoding, d.questions, d.response
            res.timing_ms.update(d.timing_ms)
            counts["ok"] += 1
        except ValueError as e:  # encode_record's own rejections (schema too long, bad question)
            res.error = str(e)
            counts["error"] += 1
        except torch.OutOfMemoryError as e:  # recorded, not fatal: the run must finish and push
            res.error = "cuda oom: " + str(e).split(".")[0]
            counts["error"] += 1
            torch.cuda.empty_cache()
        except RuntimeError as e:  # e.g. "No available kernel" from a restricted SDPA backend set
            res.error = "runtime: " + str(e)[:200]
            counts["error"] += 1
            if eng.device.type == "cuda":
                torch.cuda.empty_cache()
        res.timing_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
        (a.out / f"{res.id}.json").write_text(json.dumps(asdict(res), ensure_ascii=False))
        tok = res.encoding["input_tokens"] if res.encoding else "-"
        err = f"  ERROR {res.error[:60]}" if res.error else ""
        ms = res.timing_ms["total"]
        print(
            f"[{i:>3}/{len(lines)}] {res.id:<24} tokens={tok!s:>6} {ms:>9.1f} ms{err}", flush=True
        )

    manifest = {
        **eng.manifest(),
        "counts": counts,
        "requests": a.requests,
        "excluded": sorted(skip),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {k: manifest[k] for k in ("revision", "dtype", "gpu", "versions", "counts", "load_s")}
        )
    )


if __name__ == "__main__":
    main()
