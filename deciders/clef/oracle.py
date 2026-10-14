#!/usr/bin/env python3
"""Reference runner for Cloudflare/clef-flash: fixtures in, per-record JSON out.

Uses the checkpoint's own ``joint_schema_model.py`` (encode_record, collate_records,
load_release_model, systemone_answer) so the numbers are Cloudflare's, not ours. Batch size
is 1 on purpose: the oracle should not depend on padding.

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
        --out fixtures/clef-flash/fp32 --dtype float32 [--dry-run] [--revision SHA] [--limit N]

--dry-run encodes only (no weights): it writes the encoding block and input_tokens, which is
how token-length targets and the span gate are checked before paying for a forward pass.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Cloudflare/clef-flash"
TYPE_NAMES = {0: "noul", 1: "choice", 2: "score"}


# ---- pure helpers (importable without torch) ---------------------------------------------


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encoding_block(encoded: Any) -> dict[str, Any]:
    """Span bookkeeping for the parity gate, from an EncodedRecord-like object."""
    questions = [
        {
            "id": q.question_id,
            "type": TYPE_NAMES[q.question_type],
            "question_span": list(q.question_span),
            "option_spans": [list(s) for s in q.option_spans],
            "option_ids": list(q.option_ids),
        }
        for q in encoded.questions
    ]
    return {
        "input_tokens": len(encoded.input_ids),
        "ids_sha": sha(",".join(map(str, encoded.input_ids))),
        "spans_sha": sha(canonical(questions)),
        "questions": questions,
    }


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
        from PIL import Image

        rec["images"] = [Image.open(root / p).convert("RGB") for p in line["images"]]
    return rec


@dataclass
class Result:
    id: str
    request_sha: str
    encoding: dict[str, Any] | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] | None = None
    error: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)


# ---- the run ----------------------------------------------------------------------------


def _gdn_fast_path() -> bool:
    """Whether Qwen3.5 Gated DeltaNet runs the fla/causal-conv1d kernels or pure torch."""
    try:
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401

        return True
    except ImportError:
        return False


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def file_shas(snapshot: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in (
        "joint_head.safetensors",
        "joint_head_config.json",
        "joint_schema_model.py",
        "config.json",
    ):
        p = snapshot / name
        if p.exists():
            out[name] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
    for p in sorted(snapshot.glob("model-*.safetensors")):
        out[p.name] = {"bytes": p.stat().st_size}  # 5 GB shards: size only, revision pins them
    return out


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
    a = ap.parse_args()

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(a.model, revision=a.revision))
    sys.path.insert(0, str(snapshot))
    import joint_schema_model as jsm  # the checkpoint's own code
    import torch
    from transformers import AutoProcessor

    lines = load_requests(a.requests)
    if a.limit:
        lines = lines[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    if a.dry_run:
        model, processor = None, AutoProcessor.from_pretrained(snapshot)
    else:
        model, processor = jsm.load_release_model(
            snapshot, device=a.device, dtype=getattr(torch, a.dtype)
        )
    load_s = time.perf_counter() - t_load
    device = torch.device(a.device if not a.dry_run else "cpu")
    sdp_backends = None
    if device.type == "cuda" and a.sdp != "auto":
        from torch.nn.attention import SDPBackend

        sdp_backends = (
            [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]
            if a.sdp == "efficient"
            else [SDPBackend.MATH]
        )
    if device.type == "cuda" and a.sdp == "efficient":
        # transformers passes enable_gqa=True when the mask is None (Qwen3.5: 16 q heads, 4 kv
        # heads); torch's fused kernels then refuse ("both fused kernels require ... the same
        # num_heads") and only the math kernel is left, which materializes T×T in fp32. Forcing
        # repeat_kv makes the memory-efficient kernel eligible. Verified against transformers
        # 5.10.2's sdpa_attention.py: use_gqa_in_sdpa → False means repeat_kv, nothing else.
        import transformers.integrations.sdpa_attention as sdpa_mod

        sdpa_mod.use_gqa_in_sdpa = lambda attention_mask, key: False

    def forward(batch):
        if sdp_backends is None:
            return model(batch)[0]
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with sdpa_kernel(sdp_backends):
                return model(batch)[0]
        except RuntimeError as e:
            if "No available kernel" not in str(e) or sdp_backends == [SDPBackend.MATH]:
                raise
            with sdpa_kernel([SDPBackend.MATH]):  # last resort; OOMs at 16k in fp32
                return model(batch)[0]

    counts = {"ok": 0, "error": 0}
    for i, line in enumerate(lines, 1):
        res = Result(id=line["id"], request_sha=sha(canonical(line["request"])))
        t0 = time.perf_counter()
        try:
            rec = record_for_model(line)
            encoded = jsm.encode_record(
                processor.tokenizer, rec, max_length=a.max_length, processor=processor
            )
            res.encoding = encoding_block(encoded)
            res.timing_ms["encode"] = round((time.perf_counter() - t0) * 1000, 2)
            if model is not None:
                t1 = time.perf_counter()
                with torch.inference_mode():
                    batch = jsm.collate_records([encoded], processor.tokenizer.pad_token_id, device)
                    logits = forward(batch)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                res.timing_ms["forward"] = round((time.perf_counter() - t1) * 1000, 2)
                answers = {}
                for q, ql in zip(encoded.questions, logits, strict=True):
                    lg = ql.float().cpu()
                    probs = lg.softmax(-1).tolist()
                    res.questions.append(
                        {
                            "id": q.question_id,
                            "type": TYPE_NAMES[q.question_type],
                            "option_ids": list(q.option_ids),
                            "logits": lg.tolist(),
                            "probs": probs,
                        }
                    )
                    answers[q.question_id] = jsm.systemone_answer(
                        line["request"]["questions"][q.question_id],
                        dict(zip(q.option_ids, probs, strict=True)),
                    )
                res.response = {
                    "model": a.model,
                    "answers": answers,
                    "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
                }
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
        res.timing_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
        (a.out / f"{res.id}.json").write_text(json.dumps(asdict(res), ensure_ascii=False))
        tok = res.encoding["input_tokens"] if res.encoding else "-"
        err = f"  ERROR {res.error[:60]}" if res.error else ""
        ms = res.timing_ms["total"]
        print(
            f"[{i:>3}/{len(lines)}] {res.id:<24} tokens={tok!s:>6} {ms:>9.1f} ms{err}", flush=True
        )

    import transformers

    manifest = {
        "model": a.model,
        "revision": snapshot.name,  # snapshots/<commit sha>
        "snapshot_path": str(snapshot),
        "files": file_shas(snapshot),
        "dtype": a.dtype,
        "device": a.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dry_run": a.dry_run,
        "max_length": a.max_length,
        "sdp": a.sdp,
        "gdn_fast_path": _gdn_fast_path(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "worker_decide_commit": git_commit(),
        "load_s": round(load_s, 1),
        "counts": counts,
        "requests": a.requests,
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
