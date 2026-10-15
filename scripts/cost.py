#!/usr/bin/env python3
"""Cost per 1,000 requests from load-test results, next to Workers AI's price (milestone 5b).

Steady-state model: a flex worker billed per second while it runs, kept busy at the measured
throughput, costs  price_per_second / req_per_s  per request (one request may carry several
questions; both sides are priced per request). That is the floor for a busy worker; real
traffic also pays the billed parts of a cold start (container start and model load; per
Runpod's docs the image pull and the cached-model download are not billed) and the idle
timeout, which this does not include (milestone 3 has the cold-start numbers). Cells with
errors or no answers are not priced. Workers AI charges per input token (output
tokens are free for Clef), so a decision costs input_tokens x price per token; M1 found its
prompt_tokens equal to our encoder's input_tokens on all 50 Pulse records, so the same count
is used for both.

    python scripts/cost.py runs/load-ours-a40/load.json [more load.json ...] --gpu "NVIDIA A40"
        [--prices fixtures/prices/2026-10-15.json]

--gpu names the GPU each file was measured on (one per file, or one for all); the tier and
hourly price come from the prices file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICES = ROOT / "fixtures/prices/2026-10-15.json"


def runpod_usd_per_1k(usd_per_hour: float, req_per_s: float) -> float | None:
    if not req_per_s:
        return None
    return usd_per_hour / 3600 / req_per_s * 1000


def workers_ai_usd_per_1k(input_tokens: int, usd_per_million: float) -> float:
    return input_tokens * usd_per_million / 1e6 * 1000


def rows(load: dict[str, Any], gpu: str, prices: dict[str, Any]) -> list[dict[str, Any]]:
    tier = prices["gpu_tier"].get(gpu)
    if tier is None:
        raise SystemExit(f"no price tier for GPU {gpu!r}; add it to the prices file")
    hourly = prices["runpod_serverless_flex_usd_per_hour"][tier]
    wai = prices["workers_ai_clef_flash"]["usd_per_million_input_tokens"]
    out = []
    for c in load["cells"]:
        s, tokens = c["stats"], c.get("input_tokens")
        clean = s.get("n", 0) > 0 and not s.get("errors")
        rp = runpod_usd_per_1k(hourly, s.get("req_per_s") or 0.0) if clean else None
        hosted = workers_ai_usd_per_1k(tokens, wai) if tokens else None
        out.append(
            {
                "label": load.get("label"),
                "gpu": gpu,
                "tier": tier,
                "usd_per_hour": hourly,
                "workload": c["workload"],
                "input_tokens": tokens,
                "concurrency": c["concurrency"],
                "req_per_s": s.get("req_per_s"),
                "p50_ms": s.get("p50_ms"),
                "runpod_usd_per_1k": rp,
                "workers_ai_usd_per_1k": hosted,
                "runpod_over_workers_ai": (rp / hosted) if rp and hosted else None,
                "excluded": None
                if clean
                else f"{s.get('errors', 0)} errors, {s.get('n', 0)} answers",
            }
        )
    return out


def render(all_rows: list[dict[str, Any]], prices: dict[str, Any]) -> str:
    wai = prices["workers_ai_clef_flash"]["usd_per_million_input_tokens"]
    lines = [
        f"## Cost per 1,000 requests (Runpod flex, steady state; Workers AI ${wai}/M input)",
        "",
        "| run | GPU | workload | tokens | concurrency | req/s | Runpod $/1k | Workers AI $/1k "
        "| Runpod / Workers AI |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def money(x):
        return "-" if x is None else f"${x:.4f}"

    for r in all_rows:
        ratio = r["runpod_over_workers_ai"]
        lines.append(
            f"| {r['label']} | {r['gpu']} ({r['tier']}, ${r['usd_per_hour']}/h) | {r['workload']} "
            f"| {r['input_tokens']} | {r['concurrency']} | {r['req_per_s']} "
            f"| {money(r['runpod_usd_per_1k'])} | {money(r['workers_ai_usd_per_1k'])} "
            f"| {'-' if ratio is None else f'{ratio:.2f}x'} |"
            + (f" not priced: {r['excluded']}" if r["excluded"] else "")
        )
    lines += [
        "",
        "Steady state: the worker is kept busy at the measured throughput. The billed parts of "
        "cold starts and the idle timeout are not included.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("loads", nargs="+", type=Path)
    ap.add_argument("--gpu", action="append", required=True)
    ap.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if len(a.gpu) not in (1, len(a.loads)):
        raise SystemExit("--gpu once for all files, or once per file")
    prices = json.loads(a.prices.read_text())
    gpus = a.gpu * len(a.loads) if len(a.gpu) == 1 else a.gpu
    all_rows = []
    for path, gpu in zip(a.loads, gpus, strict=True):
        all_rows += rows(json.loads(path.read_text()), gpu, prices)
    if a.json:
        a.json.write_text(json.dumps(all_rows, indent=1))
    print(render(all_rows, prices))


if __name__ == "__main__":
    main()
