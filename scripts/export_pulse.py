#!/usr/bin/env python3
"""Export Pulse's 50-article SystemOne requests (and the hosted Clef-flash responses) as fixtures.

The request bodies are built by Pulse's own ``bench/quality.py`` (``systemone_body`` +
``render_user``), so they are byte-identical to what Brian's table sent to Jev,
Workers AI and the Typical shim on Oct 13, 2026. Nothing is re-rendered here.

Usage:
    python scripts/export_pulse.py --pulse-bench ~/Pulse/bench \
        --requests fixtures/requests/pulse50.jsonl \
        --hosted fixtures/hosted/clef-flash-workers-ai-2026-10-13.jsonl \
        --hosted-batch cad64126

Request line:  {"id": "pulse-<article_id>", "source": {...}, "request": <SystemOne body>}
Hosted line:   {"id": "pulse-<article_id>", "probabilities": [...], "confidence", "expected",
                "prompt_tokens", "wall_ms", "served_model", "batch_id"}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HOSTED_BACKEND = "clef"


def load_quality(bench: Path):
    sys.path.insert(0, str(bench))
    import quality  # noqa: PLC0415  (Pulse's bench/quality.py)

    return quality


def export_requests(quality, bench: Path, out: Path) -> tuple[int, str]:
    raw = (bench / "articles_50.json").read_bytes()
    fx = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()[:12]  # same rule as quality.load_fixture
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for art in fx["articles"]:
            body = quality.systemone_body(
                None, fx["system_prompt"][art["domain"]], quality.render_user(art)
            )
            line = {
                "id": f"pulse-{art['id']}",
                "source": {
                    "pulse_commit": fx.get("pulse_commit"),
                    "fixture_sha": sha,
                    "domain": art["domain"],
                    "levels": "bare",
                },
                "request": body,
            }
            f.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n, sha


def export_hosted(bench: Path, out: Path, batch_id: str, fixture_sha: str) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((bench / "quality.csv").open()))
    picked = [r for r in rows if r["backend"] == HOSTED_BACKEND and r["batch_id"] == batch_id]
    if not picked:
        sys.exit(f"no rows for backend={HOSTED_BACKEND} batch_id={batch_id}")
    bad = [r for r in picked if r["fixture_sha"] != fixture_sha]
    if bad:
        sys.exit(f"{len(bad)} hosted rows have fixture_sha != {fixture_sha}")
    n = 0
    with out.open("w") as f:
        for r in picked:
            extra = json.loads(r["extra"] or "{}")
            line = {
                "id": f"pulse-{r['article_id']}",
                "question": "urgency",
                "probabilities": extra.get("probabilities"),
                "confidence": extra.get("confidence"),
                "expected": extra.get("expected"),
                "prompt_tokens": int(r["prompt_tokens"]) if r["prompt_tokens"] else None,
                "wall_ms": int(r["wall_ms"]) if r["wall_ms"] else None,
                "served_model": extra.get("served_model"),
                "model": r["model"],
                "batch_id": r["batch_id"],
                "ts_utc": r["ts_utc"],
            }
            f.write(json.dumps(line, sort_keys=True) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pulse-bench", required=True, type=Path)
    ap.add_argument("--requests", required=True, type=Path)
    ap.add_argument("--hosted", type=Path)
    ap.add_argument("--hosted-batch", help="quality.csv batch_id of the hosted clef run")
    a = ap.parse_args()
    quality = load_quality(a.pulse_bench)
    n, sha = export_requests(quality, a.pulse_bench, a.requests)
    print(f"requests: {n} → {a.requests} (fixture_sha {sha})")
    if a.hosted:
        if not a.hosted_batch:
            sys.exit("--hosted needs --hosted-batch")
        m = export_hosted(a.pulse_bench, a.hosted, a.hosted_batch, sha)
        print(f"hosted:   {m} → {a.hosted} (batch {a.hosted_batch})")


if __name__ == "__main__":
    main()
