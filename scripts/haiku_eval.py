#!/usr/bin/env python3
"""Score a run directory against the Haiku reference labels, the way Pulse's bench harness does.

Pulse's `bench/quality.py summary` scores urgency 1-5 per article and compares every backend to
Claude Haiku's answers: exact agreement, within-1, MAE, and precision/recall on "urgent"
(score >= 4, Pulse's notification threshold). This reproduces that table for any run directory
in the oracle schema (fixtures/clef-flash/<dtype>, runs/hosted, a worker run), so the quality
numbers in REVIEW.md are checkable from this repo alone.

Two decoders, as in the harness: `argmax` (option index + 1) and `ev` (round of the
probability-weighted level, clamped to 1-5).

    python scripts/haiku_eval.py fixtures/clef-flash/float32 fixtures/clef-flash/bfloat16 \\
        runs/hosted
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "fixtures" / "labels" / "haiku-pulse50.json"


def decode(probs: list[float]) -> dict[str, int]:
    """Both decoders for one score question; levels are 1-based like Pulse's scale."""
    argmax = max(range(len(probs)), key=lambda i: probs[i]) + 1
    expected = sum((i + 1) * p for i, p in enumerate(probs))
    return {"argmax": argmax, "ev": min(len(probs), max(1, int(expected + 0.5)))}


def evaluate(scores: dict[str, int], labels: dict[str, int], urgent: int = 4) -> dict[str, float]:
    pairs = [(scores[k], labels[k]) for k in labels if k in scores]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    tp = sum(1 for p, q in pairs if p >= urgent and q >= urgent)
    fp = sum(1 for p, q in pairs if p >= urgent and q < urgent)
    fn = sum(1 for p, q in pairs if p < urgent and q >= urgent)
    return {
        "n": n,
        "exact": sum(1 for p, q in pairs if p == q) / n,
        "within1": sum(1 for p, q in pairs if abs(p - q) <= 1) / n,
        "mae": sum(abs(p - q) for p, q in pairs) / n,
        "prec": tp / (tp + fp) if tp + fp else math.nan,
        "rec": tp / (tp + fn) if tp + fn else math.nan,
        "flagged": tp + fp,
        "ref_urgent": tp + fn,
    }


def scores_from_run(run_dir: Path, ids: list[str], question: str) -> dict[str, dict[str, int]]:
    """id -> {decoder: level} for every labelled record the run answered."""
    out: dict[str, dict[str, int]] = {}
    for rid in ids:
        p = run_dir / f"{rid}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        q = next((q for q in rec.get("questions", []) if q["id"] == question), None)
        if q is not None and not rec.get("error"):
            out[rid] = decode(q["probs"])
    return out


def _f(x: float, pc: bool = False) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "-"
    return f"{x * 100:.0f}%" if pc else f"{x:.2f}"


def render(rows: list[tuple[str, str, dict[str, float]]], labels_meta: dict) -> str:
    lines = [
        f"Reference: {labels_meta['reference_model']}, {len(labels_meta['labels'])} labels, "
        f"{sum(1 for v in labels_meta['labels'].values() if v >= labels_meta['urgent_threshold'])}"
        f" urgent (>= {labels_meta['urgent_threshold']}).",
        "",
        "| run | decoder | n | exact | within 1 | MAE | urgent P | urgent R | flagged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run, dec, m in rows:
        if m["n"] == 0:
            lines.append(f"| {run} | {dec} | 0 | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {run} | {dec} | {m['n']} | {_f(m['exact'], True)} | {_f(m['within1'], True)} | "
            f"{_f(m['mae'])} | {_f(m['prec'], True)} | {_f(m['rec'], True)} | {m['flagged']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--question", default="urgency")
    a = ap.parse_args()
    meta = json.loads(a.labels.read_text())
    labels = {k: int(v) for k, v in meta["labels"].items()}
    rows = []
    for run_dir in a.run_dirs:
        scored = scores_from_run(run_dir, list(labels), a.question)
        for dec in ("argmax", "ev"):
            m = evaluate({k: v[dec] for k, v in scored.items()}, labels, meta["urgent_threshold"])
            rows.append((str(run_dir), dec, m))
    print(render(rows, meta))


if __name__ == "__main__":
    main()
