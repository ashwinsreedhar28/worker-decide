#!/usr/bin/env python3
"""What option and question order do to Clef-flash's answers (fixtures/requests-order).

Compares the order records' oracle against the committed key-sorted fixtures:
- order_choice_unsorted vs unsorted_choice: the encoder sorts choice ids, so the token ids
  should be identical and the probabilities equal to float noise.
- order_questions_as_card / _reversed vs card_trio (alphabetical on disk): the same request in
  three question orders; the max |dp| per question is how much order moves the answer.

    python scripts/order_findings.py fixtures/clef-flash/float32 fixtures/clef-flash/float32-order
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _probs(rec: dict) -> dict[str, list[float]]:
    return {q["id"]: q["probs"] for q in rec["questions"]}


def _maxdp(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def findings(base: Path, order: Path) -> dict:
    u, o = _load(base / "unsorted_choice.json"), _load(order / "order_choice_unsorted.json")
    out = {
        "unsorted_same_ids": u["encoding"]["ids_sha"] == o["encoding"]["ids_sha"],
        "unsorted_max_dp": _maxdp(_probs(u)["team"], _probs(o)["team"]),
        "question_orders": {},
    }
    ref = _probs(_load(base / "card_trio.json"))
    for name, path in [
        ("alphabetical", base / "card_trio.json"),
        ("card", order / "order_questions_as_card.json"),
        ("reversed", order / "order_questions_reversed.json"),
    ]:
        r = _load(path)
        p = _probs(r)
        out["question_orders"][name] = {
            "input_tokens": r["encoding"]["input_tokens"],
            "max_dp_vs_alphabetical": {k: round(_maxdp(p[k], ref[k]), 6) for k in ref},
            "probs": {k: [round(x, 4) for x in v] for k, v in p.items()},
        }
    return out


def render(f: dict) -> str:
    lines = [
        "## Order findings",
        "",
        f"- Options out of order on the wire: token ids "
        f"{'identical' if f['unsorted_same_ids'] else 'DIFFERENT'} to the sorted fixture, "
        f"max |dp| {f['unsorted_max_dp']:.2e}.",
        "",
        "| question order | tokens | max \\|dp\\| vs alphabetical, per question |",
        "|---|---|---|",
    ]
    for name, q in f["question_orders"].items():
        dps = ", ".join(f"{k} {v:.4f}" for k, v in q["max_dp_vs_alphabetical"].items())
        lines.append(f"| {name} | {q['input_tokens']} | {dps} |")
    return "\n".join(lines)


def main() -> None:
    base, order = Path(sys.argv[1]), Path(sys.argv[2])
    f = findings(base, order)
    (order / "findings.json").write_text(json.dumps(f, indent=1))
    print(render(f))


if __name__ == "__main__":
    main()
