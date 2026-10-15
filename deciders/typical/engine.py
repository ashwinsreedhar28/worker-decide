"""Typical (Oz Labs) behind the SystemOne wire format: milestone 6, built ahead (Oct 15).

Interface verified against typical-ai 0.1.1 from PyPI (typical_ai/core.py, native.py, README):

- ``Typical.from_pretrained(repo_id, device="auto", filename="best.pt", revision=None,
  cache_dir=None, max_state=4096, max_states=16, max_option_tokens=24)``.
- ``Typical.decide(state, question, labels) -> (probs over exactly labels, runtime)`` takes a
  JevBench/SystemOne-style question dict ({"type", "instructions", "criteria"}) and renders
  the query itself (``query_text``: instructions plus a "label: description" rubric);
  ``runtime["p_null"]`` is the abstain probability from Typical's separate null head, outside
  the label softmax. States are KV-cached (LRU of ``max_states``), so several questions on one
  state pay the state prefill once; each question is still its own forward over the suffix,
  unlike Clef, which answers all questions in one prompt where they attend to each other.
- Labels: choice -> the criteria keys, in request order (Typical renders lettered options, so
  order is part of its input); score -> "0".."n-1" (the README's own score example, and what
  query_text's "0: .. 1: .." rubric refers to); noul -> ["no", "yes"] (the SDK's noul(),
  label-order invariant through its Bernoulli head) and the answer is p(yes).

Choices made here, not by Typical:
- A question without ``instructions`` uses its id (query_text would raise KeyError; Clef's
  reference does the same substitution).
- ``abstain`` = p_null is an extension field on every answer; the System One API has none.
- confidence = max p over the labels (as our Clef worker reports it).

Environment: typical-ai requires torch >= 2.14 and transformers >= 5.17, while Clef is pinned to
torch 2.11 / transformers 5.10.2, so this backend needs its own image; it imports typical_ai
lazily and nothing here touches the Clef worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "OzLabs/typical-medium"


def labels_for(question: dict[str, Any]) -> list[str]:
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    if kind == "noul":
        return ["no", "yes"]
    raise ValueError(f"unknown question type {kind!r}")


def typical_question(qid: str, question: dict[str, Any]) -> dict[str, Any]:
    """The question dict typical_ai's query_text expects (instructions always present)."""
    q = {"type": question["type"], "instructions": question.get("instructions") or qid}
    if question.get("criteria") is not None:
        q["criteria"] = question["criteria"]
    return q


def answer(question: dict[str, Any], probs: dict[str, float], p_null: float) -> dict[str, Any]:
    """SystemOne answer from Typical's label probabilities, plus the abstain extension."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probs["yes"], "abstain": p_null}
    labels = labels_for(question)
    p = [probs[label] for label in labels]
    top = max(range(len(p)), key=lambda i: p[i])
    if kind == "choice":
        return {
            "type": "choice",
            "choice": labels[top],
            "confidence": p[top],
            "probabilities": dict(zip(labels, p, strict=True)),
            "abstain": p_null,
        }
    return {
        "type": "score",
        "score": sum(i * x for i, x in enumerate(p)),
        "confidence": p[top],
        "legend": {str(i): c for i, c in enumerate(question["criteria"])},
        "probabilities": dict(zip(labels, p, strict=True)),
        "abstain": p_null,
    }


@dataclass
class TypicalConfig:
    model: str = DEFAULT_MODEL
    revision: str | None = None
    device: str = "auto"
    max_state: int = 4096
    max_states: int = 16


@dataclass
class TypicalDecision:
    response: dict[str, Any] | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)


class TypicalEngine:
    """Loads typical_ai.Typical (or takes a model with the same decide() for tests)."""

    def __init__(self, cfg: TypicalConfig, model: Any = None):
        self.cfg = cfg
        t0 = time.perf_counter()
        if model is None:
            from typical_ai import Typical  # lazy: needs torch >= 2.14

            model = Typical.from_pretrained(
                cfg.model,
                device=cfg.device,
                revision=cfg.revision,
                max_state=cfg.max_state,
                max_states=cfg.max_states,
            )
        self.model = model
        self.load_s = round(time.perf_counter() - t0, 1)

    def decide(self, body: dict[str, Any]) -> TypicalDecision:
        from deciders.systemone import validate_request

        problems = validate_request(body)
        if problems:
            raise ValueError("; ".join(problems))
        d = TypicalDecision()
        t0 = time.perf_counter()
        answers = {}
        for qid, q in body["questions"].items():
            probs, runtime = self.model.decide(
                body["state"], typical_question(qid, q), labels_for(q)
            )
            answers[qid] = answer(q, probs, float(runtime["p_null"]))
            d.runtime[qid] = {k: runtime[k] for k in runtime if k != "p_null"}
        d.timing_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
        d.response = {"model": self.cfg.model, "answers": answers}
        return d
