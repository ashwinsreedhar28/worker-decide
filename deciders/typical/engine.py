"""Typical (Oz Labs) behind the SystemOne wire format: milestone 6, built ahead (Oct 15).

Interface verified against typical-ai 0.1.1 from PyPI (typical_ai/core.py, native.py, README):

- ``Typical.from_pretrained(repo_id, device="auto", filename="best.pt", revision=None,
  cache_dir=None, max_state=4096, max_states=16, max_option_tokens=24)``.
- ``Typical.decide(state, question, labels) -> (probs over exactly labels, runtime)`` takes a
  JevBench/SystemOne-style question dict ({"type", "instructions", "criteria"}) and renders the
  query itself (``query_text``: instructions, then a rubric: "yes: .. no: .." for noul from the
  criteria's true/false, "i: description" for a list, "key: description" for a mapping).
  ``runtime["p_null"]`` comes from Typical's separate null head; the label probabilities are
  renormalized without it, so confidence is conditional on not abstaining. States are
  KV-cached (LRU of ``max_states``): several questions on one state pay its prefill once, but
  each question is its own forward over the suffix, unlike Clef, whose questions share one
  prompt and attend to each other.
- Labels: choice -> the criteria keys in request order (Typical renders lettered options, so
  order is part of its input; Clef sorts ids); noul -> ["no", "yes"], answered as p(yes) (the
  SDK's noul(); label-order invariant only for checkpoints with noul_head="bern").
- Score has two label modes, because the SDK's two entry points disagree for lists like
  Pulse's ["1".."5"]: ``index`` (default) passes the criteria to query_text and labels
  "0".."n-1", which is what its "i: description" rubric refers to (Pulse then reads options
  0..4 next to instructions that say 1 to 5); ``criteria`` labels the levels by their own text
  and leaves the rubric out, which is the SDK's score(state, question, levels). Which one
  Brian's table uses is a methodology choice to state, not a default to hide.

Choices made here, not by Typical:
- A question without instructions uses its id (query_text raises KeyError otherwise; Clef's
  reference does the same substitution). Null noul criteria are dropped so query_text's own
  defaults ("Yes"/"No") apply instead of rendering "None".
- ``abstain`` = p_null on choice and score answers, an extension field (the System One API has
  none). Not on noul: the SDK's noul() discards p_null and its meaning on the Bernoulli route
  is unvalidated.
- confidence = max p over the labels, score = sum i * p, legend keyed "0".."n-1": as the Clef
  worker reports them. Probabilities are not rounded.
- Dict states are left to typical_ai, which dumps them with json.dumps(ensure_ascii=False) as
  its training decider did; Clef's canonical sorted, compact rendering is not imposed.
- Images are refused (ValueError -> 400): Typical's checkpoints are text-only.
- Each question records query_tokens and whether the query exceeds the trained length (64),
  which typical_ai reports as always False; state truncation (max_state, default 4096 vs Clef's
  16,384) is recorded too.

Environment: typical-ai requires torch >= 2.14 and transformers >= 5.17; Clef is pinned to
torch 2.11 / transformers 5.10.2, so this backend needs its own image. typical_ai is imported
lazily and nothing here touches the Clef worker. The engine matches ClefEngine's interface
(decide(body, images, rid), manifest(), a lock), so WorkerCore can serve it unchanged.
"""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from typing import Any

from deciders.clef.engine import Decision, git_commit, request_sha

DEFAULT_MODEL = "OzLabs/typical-medium"
TRAINED_MAX_QUERY = 64  # typical_ai.core.MAX_QUERY
SCORE_LABEL_MODES = ("index", "criteria")


def labels_for(question: dict[str, Any], score_labels: str = "index") -> list[str]:
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        if score_labels == "criteria":
            labels = [str(c) for c in question["criteria"]]
            if len(set(labels)) != len(labels):
                raise ValueError("score levels must be distinct to be used as labels")
            return labels
        return [str(i) for i in range(len(question["criteria"]))]
    if kind == "noul":
        return ["no", "yes"]
    raise ValueError(f"unknown question type {kind!r}")


def typical_question(qid: str, question: dict[str, Any], score_labels: str = "index") -> dict:
    """The question dict typical_ai's query_text expects (instructions always present)."""
    q: dict[str, Any] = {
        "type": question["type"],
        "instructions": question.get("instructions") or qid,
    }
    crit = question.get("criteria")
    if question["type"] == "noul" and isinstance(crit, dict):
        crit = {k: v for k, v in crit.items() if v is not None} or None
    if question["type"] == "score" and score_labels == "criteria":
        crit = None  # the labels carry the levels; an "i: level" rubric would contradict them
    if crit is not None:
        q["criteria"] = crit
    return q


def answer(question: dict[str, Any], labels: list[str], probs: dict[str, float],
           p_null: float) -> dict[str, Any]:  # fmt: skip
    """SystemOne answer from Typical's label probabilities (+ abstain on choice and score)."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probs["yes"]}
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
        "probabilities": {str(i): x for i, x in enumerate(p)},
        "abstain": p_null,
    }


@dataclass
class TypicalConfig:
    model: str = DEFAULT_MODEL
    revision: str | None = None  # pin a commit for anything published
    device: str = "auto"
    max_state: int = 4096
    max_states: int = 16
    max_option_tokens: int | None = 24  # the SDK default; None disables its "suffix diet"
    score_labels: str = "index"


class TypicalEngine:
    """typical_ai.Typical behind ClefEngine's interface (or a model with the same decide())."""

    def __init__(self, cfg: TypicalConfig, model: Any = None):
        if cfg.score_labels not in SCORE_LABEL_MODES:
            raise ValueError(f"score_labels must be one of {SCORE_LABEL_MODES}")
        self.cfg = cfg
        self.lock = threading.Lock()  # typical_ai's state cache is a plain OrderedDict
        t0 = time.perf_counter()
        if model is None:
            from typical_ai import Typical  # lazy: needs torch >= 2.14

            model = Typical.from_pretrained(
                cfg.model,
                device=cfg.device,
                revision=cfg.revision,
                max_state=cfg.max_state,
                max_states=cfg.max_states,
                max_option_tokens=cfg.max_option_tokens,
            )
        self.model = model
        self.load_s = round(time.perf_counter() - t0, 1)

    def head_config(self) -> dict[str, Any]:
        """The checkpoint's head settings that change what noul and p_null mean."""
        head = getattr(self.model, "model", None)
        return {k: getattr(head, k, None) for k in ("nc_head", "null", "render", "score_head",
                                                     "noul_head")}  # fmt: skip

    def manifest(self) -> dict[str, Any]:
        versions = {"python": platform.python_version()}
        try:
            import importlib.metadata as md

            for pkg in ("typical-ai", "torch", "transformers"):
                versions[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001 — absent in tests
            pass
        return {
            "model": self.cfg.model,
            "revision": self.cfg.revision,
            "device": str(getattr(self.model, "device", self.cfg.device)),
            "load_s": self.load_s,
            "versions": versions,
            "worker_decide_commit": git_commit(),
            "decider": "typical",
            "score_labels": self.cfg.score_labels,
            "max_state": self.cfg.max_state,
            "max_option_tokens": self.cfg.max_option_tokens,
            "head": self.head_config(),
        }

    def decide(self, body: dict[str, Any], images: list[Any] | None = None,
               rid: str = "") -> Decision:  # fmt: skip
        from deciders.systemone import validate_request

        if images or body.get("images"):
            raise ValueError("this decider is text-only; images are not supported")
        problems = validate_request(body)
        if problems:
            raise ValueError("; ".join(problems))
        d = Decision(request_sha=request_sha(body))
        t0 = time.perf_counter()
        answers, state_tokens, query_tokens = {}, 0, 0
        with self.lock:
            for qid, q in body["questions"].items():
                labels = labels_for(q, self.cfg.score_labels)
                try:
                    probs, runtime = self.model.decide(
                        body["state"], typical_question(qid, q, self.cfg.score_labels), labels
                    )
                except AssertionError as e:  # native.py asserts suffixes <= 1024 tokens
                    raise ValueError(f"question {qid!r} is too long for this decider: {e}") from e
                p_null = float(runtime["p_null"])
                answers[qid] = answer(q, labels, probs, p_null)
                qt = runtime.get("query_tokens")
                d.questions.append(
                    {
                        "id": qid,
                        "type": q["type"],
                        "option_ids": labels,
                        "probs": [probs[label] for label in labels],
                        "p_null": p_null,
                        "query_tokens": qt,
                        "query_beyond_train_len": qt is not None and qt > TRAINED_MAX_QUERY,
                        "state_tokens": runtime.get("state_tokens"),
                        "state_truncated": runtime.get("state_truncated"),
                    }
                )
                state_tokens = max(state_tokens, runtime.get("state_tokens") or 0)
                query_tokens += qt or 0
        d.timing_ms["total"] = round((time.perf_counter() - t0) * 1000, 2)
        d.response = {
            "model": self.cfg.model,
            "answers": answers,
            # The state is prefilled once (KV-cached), each question's suffix once.
            "usage": {"input_tokens": state_tokens + query_tokens, "output_tokens": 0},
        }
        return d
