"""SystemOne wire format: request validation shared by fixtures, tests and (later) the worker.

Mirrors the checks in Cloudflare's ``joint_schema_model.systemone`` so a request the oracle
accepts is one the worker accepts, and vice versa.
"""

from __future__ import annotations

from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")


def validate_request(body: dict[str, Any]) -> list[str]:
    """Return a list of problems; empty means valid."""
    problems: list[str] = []
    if "state" not in body:
        problems.append("missing state")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        return problems + ["questions must be a non-empty mapping"]
    for qid, q in questions.items():
        if not isinstance(q, dict):
            problems.append(f"{qid}: question must be an object")
            continue
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            problems.append(f"{qid}: type must be one of {QUESTION_TYPES}, got {qtype!r}")
            continue
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                problems.append(f"{qid}: choice needs a mapping of ≥2 option ids")
        elif qtype == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                problems.append(f"{qid}: score needs a list of ≥2 levels")
        elif criteria is not None and not isinstance(criteria, dict):
            problems.append(f"{qid}: noul criteria, if given, must be a mapping")
    return problems


def read_jsonl(path) -> list[dict[str, Any]]:
    import json

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
