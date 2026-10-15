"""Typical decider: SystemOne <-> typical_ai mapping, against a fake with typical_ai's decide().

The fake mirrors typical_ai 0.1.1's contract: decide(state, question, labels) returns
(probabilities over exactly `labels`, runtime with p_null and token counts), and renders the
query with query_text, vendored below verbatim from typical_ai/core.py (Apache-2.0) so the
rendered text is checked, not just the call.
"""

import sys
import types

import pytest

from deciders.typical import engine as te


def query_text(question: dict) -> str:  # typical_ai 0.1.1 core.query_text, verbatim
    q, crit = question["instructions"], question.get("criteria")
    if not crit:
        return q
    if question["type"] == "noul":
        rubric = f"yes: {crit.get('true', 'Yes')}  no: {crit.get('false', 'No')}"
    elif isinstance(crit, list):
        rubric = "  ".join(f"{i}: {c}" for i, c in enumerate(crit))
    else:
        rubric = "  ".join(f"{k}: {v or k}" for k, v in crit.items())
    return f"{q}\n{rubric}"


class FakeTypical:
    """Peaked in the middle and skewed (w = 1 + 3*[middle] + i/2), so argmax is not the last
    label and reversing the labels changes the answer."""

    device = "cuda"

    def __init__(self, p_null=0.07, raise_assert=False):
        self.calls, self.p_null, self.raise_assert = [], p_null, raise_assert
        self.model = types.SimpleNamespace(nc_head="none", null="factored", render="letters",
                                           score_head="choice", noul_head="bern")  # fmt: skip

    def decide(self, state, question, labels):
        if self.raise_assert:
            raise AssertionError("suffix too long")
        query = query_text(question)
        self.calls.append({"state": state, "query": query, "labels": list(labels)})
        n = len(labels)
        w = [1.0 + (i == n // 2) * 3.0 + i / 2 for i in range(n)]
        probs = {label: x / sum(w) for label, x in zip(labels, w, strict=True)}
        return probs, {"p_null": self.p_null, "state_tokens": 40,
                       "query_tokens": len(query.split()), "state_truncated": False}  # fmt: skip


BODY = {
    "state": "Checkout errors, orders blocked.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"technical": "bugs", "billing": "payments", "sales": None}},
        "outage": {"type": "noul", "criteria": {"true": "down", "false": None}},
        "urgency": {"type": "score", "instructions": "How urgent, 1 to 3?",
                    "criteria": ["1", "2", "3"]},
    },
}  # fmt: skip


def engine(**cfg):
    fake = FakeTypical()
    return te.TypicalEngine(te.TypicalConfig(**cfg), model=fake), fake


def test_labels_and_rendered_queries_follow_typical():
    eng, fake = engine()
    d = eng.decide(BODY)
    team, outage, urgency = fake.calls
    assert team["labels"] == ["technical", "billing", "sales"]  # request order, not sorted
    assert team["query"] == "Which team?\ntechnical: bugs  billing: payments  sales: sales"
    assert outage["labels"] == ["no", "yes"]
    assert outage["query"] == "outage\nyes: down  no: No"  # id as instructions; null dropped
    assert urgency["labels"] == ["0", "1", "2"]
    assert urgency["query"] == "How urgent, 1 to 3?\n0: 1  1: 2  2: 3"  # the index-mode mismatch
    assert all(c["state"] == BODY["state"] for c in fake.calls)
    assert [q["option_ids"] for q in d.questions][2] == ["0", "1", "2"]


def test_criteria_mode_labels_levels_by_their_text_without_a_rubric():
    eng, fake = engine(score_labels="criteria")
    d = eng.decide(BODY)
    urgency = fake.calls[2]
    assert urgency["labels"] == ["1", "2", "3"] and urgency["query"] == "How urgent, 1 to 3?"
    a = d.response["answers"]["urgency"]
    assert list(a["probabilities"]) == ["0", "1", "2"]  # ids stay Clef's "0".."n-1"
    with pytest.raises(ValueError, match="distinct"):
        te.labels_for({"type": "score", "criteria": ["x", "x"]}, "criteria")
    with pytest.raises(ValueError, match="score_labels"):
        te.TypicalEngine(te.TypicalConfig(score_labels="names"), model=FakeTypical())


def test_answers_match_the_clef_worker_fields():
    eng, _ = engine()
    a = eng.decide(BODY).response["answers"]
    assert list(a) == ["team", "outage", "urgency"]
    assert a["team"]["choice"] == "billing"  # the middle label carries the peak
    assert a["team"]["confidence"] == pytest.approx(4.5 / 7.5)
    assert a["team"]["probabilities"] == pytest.approx(
        {"technical": 1 / 7.5, "billing": 4.5 / 7.5, "sales": 2 / 7.5}
    )
    assert a["outage"] == {"type": "noul", "noul": pytest.approx(4.5 / 5.5)}  # p(yes); no abstain
    assert a["urgency"]["score"] == pytest.approx(1 * 4.5 / 7.5 + 2 * 2 / 7.5)
    assert a["urgency"]["confidence"] == pytest.approx(4.5 / 7.5)
    assert a["urgency"]["legend"] == {"0": "1", "1": "2", "2": "3"}
    assert a["team"]["abstain"] == a["urgency"]["abstain"] == 0.07


def test_decision_has_the_run_record_shape_haiku_eval_reads(tmp_path):
    import json
    from dataclasses import asdict

    import haiku_eval

    eng, _ = engine()
    d = eng.decide(BODY)
    rec = {"id": "r1", **asdict(d), "error": None}
    (tmp_path / "r1.json").write_text(json.dumps(rec))
    scores = haiku_eval.scores_from_run(tmp_path, ["r1"], "urgency")
    assert "r1" in scores  # Oct 15 audit: the first draft gave {} (no questions block)
    q = d.questions[0]
    assert q["p_null"] == 0.07 and q["query_tokens"] is not None
    assert d.request_sha == te.request_sha(BODY)
    assert d.response["usage"]["input_tokens"] == 40 + sum(x["query_tokens"] for x in d.questions)


def test_long_queries_are_flagged_beyond_the_trained_length():
    eng, _ = engine()
    body = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "w " * 100}}}
    assert eng.decide(body).questions[0]["query_beyond_train_len"] is True
    assert eng.decide(BODY).questions[0]["query_beyond_train_len"] is False


def test_refusals_images_invalid_and_too_long():
    eng, fake = engine()
    with pytest.raises(ValueError, match="text-only"):
        eng.decide({**BODY, "images": ["data:image/png;base64,AA=="]})
    with pytest.raises(ValueError, match="choice needs"):
        eng.decide({"state": "s", "questions": {"c": {"type": "choice", "criteria": {"a": "x"}}}})
    assert fake.calls == []
    long_eng = te.TypicalEngine(te.TypicalConfig(), model=FakeTypical(raise_assert=True))
    with pytest.raises(ValueError, match="too long"):
        long_eng.decide(BODY)


def test_worker_core_serves_it(monkeypatch):
    from worker.core import WorkerCore

    eng, _ = engine()
    core = WorkerCore(eng)
    out = core.decide(BODY, debug=True)
    assert out.status == 200 and out.body["answers"]["team"]["choice"] == "billing"
    assert out.body["debug"]["questions"][0]["option_ids"] == ["technical", "billing", "sales"]
    assert core.decide({**BODY, "images": []}, debug=False).status == 200  # empty list is fine
    m = eng.manifest()
    assert m["decider"] == "typical" and m["head"]["noul_head"] == "bern"


def test_from_pretrained_gets_the_config(monkeypatch):
    got = {}

    class Typical:
        @classmethod
        def from_pretrained(cls, repo, **kw):
            got.update(repo=repo, **kw)
            return FakeTypical()

    monkeypatch.setitem(sys.modules, "typical_ai", types.SimpleNamespace(Typical=Typical))
    cfg = te.TypicalConfig(revision="abc123", max_state=2048, max_states=4, max_option_tokens=None)
    te.TypicalEngine(cfg)
    assert got == {"repo": "OzLabs/typical-medium", "device": "auto", "revision": "abc123",
                   "max_state": 2048, "max_states": 4, "max_option_tokens": None}  # fmt: skip


def test_the_real_package_is_not_imported_with_a_model_given():
    sys.modules.pop("typical_ai", None)
    te.TypicalEngine(te.TypicalConfig(), model=FakeTypical())
    assert "typical_ai" not in sys.modules
