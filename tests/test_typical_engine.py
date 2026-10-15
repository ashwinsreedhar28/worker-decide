"""Typical decider: SystemOne <-> typical_ai mapping, against a fake with typical_ai's decide().

The fake mirrors typical_ai 0.1.1's contract: decide(state, question, labels) returns
(probabilities over exactly `labels`, runtime with p_null), and query_text needs
question["instructions"] (it raises KeyError without it).
"""

import pytest

from deciders.typical import engine as te


class FakeTypical:
    def __init__(self):
        self.calls = []

    def decide(self, state, question, labels):
        question["instructions"]  # what typical_ai.query_text does first
        self.calls.append((state, question, labels))
        n = len(labels)
        probs = {label: (i + 1) / (n * (n + 1) / 2) for i, label in enumerate(labels)}
        return probs, {"p_null": 0.07, "latency_s": 0.01, "state_tokens": 40}


BODY = {
    "state": "Checkout errors, orders blocked.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"technical": "bugs", "billing": "payments"}},
        "outage": {"type": "noul", "criteria": {"true": "down", "false": "up"}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["Can wait", "This week", "Today"]},
    },
}  # fmt: skip


def test_labels_follow_typicals_conventions():
    q = BODY["questions"]
    assert te.labels_for(q["team"]) == ["technical", "billing"]  # request order, not sorted
    assert te.labels_for(q["urgency"]) == ["0", "1", "2"]
    assert te.labels_for(q["outage"]) == ["no", "yes"]
    with pytest.raises(ValueError):
        te.labels_for({"type": "rank"})


def test_missing_instructions_fall_back_to_the_question_id():
    tq = te.typical_question("outage", BODY["questions"]["outage"])
    assert tq["instructions"] == "outage" and tq["criteria"] == {"true": "down", "false": "up"}
    assert "criteria" not in te.typical_question("q", {"type": "noul", "instructions": "x"})


def test_decide_maps_every_type_and_adds_abstain():
    fake = FakeTypical()
    d = te.TypicalEngine(te.TypicalConfig(), model=fake).decide(BODY)
    a = d.response["answers"]
    assert list(a) == ["team", "outage", "urgency"]  # request order
    assert a["team"]["choice"] == "billing" and a["team"]["confidence"] == pytest.approx(2 / 3)
    assert a["outage"] == {"type": "noul", "noul": pytest.approx(2 / 3), "abstain": 0.07}
    assert a["urgency"]["score"] == pytest.approx(1 * 2 / 6 + 2 * 3 / 6)
    assert a["urgency"]["legend"] == {"0": "Can wait", "1": "This week", "2": "Today"}
    assert all(x["abstain"] == 0.07 for x in a.values())
    assert [c[2] for c in fake.calls] == [["technical", "billing"], ["no", "yes"], ["0", "1", "2"]]
    assert d.runtime["team"]["state_tokens"] == 40 and "p_null" not in d.runtime["team"]


def test_invalid_requests_are_refused_before_the_model():
    fake = FakeTypical()
    eng = te.TypicalEngine(te.TypicalConfig(), model=fake)
    with pytest.raises(ValueError, match="choice needs"):
        eng.decide({"state": "s", "questions": {"c": {"type": "choice", "criteria": {"a": "x"}}}})
    assert fake.calls == []


def test_the_real_package_is_not_imported_with_a_model_given():
    import sys

    te.TypicalEngine(te.TypicalConfig(), model=FakeTypical())
    assert "typical_ai" not in sys.modules
