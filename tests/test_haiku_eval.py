"""haiku_eval.py: decoders and the agreement metrics match Pulse's harness definitions."""

import math

import haiku_eval


def test_decoders_are_one_based_and_ev_rounds_half_up():
    assert haiku_eval.decode([0.1, 0.2, 0.7]) == {"argmax": 3, "ev": 3}  # E = 2.6 -> 3
    assert haiku_eval.decode([0.5, 0.0, 0.5]) == {"argmax": 1, "ev": 2}  # tie -> first; E = 2.0
    assert haiku_eval.decode([0.0, 0.5, 0.5])["ev"] == 3  # E = 2.5 rounds up like int(x + 0.5)


def test_metrics():
    labels = {"a": 4, "b": 2, "c": 5, "d": 1, "e": 3}
    scores = {"a": 4, "b": 3, "c": 3, "d": 1, "e": 5}  # c: missed urgent; e: false alarm
    m = haiku_eval.evaluate(scores, labels)
    assert m["n"] == 5 and m["exact"] == 0.4 and m["within1"] == 0.6 and m["mae"] == 1.0
    assert m["prec"] == 0.5 and m["rec"] == 0.5 and m["flagged"] == 2 and m["ref_urgent"] == 2
    assert haiku_eval.evaluate({"zzz": 1}, labels) == {"n": 0}
    assert math.isnan(haiku_eval.evaluate({"b": 2, "d": 1}, labels)["prec"])  # nothing flagged
