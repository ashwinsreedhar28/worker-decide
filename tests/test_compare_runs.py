"""compare_runs.py: two runs that err in opposite directions are each close to the truth."""

import json

import compare_runs as cr


def _write(d, rid, probs, error=None):
    d.mkdir(parents=True, exist_ok=True)
    rec = {"id": rid, "error": error, "questions": [] if error else [{"id": "q", "probs": probs}]}
    (d / f"{rid}.json").write_text(json.dumps(rec))


def test_opposite_errors_are_reported_as_noise_not_a_wrong_run(tmp_path):
    t, a, b = tmp_path / "t", tmp_path / "a", tmp_path / "b"
    _write(t, "r1", [0.60, 0.40])  # clear margin 0.20
    _write(a, "r1", [0.61, 0.39])  # +0.01
    _write(b, "r1", [0.59, 0.41])  # -0.01: a and b are 0.02 apart
    _write(t, "r2", [0.51, 0.49])  # near-tie
    _write(a, "r2", [0.49, 0.51])  # flips, but inside the tie window
    _write(b, "r2", [0.51, 0.49])
    _write(t, "r3", [0.5, 0.5])
    _write(a, "r3", None, error="schema too long")  # skipped
    _write(b, "r3", [0.5, 0.5])
    c = cr.compare(t, a, b, near_tie=0.04, labels=("A", "B"))
    assert len(c.rows) == 2 and c.skipped == 1
    assert abs(c.stats("a_truth")["max"] - 0.02) < 1e-9  # r2 moved 0.02
    assert abs(c.stats("a_b")["max"] - 0.02) < 1e-9  # r1: 0.61 vs 0.59
    n, clear = c.flips("flip_a")
    assert n == 1 and clear == []  # the only flip is a near-tie
    text = cr.render(c)
    assert "A vs truth" in text and "0 clear" in text


def test_a_clear_flip_is_named(tmp_path):
    t, a, b = tmp_path / "t", tmp_path / "a", tmp_path / "b"
    _write(t, "r", [0.7, 0.3])
    _write(a, "r", [0.3, 0.7])
    _write(b, "r", [0.7, 0.3])
    c = cr.compare(t, a, b)
    n, clear = c.flips("flip_a")
    assert n == 1 and [r.record for r in clear] == ["r"]
    assert "CLEAR: r/q" in cr.render(c)
    assert c.b_closer == 1 and c.a_closer == 0
