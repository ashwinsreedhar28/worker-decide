"""parity.py on a synthetic oracle: identical passes; each failure mode fails for its reason."""

import copy
import json
from pathlib import Path

import parity


def _rec(rid, questions, error=None, encoding=True):
    rec = {"id": rid, "questions": questions, "error": error}
    if encoding:
        rec["encoding"] = {
            "input_tokens": 100,
            "ids_sha": f"ids-{rid}",
            "spans_sha": f"spans-{rid}",
        }
    return rec


ORACLE = [
    _rec(
        "a",
        [
            {"id": "q1", "type": "choice", "option_ids": ["x", "y", "z"], "probs": [0.7, 0.2, 0.1]},
            {"id": "q2", "type": "noul", "option_ids": ["true", "false"], "probs": [0.9, 0.1]},
        ],
    ),
    _rec(
        "tie",
        [{"id": "q", "type": "choice", "option_ids": ["x", "y"], "probs": [0.505, 0.495]}],
    ),
    _rec("boom", [], error="schema requires 300000 tokens before state; maximum is 16384"),
]


def _write(dirpath: Path, recs):
    dirpath.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (dirpath / f"{r['id']}.json").write_text(json.dumps(r))


def _run(tmp_path, run_recs, **kw):
    o, r = tmp_path / "oracle", tmp_path / "run"
    _write(o, ORACLE)
    _write(r, run_recs)
    rep = parity.compare(o, r, **kw)
    rep.by = {rec.record: rec for rec in rep.records}  # records load in filename order
    return rep


def test_identical_passes(tmp_path):
    rep = _run(tmp_path, copy.deepcopy(ORACLE))
    assert rep.passed
    assert {r.status for r in rep.records} == {"PASS"}
    assert rep.by["a"].span == "ok"
    assert rep.overall_mean_dp == 0.0
    assert "both errored" in rep.by["boom"].reasons


def test_small_drift_passes_large_drift_fails(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[0]["questions"][0]["probs"] = [0.69, 0.205, 0.105]
    assert _run(tmp_path, run).passed
    run[0]["questions"][0]["probs"] = [0.65, 0.25, 0.1]  # Δ 0.05 > 0.02
    rep = _run(tmp_path, run)
    assert not rep.passed
    assert any("max |Δp| 0.0500" in s for s in rep.by["a"].reasons)


def test_argmax_flip_fails_but_near_tie_is_listed_not_failed(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[1]["questions"][0]["probs"] = [0.495, 0.505]  # flip inside near-tie margin, Δ 0.01
    rep = _run(tmp_path, run, mean_dp=0.01)  # mean bar relaxed; this test is about the tie rule
    assert rep.passed
    tie_q = rep.by["tie"].questions[0]
    assert tie_q.near_tie and tie_q.argmax_ok is None
    run[0]["questions"][0]["probs"] = [0.2, 0.7, 0.1]  # clear flip
    rep = _run(tmp_path, run, max_dp=1.0, mean_dp=1.0)
    assert not rep.passed
    assert any("argmax x → y" in s for s in rep.by["a"].reasons)


def test_span_gate_reported_before_numbers(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[0]["encoding"]["spans_sha"] = "different"
    rep = _run(tmp_path, run)
    assert not rep.passed
    assert rep.by["a"].span == "mismatch"
    assert rep.by["a"].reasons[0].startswith("span gate: spans")


def test_missing_encoding_is_na_not_failure(tmp_path):
    run = [{k: v for k, v in r.items() if k != "encoding"} for r in copy.deepcopy(ORACLE)]
    rep = _run(tmp_path, run)
    assert rep.passed
    assert rep.by["a"].span == "n/a"


def test_missing_record_and_error_mismatch_fail(tmp_path):
    run = [r for r in copy.deepcopy(ORACLE) if r["id"] != "boom"]
    rep = _run(tmp_path, run)
    assert not rep.passed and rep.by["boom"].reasons == ["missing in run"]
    run = copy.deepcopy(ORACLE)
    run[2]["error"] = None
    run[2]["questions"] = []
    rep = _run(tmp_path, run)
    assert not rep.passed and "oracle errored" in rep.by["boom"].reasons[0]


def test_mean_of_means_bar(tmp_path):
    run = copy.deepcopy(ORACLE)
    for q in run[0]["questions"]:
        q["probs"] = [
            p + (0.015 if i == 0 else -0.015 / (len(q["probs"]) - 1))
            for i, p in enumerate(q["probs"])
        ]
    rep = _run(tmp_path, run)  # every option off by ≤0.015: max_dp passes, mean 0.0125 > 0.002
    assert not rep.passed
    assert all(r.status == "PASS" for r in rep.records)
    assert rep.overall_mean_dp > 0.002


def test_render_has_table_and_sections(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[1]["questions"][0]["probs"] = [0.495, 0.505]
    text = parity.render(_run(tmp_path, run, mean_dp=0.01))
    assert "| result | PASS |" in text and "Near-ties" in text and "tie/q" in text
