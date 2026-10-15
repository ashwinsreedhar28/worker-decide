"""parity.py on a synthetic oracle: identical passes; each failure mode fails for its reason."""

import copy
import json
from pathlib import Path

import parity


def _rec(rid, questions, error=None, encoding=True):
    rec = {"id": rid, "request_sha": f"req-{rid}", "questions": questions, "error": error}
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
    rep = _run(tmp_path, run, max_dp=1.0, mean_dp=1.0, near_tie=0.02)
    assert not rep.passed
    assert any("argmax x → y" in s for s in rep.by["a"].reasons)


def test_near_tie_defaults_to_twice_max_dp(tmp_path):
    # Oracle margin 0.03, run flips it with every option moving ≤ 0.016: inside the 0.02 bar.
    oracle = copy.deepcopy(ORACLE)
    oracle[1]["questions"][0]["probs"] = [0.515, 0.485]
    run = copy.deepcopy(oracle)
    run[1]["questions"][0]["probs"] = [0.499, 0.501]
    o, r = tmp_path / "oracle", tmp_path / "run"
    _write(o, oracle)
    _write(r, run)
    rep = parity.compare(o, r, mean_dp=0.01)
    assert rep.near_tie == 0.04 and rep.passed  # listed as a tie, not an argmax failure
    rep = parity.compare(o, r, mean_dp=0.01, near_tie=0.02)
    assert not rep.passed  # the old default would have failed a within-bar run


def test_request_gate_reported_first_and_skipped_when_absent(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[0]["request_sha"] = "other-body"
    run[0]["questions"][0]["probs"] = [0.2, 0.7, 0.1]  # numeric failure too; request comes first
    rep = _run(tmp_path, run)
    assert not rep.passed and rep.by["a"].request == "mismatch"
    assert rep.by["a"].reasons[0].startswith("request gate: sha req-a")
    assert rep.by["boom"].request == "ok"  # checked on error records as well
    run = [{k: v for k, v in r.items() if k != "request_sha"} for r in copy.deepcopy(ORACLE)]
    rep = _run(tmp_path, run)
    assert rep.passed and rep.by["a"].request == "n/a"


def test_span_gate_reported_before_numbers(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[0]["encoding"]["spans_sha"] = "different"
    rep = _run(tmp_path, run)
    assert not rep.passed
    assert rep.by["a"].span == "mismatch"
    assert rep.by["a"].request == "ok" and rep.by["a"].reasons[0].startswith("span gate: spans")


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


def test_subset_skips_missing_records_instead_of_failing(tmp_path):
    run = [r for r in copy.deepcopy(ORACLE) if r["id"] != "boom"]
    assert not _run(tmp_path, run).passed
    rep = _run(tmp_path, run, subset=True)
    assert rep.passed and rep.skipped == 1 and {r.record for r in rep.records} == {"a", "tie"}
    assert "+1 skipped" in rep.summary_lines()[2]


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


def test_question_mean_max_dp_is_informational(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[0]["questions"][0]["probs"] = [0.69, 0.205, 0.105]  # max 0.01; other two questions 0
    rep = _run(tmp_path, run)
    assert rep.passed and abs(rep.question_mean_max_dp - 0.01 / 3) < 1e-9
    assert "mean of question max" in "\n".join(rep.summary_lines())


def test_render_has_table_and_sections(tmp_path):
    run = copy.deepcopy(ORACLE)
    run[1]["questions"][0]["probs"] = [0.495, 0.505]
    text = parity.render(_run(tmp_path, run, mean_dp=0.01))
    assert "| result | PASS |" in text and "Near-ties" in text and "tie/q" in text


def test_zero_records_compared_is_a_failure(tmp_path):
    (tmp_path / "oracle").mkdir()
    (tmp_path / "run").mkdir()
    report = parity.compare(tmp_path / "oracle", tmp_path / "run")
    assert report.records == [] and not report.passed


def test_non_record_json_in_a_directory_is_ignored(tmp_path, capsys):
    import shutil

    src = Path(__file__).resolve().parents[1] / "fixtures/clef-flash/float32"
    (tmp_path / "o").mkdir()
    shutil.copy(src / "card_trio.json", tmp_path / "o" / "card_trio.json")
    (tmp_path / "o" / "findings.json").write_text('{"unsorted_same_ids": true}')
    report = parity.compare(tmp_path / "o", tmp_path / "o")
    assert report.passed and len(report.records) == 1
    assert "findings.json" in capsys.readouterr().err
