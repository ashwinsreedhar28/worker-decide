"""parity_sglang.py: SGLang /v1/systemone replies mapped onto the oracle schema and gated."""

import json
from pathlib import Path

import parity_sglang as ps
import pytest
from parity import compare

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "fixtures/clef-flash/float32"


def _line(rid: str) -> dict:
    for path in sorted((ROOT / "fixtures/requests").glob("*.jsonl")):
        for raw in path.read_text().splitlines():
            if raw.strip() and json.loads(raw)["id"] == rid:
                return json.loads(raw)
    raise KeyError(rid)


def _reply_from_oracle(rid: str, shuffle: bool = True, nudge: float = 0.0) -> dict:
    """What SGLang's joint-schema path would return if it matched the oracle exactly: choice
    and score probabilities keyed by option id (key order shuffled, as a dict may come back),
    noul as p(true) only, confidence from a different formula."""
    o = json.loads((ORACLE / f"{rid}.json").read_text())
    answers = {}
    for q in o["questions"]:
        probs = [p + (nudge if i == 0 else -nudge if i == 1 else 0) for i, p in
                 enumerate(q["probs"])]  # fmt: skip
        if q["type"] == "noul":
            answers[q["id"]] = {"noul": probs[0]}
            continue
        pairs = list(zip(q["option_ids"], probs, strict=True))
        if shuffle:
            pairs = pairs[::-1]
        answers[q["id"]] = {"probabilities": dict(pairs), "confidence": 0.123}
    return {
        "model": "Cloudflare/clef-flash",
        "answers": answers,
        "usage": {"input_tokens": o["encoding"]["input_tokens"], "output_tokens": 0},
    }


def test_option_order_is_the_heads_not_the_requests():
    # Built here: fixtures/requests/*.jsonl were written with sort_keys=True, so the
    # "unsorted_choice" fixture is sorted on disk (Oct 15 finding; order.jsonl covers it).
    q = {"type": "choice", "criteria": {"technical": "t", "billing": "b", "sales": "s"}}
    assert ps.option_ids(q) == ["billing", "sales", "technical"]
    assert ps.option_ids({"type": "noul"}) == ["true", "false"]
    assert ps.option_ids({"type": "score", "criteria": ["a", "b", "c"]}) == ["0", "1", "2"]


def test_a_matching_reply_passes_every_gate(tmp_path):
    for rid in ("card_trio", "unsorted_choice", "noul_custom", "many_options"):
        rec = ps.record_from_systemone(_line(rid), 200, _reply_from_oracle(rid))
        assert rec["error"] is None and rec["encoding"] is None
        (tmp_path / f"{rid}.json").write_text(json.dumps(rec))
    report = compare(ORACLE, tmp_path, max_dp=0.025, mean_dp=0.0025, subset=True)
    assert report.passed and report.skipped > 0
    assert all(r.span == "n/a" and r.request == "ok" for r in report.records)
    tokens = ps.token_report(ORACLE, tmp_path)
    assert tokens["equal"] == 4 and tokens["different"] == []


def test_a_drifted_reply_fails_the_numeric_gate(tmp_path):
    rec = ps.record_from_systemone(
        _line("card_trio"), 200, _reply_from_oracle("card_trio", nudge=0.05)
    )
    (tmp_path / "card_trio.json").write_text(json.dumps(rec))
    report = compare(ORACLE, tmp_path, max_dp=0.025, mean_dp=0.0025, subset=True)
    assert not report.passed


def test_noul_is_rebuilt_from_p_true():
    line = _line("noul_custom")
    rec = ps.record_from_systemone(line, 200, _reply_from_oracle("noul_custom"))
    q = rec["questions"][0]
    assert q["option_ids"] == ["true", "false"]
    assert q["probs"][0] + q["probs"][1] == pytest.approx(1.0)


def test_http_errors_and_unreadable_replies_become_error_records():
    line = _line("card_trio")
    rec = ps.record_from_systemone(line, 400, {"detail": [{"msg": "too many options"}]})
    assert rec["error"].startswith("http 400") and rec["questions"] == []
    bad = _reply_from_oracle("card_trio")
    bad["answers"]["department"]["probabilities"] = {"billing": 1.0}  # an option missing
    rec = ps.record_from_systemone(line, 200, bad)
    assert rec["error"].startswith("unreadable reply") and rec["questions"] == []
    del bad["answers"]["department"]
    assert "no answer" in ps.record_from_systemone(line, 200, bad)["error"]


def test_schema_overflow_must_error_to_pass(tmp_path):
    o = json.loads((ORACLE / "schema_overflow.json").read_text())
    assert o["error"]  # the oracle refuses it, so must the server
    line = _line("schema_overflow")
    (tmp_path / "schema_overflow.json").write_text(
        json.dumps(ps.record_from_systemone(line, 400, {"detail": "prompt too long"}))
    )
    assert compare(ORACLE, tmp_path, subset=True).passed


def test_token_report_flags_a_different_prompt(tmp_path):
    reply = _reply_from_oracle("card_trio")
    reply["usage"]["input_tokens"] += 7
    (tmp_path / "card_trio.json").write_text(
        json.dumps(ps.record_from_systemone(_line("card_trio"), 200, reply))
    )
    t = ps.token_report(ORACLE, tmp_path)
    assert t["different"] == [{"id": "card_trio", "oracle": 300, "run": 307}]
    assert "card_trio" in ps.render_tokens(t)


def test_the_model_field_is_overridden_for_the_server():
    assert ps.with_model("Cloudflare/clef-flash")({"model": "x", "state": "s"})["model"] == (
        "Cloudflare/clef-flash"
    )


def _write(tmp_path, rid, status, reply):
    rec = ps.record_from_systemone(_line(rid), status, reply)
    (tmp_path / f"{rid}.json").write_text(json.dumps(rec))


def test_a_known_refusal_is_skipped_and_any_other_refusal_fails(tmp_path):
    o = json.loads((ORACLE / "empty_instructions.json").read_text())
    assert not o["error"]  # the reference answers it; SGLang's request model refuses it
    _write(tmp_path, "card_trio", 200, _reply_from_oracle("card_trio"))
    _write(tmp_path, "empty_instructions", 422, {"detail": [{"msg": "noul needs instructions"}]})
    report = compare(ORACLE, tmp_path, max_dp=0.025, mean_dp=0.0025, subset=True)
    assert not report.passed
    assert ps.apply_known_refusals(report, tmp_path) == ["empty_instructions"]
    assert report.passed
    # The same refusal on a record the server should answer is a failure.
    _write(tmp_path, "noul_custom", 422, {"detail": "nope"})
    report = compare(ORACLE, tmp_path, max_dp=0.025, mean_dp=0.0025, subset=True)
    ps.apply_known_refusals(report, tmp_path)
    assert not report.passed
    # A known record refused with a different status is not the known difference.
    (tmp_path / "noul_custom.json").unlink()
    _write(tmp_path, "empty_instructions", 500, {"error": "boom"})
    report = compare(ORACLE, tmp_path, max_dp=0.025, mean_dp=0.0025, subset=True)
    assert ps.apply_known_refusals(report, tmp_path) == [] and not report.passed


def test_skips_alone_do_not_pass(tmp_path):
    _write(tmp_path, "empty_instructions", 422, {"detail": "x"})
    report = compare(ORACLE, tmp_path, subset=True)
    ps.apply_known_refusals(report, tmp_path)
    assert not report.passed


def test_an_extra_option_in_the_reply_is_unreadable():
    reply = _reply_from_oracle("card_trio")
    reply["answers"]["department"]["probabilities"]["sales"] = 0.0
    rec = ps.record_from_systemone(_line("card_trio"), 200, reply)
    assert rec["error"].startswith("unreadable reply")


def test_noul_probability_is_p_true_first():
    q = {"type": "noul"}
    assert ps.probs_from_answer(q, {"noul": 0.8}) == [0.8, pytest.approx(0.2)]


def test_post_sends_the_body_in_its_order_and_survives_a_dropped_connection(monkeypatch):
    import urllib.error

    sent = {}

    class Resp:
        status = 200

        def read(self):
            return b'{"answers": {}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        sent["body"] = req.data
        return Resp()

    monkeypatch.setattr(ps.urllib.request, "urlopen", fake_urlopen)
    body = {"state": "s", "questions": {"z": {"type": "noul"}, "a": {"type": "noul"}}}
    assert ps.post("http://x", body, 5)[0] == 200
    assert list(json.loads(sent["body"])["questions"]) == ["z", "a"]  # not sorted

    def dropped(req, timeout):
        raise urllib.error.URLError(ConnectionResetError("reset"))

    monkeypatch.setattr(ps.urllib.request, "urlopen", dropped)
    status, reply = ps.post("http://x", body, 5)
    assert status == 0 and "URLError" in reply["error"]
