"""batch_report.py: batch sizes behind a parity run, and the verdict."""

import json

import batch_report as br
import pytest


def _run(tmp_path, sizes):
    d = tmp_path / "run"
    d.mkdir()
    for i, s in enumerate(sizes):
        rec = {"id": f"r{i}", "questions": []}
        if s is not None:
            rec["batch"] = {"size": s, "tokens": [1] * s, "forward_ms": 1.0}
        (d / f"r{i}.json").write_text(json.dumps(rec))
    (d / "manifest.json").write_text(json.dumps({"counts": {}}))
    return d


def test_counts_sizes_and_passes_when_some_records_batched(tmp_path, capsys):
    d = _run(tmp_path, [1, 3, 3, 3, None, 2, 2])
    assert br.sizes(d) == {1: 1, 3: 3, None: 1, 2: 2}
    root = tmp_path / "root.json"
    root.write_text(json.dumps({"batcher": {"batches": 4, "retried_alone": 0, "stopped": None}}))
    br.main([str(d), "--root", str(root)])
    out = capsys.readouterr().out
    assert "5 of 7 records answered in batches of 2+" in out and "result: PASS" in out
    assert "| none | 1 |" in out


def test_fails_when_nothing_batched(tmp_path):
    d = _run(tmp_path, [1, 1, None])
    with pytest.raises(SystemExit):
        br.main([str(d)])
    assert br.verdict(br.sizes(d), None) == (
        False,
        ["no record was answered in a batch of 2 or more"],
    )


def test_fails_when_batches_were_retried_alone_or_the_batcher_stopped(tmp_path):
    counts = br.sizes(_run(tmp_path, [2, 2]))
    ok, reasons = br.verdict(counts, {"retried_alone": 4, "stopped": None})
    assert not ok and "4 requests were retried alone" in reasons[0]
    ok, reasons = br.verdict(counts, {"retried_alone": 0, "stopped": "batcher stopped: x"})
    assert not ok and "the batcher stopped" in reasons[0]
    assert br.verdict(counts, {"retried_alone": 0, "stopped": None}) == (True, [])
