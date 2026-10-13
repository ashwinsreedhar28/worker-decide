"""Every committed request fixture is a valid SystemOne body with a unique id."""

from pathlib import Path

import pytest

from deciders.systemone import read_jsonl, validate_request

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "fixtures" / "requests").glob("*.jsonl"))


def test_validate_request_rejects_bad_bodies():
    assert validate_request({}) == ["missing state", "questions must be a non-empty mapping"]
    assert validate_request({"state": "x", "questions": {"q": {"type": "pick"}}})
    assert validate_request(
        {"state": "x", "questions": {"q": {"type": "score", "criteria": ["1"]}}}
    )
    assert validate_request({"state": "x", "questions": {"q": {"type": "noul"}}}) == []


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_request_fixture_file(path):
    lines = read_jsonl(path)
    assert lines, f"{path.name} is empty"
    ids = [line["id"] for line in lines]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for line in lines:
        assert validate_request(line["request"]) == [], (
            f"{line['id']}: {validate_request(line['request'])}"
        )


def test_pulse50_matches_brians_table():
    path = ROOT / "fixtures" / "requests" / "pulse50.jsonl"
    if not path.exists():
        pytest.skip("pulse50.jsonl not exported yet")
    lines = read_jsonl(path)
    assert len(lines) == 50
    for line in lines:
        q = line["request"]["questions"]
        assert list(q) == ["urgency"]
        assert q["urgency"]["type"] == "score"
        assert q["urgency"]["criteria"] == ["1", "2", "3", "4", "5"]
        assert line["request"]["state"].startswith("Headline: ")
        assert "Respond in JSON" not in q["urgency"]["instructions"]
