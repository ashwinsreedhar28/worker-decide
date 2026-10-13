"""The oracle's torch-free helpers: encoding block, request loading, image record assembly."""

import json
from dataclasses import dataclass

import pytest

from deciders.clef import oracle

ROOT = oracle.ROOT


@dataclass(frozen=True)
class Q:
    question_id: str
    question_type: int
    question_span: tuple
    option_spans: tuple
    option_ids: tuple


@dataclass(frozen=True)
class Enc:
    input_ids: tuple
    questions: tuple


def test_encoding_block_is_deterministic_and_span_sensitive():
    enc = Enc(
        input_ids=(1, 2, 3, 4, 5),
        questions=(Q("q", 1, (1, 2), ((3, 4), (4, 5)), ("a", "b")),),
    )
    a = oracle.encoding_block(enc)
    assert a["input_tokens"] == 5 and a["questions"][0]["type"] == "choice"
    assert a == oracle.encoding_block(enc)
    shifted = Enc(enc.input_ids, (Q("q", 1, (1, 2), ((3, 4), (3, 5)), ("a", "b")),))
    b = oracle.encoding_block(shifted)
    assert b["ids_sha"] == a["ids_sha"] and b["spans_sha"] != a["spans_sha"]
    other_ids = Enc((1, 2, 3, 4, 6), enc.questions)
    assert oracle.encoding_block(other_ids)["ids_sha"] != a["ids_sha"]


def test_load_requests_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"id": "a", "request": {}})
        + "\n"
        + json.dumps({"id": "a", "request": {}})
        + "\n"
    )
    with pytest.raises(SystemExit, match="duplicate"):
        oracle.load_requests([str(p)])


def test_load_requests_over_committed_fixtures():
    lines = oracle.load_requests([str(ROOT / "fixtures" / "requests" / "*.jsonl")])
    assert len(lines) == 87
    assert len({ln["id"] for ln in lines}) == 87


def test_record_for_model_loads_images():
    line = {
        "id": "img_two",
        "request": {"state": "s", "questions": {"q": {"type": "noul"}}},
        "images": ["fixtures/requests/images/receipt.png", "fixtures/requests/images/chart.png"],
    }
    rec = oracle.record_for_model(line)
    assert rec["id"] == "img_two" and rec["state"] == "s"
    assert len(rec["images"]) == 2 and rec["images"][0].mode == "RGB"
    assert "images" not in oracle.record_for_model(
        {"id": "t", "request": {"state": "s", "questions": {}}}
    )


def test_canonical_request_sha_ignores_key_order():
    a = oracle.sha(oracle.canonical({"b": 1, "a": [1, 2]}))
    b = oracle.sha(oracle.canonical({"a": [1, 2], "b": 1}))
    assert a == b
