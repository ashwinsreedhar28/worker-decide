"""make_fixtures.py is deterministic and each edge record targets what it says."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "make_fixtures", ROOT / "scripts" / "make_fixtures.py"
)
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

pytestmark = pytest.mark.skipif(
    not (ROOT / "fixtures" / "requests" / "pulse50.jsonl").exists(),
    reason="long records reuse Pulse states; export them first",
)


def test_edge_records_shape():
    recs = mf.edge_records()
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids))
    by_id = {r["id"]: r for r in recs}
    assert list(by_id["card_trio"]["request"]["questions"]) == ["department", "urgency", "outage"]
    assert "instructions" not in by_id["no_instruction"]["request"]["questions"]["refund_eligible"]
    assert list(by_id["unsorted_choice"]["request"]["questions"]["team"]["criteria"]) == [
        "technical",
        "billing",
        "sales",
    ]
    assert isinstance(by_id["json_state"]["request"]["state"], dict)
    assert len(by_id["many_options"]["request"]["questions"]["category"]["criteria"]) == 40
    assert 8_000 < len(by_id["long_2k"]["request"]["state"]) < 12_000
    assert (
        len(by_id["trunc_16k"]["request"]["state"]) > 70_000
    )  # > 16,384 tokens at any plausible ratio


def test_edge_records_deterministic():
    a = json.dumps(mf.edge_records(), sort_keys=True)
    b = json.dumps(mf.edge_records(), sort_keys=True)
    assert a == b


def test_image_records_have_twins_and_files():
    recs = mf.make_images()
    ids = {r["id"] for r in recs}
    for base in ("img_receipt", "img_chart", "img_shapes"):
        assert base in ids and f"{base}_textonly" in ids
    for r in recs:
        for rel in r.get("images", []):
            assert (ROOT / rel).exists(), rel
        twin = next(t for t in recs if t["id"] == r["id"].removesuffix("_textonly"))
        assert twin["request"] == r["request"]


def test_coverage_records_hit_hosted_limits():
    import coverage_records as cr

    recs = {r["id"]: r for r in cr.records()}
    q = lambda rid, name: recs[rid]["request"]["questions"][name]  # noqa: E731
    assert len(q("intent77", "intent")["criteria"]) == 77
    assert len(q("opts255", "sku")["criteria"]) == 255
    assert len(recs["q64"]["request"]["questions"]) == 64
    assert len(q("score10", "frustration")["criteria"]) == 10
    assert recs["schema_overflow"]["expect_error"] == "schema requires"
    assert q("empty_instructions", "needs_shipping_help")["instructions"] == ""
    assert isinstance(recs["list_state"]["request"]["state"], list)
    assert "none" in q("moderation", "category")["criteria"]
    assert recs["near_dup_a"]["request"]["state"] != recs["near_dup_b"]["request"]["state"]


def test_all_request_ids_unique_across_files():
    import json

    ids = []
    for path in (ROOT / "fixtures" / "requests").glob("*.jsonl"):
        ids += [json.loads(line)["id"] for line in path.read_text().splitlines() if line]
    assert len(ids) == len(set(ids))
    assert len(ids) == 87, len(ids)
