"""order_findings.py on stand-in data (the real order oracle comes from the pod)."""

import json
import shutil
from pathlib import Path

import order_findings as of

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "fixtures/clef-flash/float32"


def _stand_ins(tmp_path):
    shutil.copy(BASE / "unsorted_choice.json", tmp_path / "order_choice_sorted.json")
    shutil.copy(BASE / "unsorted_choice.json", tmp_path / "order_choice_unsorted.json")
    for name in ("alphabetical", "as_card", "reversed"):
        shutil.copy(BASE / "card_trio.json", tmp_path / f"order_questions_{name}.json")


def test_identical_stand_ins_show_no_order_effect(tmp_path):
    _stand_ins(tmp_path)
    f = of.findings(tmp_path)
    assert f["unsorted_same_ids"] and f["unsorted_max_dp"] == 0.0
    assert all(v == 0 for q in f["question_orders"].values()
               for v in q["max_dp_vs_alphabetical"].values())  # fmt: skip
    assert "identical" in of.render(f)


def test_a_moved_answer_and_different_ids_are_reported(tmp_path):
    _stand_ins(tmp_path)
    u = json.loads((tmp_path / "order_choice_unsorted.json").read_text())
    u["encoding"]["ids_sha"] = "different"
    (tmp_path / "order_choice_unsorted.json").write_text(json.dumps(u))
    c = json.loads((BASE / "card_trio.json").read_text())
    q = next(x for x in c["questions"] if x["id"] == "outage")
    q["probs"] = [q["probs"][0] - 0.03, q["probs"][1] + 0.01]  # asymmetric: max, not min
    (tmp_path / "order_questions_reversed.json").write_text(json.dumps(c))
    f = of.findings(tmp_path)
    assert not f["unsorted_same_ids"] and "DIFFERENT" in of.render(f)
    assert abs(f["question_orders"]["reversed"]["max_dp_vs_alphabetical"]["outage"] - 0.03) < 1e-6


def test_main_writes_outside_the_oracle_directory(tmp_path, monkeypatch, capsys):
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _stand_ins(oracle)
    out = tmp_path / "findings.json"
    monkeypatch.setattr("sys.argv", ["order_findings.py", str(oracle), str(out)])
    of.main()
    assert out.exists() and not (oracle / "findings.json").exists()
