"""cost.py: steady-state $/1k decisions and the Workers AI comparison."""

import json
from pathlib import Path

import cost
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRICES = json.loads((ROOT / "fixtures/prices/2026-10-15.json").read_text())


def test_arithmetic():
    # $1.22/h at 3 req/s: 1.22/3600/3 per decision -> x1000
    assert cost.runpod_usd_per_1k(1.22, 3.0) == pytest.approx(1.22 / 3600 / 3 * 1000)
    assert cost.runpod_usd_per_1k(1.22, 0.0) is None
    # 300 tokens at $0.038/M -> $0.0000114 per decision -> $0.0114 per 1k
    assert cost.workers_ai_usd_per_1k(300, 0.038) == pytest.approx(0.0114)


def test_the_price_file_carries_sources_and_the_october_price():
    assert PRICES["workers_ai_clef_flash"]["usd_per_million_input_tokens"] == 0.038
    assert "_source" in PRICES["runpod_serverless_flex_usd_per_hour"]
    tiers = PRICES["runpod_serverless_flex_usd_per_hour"]
    mapped = [t for k, t in PRICES["gpu_tier"].items() if not k.startswith("_")]
    assert all(t in tiers for t in mapped)
    assert PRICES["gpu_tier"]["NVIDIA H100 80GB HBM3"] == "80 GB PRO"  # our SXM runs' name
    assert PRICES["gpu_tier"]["NVIDIA A100-SXM4-80GB"] == "80 GB" and tiers["80 GB"] == 2.72


def test_rows_and_render_from_a_load_file():
    load = {
        "label": "ours-a40",
        "cells": [
            {"workload": "ticket", "input_tokens": 300, "concurrency": 1,
             "stats": {"n": 128, "errors": 0, "req_per_s": 3.0, "p50_ms": 330.0}},
            {"workload": "ticket", "input_tokens": 300, "concurrency": 8,
             "stats": {"n": 0, "errors": 5, "req_per_s": 0.0, "p50_ms": None}},
            {"workload": "ticket", "input_tokens": 300, "concurrency": 32,
             "stats": {"n": 100, "errors": 2, "req_per_s": 9.0, "p50_ms": 900.0}},
        ],
    }  # fmt: skip
    r = cost.rows(load, "NVIDIA A40", PRICES)
    assert r[0]["tier"] == "48 GB" and r[0]["usd_per_hour"] == 1.22
    assert r[0]["runpod_over_workers_ai"] == pytest.approx((1.22 / 3600 / 3 * 1000) / 0.0114)
    assert r[1]["runpod_usd_per_1k"] is None and r[1]["runpod_over_workers_ai"] is None
    assert r[2]["runpod_usd_per_1k"] is None and "2 errors" in r[2]["excluded"]  # not priced
    text = cost.render(r, PRICES)
    rp = 1.22 / 3600 / 3 * 1000
    assert (
        f"| ours-a40 | NVIDIA A40 (48 GB, $1.22/h) | ticket | 300 | 1 | 3.0 | ${rp:.4f} "
        f"| $0.0114 | {rp / 0.0114:.2f}x |"
    ) in text  # full row: column order and ratio direction
    assert "not priced: 2 errors" in text and "$0.038/M input" in text


def test_an_unknown_gpu_is_refused():
    with pytest.raises(SystemExit, match="no price tier"):
        cost.rows({"cells": []}, "NVIDIA B200", PRICES)


def test_main_prices_several_files_with_one_gpu(tmp_path, monkeypatch, capsys):
    load = {"label": "x", "cells": [{"workload": "t", "input_tokens": 100, "concurrency": 1,
            "stats": {"n": 10, "errors": 0, "req_per_s": 2.0}}]}  # fmt: skip
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(json.dumps({**load, "label": name}))
    monkeypatch.setattr("sys.argv", ["cost.py", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
                                     "--gpu", "NVIDIA L40S"])  # fmt: skip
    cost.main()
    out = capsys.readouterr().out
    assert out.count("NVIDIA L40S (48 GB PRO, $1.75/h)") == 2
    files = [str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    three = ["--gpu", "NVIDIA L40S", "--gpu", "NVIDIA A40", "--gpu", "x"]
    monkeypatch.setattr("sys.argv", ["cost.py", *files, *three])
    with pytest.raises(SystemExit, match="once per file"):
        cost.main()
