"""loadtest.py: closed-loop cells, warm-up exclusion, error counting, percentiles."""

import threading
import time

import loadtest as lt


def test_percentiles_are_nearest_rank():
    v = [float(x) for x in range(1, 129)]  # 128 values
    assert lt.percentile(v, 0.99) == 127.0 and lt.percentile(v, 0.5) == 64.0
    s = lt.summarize([x / 1000 for x in v], errors=2, wall_s=2.0)
    assert s["n"] == 128 and s["p99_ms"] == 127.0 and s["req_per_s"] == 64.0
    assert s["errors"] == 2 and s["max_ms"] == 128.0
    assert lt.summarize([], errors=3, wall_s=1.0)["req_per_s"] == 0.0


def test_warmup_is_sent_but_not_counted():
    calls = []
    lock = threading.Lock()

    def poster(body):
        with lock:
            calls.append(1)
        return 200

    s = lt.run_cell(poster, {}, concurrency=4, requests=20, warmup=4)
    assert len(calls) == 24 and s["n"] == 20 and s["errors"] == 0


def test_errors_and_exceptions_are_counted_not_raised():
    n = {"i": 0}
    lock = threading.Lock()

    def poster(body):
        with lock:
            n["i"] += 1
            i = n["i"]
        if i % 3 == 0:
            raise ConnectionResetError("dropped")
        return 500 if i % 3 == 1 else 200

    s = lt.run_cell(poster, {}, concurrency=2, requests=30, warmup=0)
    assert s["n"] + s["errors"] == 30 and s["errors"] == 20


def test_a_serialized_server_shows_flat_throughput_and_queueing_latency():
    # What our worker is today (one forward at a time behind a lock): more concurrency adds
    # queueing latency, not throughput. SGLang batching should break this pattern.
    gpu = threading.Lock()

    def poster(body):
        with gpu:
            time.sleep(0.01)
        return 200

    c1 = lt.run_cell(poster, {}, concurrency=1, requests=20, warmup=1)
    c4 = lt.run_cell(poster, {}, concurrency=4, requests=20, warmup=4)
    assert c4["p50_ms"] > 2.5 * c1["p50_ms"]
    assert c4["req_per_s"] < 1.5 * c1["req_per_s"]


def test_render_has_one_row_per_cell():
    cells = [
        {"workload": "ticket", "input_tokens": 300, "concurrency": c,
         "stats": lt.summarize([0.1, 0.2], 0, 1.0)}
        for c in (1, 8)
    ]  # fmt: skip
    text = lt.render("ours-a40", cells)
    assert text.count("| ticket | 300 |") == 2


def test_default_workloads_resolve_to_fixture_records():
    assert lt.load_line("card_trio", "fixtures/requests/*.jsonl")["id"] == "card_trio"
    assert lt.load_line(None, "fixtures/requests/*.jsonl")["id"].startswith("pulse-")


def test_warmup_time_is_not_in_the_wall_time():
    # Slow warm-up, fast counted phase: throughput must reflect the counted phase only.
    n = {"i": 0}
    lock = threading.Lock()

    def poster(body):
        with lock:
            n["i"] += 1
            i = n["i"]
        time.sleep(0.2 if i <= 2 else 0.001)
        return 200

    s = lt.run_cell(poster, {}, concurrency=2, requests=20, warmup=2)
    assert s["wall_s"] < 0.15 and s["req_per_s"] > 100


def test_the_model_field_is_added_only_when_asked(monkeypatch):
    sent = []

    class Resp:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(lt.urllib.request, "urlopen", lambda req, timeout: sent.append(
        __import__("json").loads(req.data)) or Resp())  # fmt: skip
    body = {"state": "s", "questions": {"z": {}, "a": {}}}
    lt.http_poster("http://x", "/v1/systemone", None, 5)(body)
    lt.http_poster("http://x", "/v1/systemone", "Cloudflare/clef-flash", 5)(body)
    assert "model" not in sent[0] and sent[1]["model"] == "Cloudflare/clef-flash"
    assert list(sent[0]["questions"]) == ["z", "a"]  # order kept on the wire
