"""first_requests.py: per-workload first vs steady latency."""

import first_requests as fr


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_first_request_cost_is_first_minus_the_steady_median():
    clock = Clock()
    costs = {"ticket": [2.0, 0.14, 0.13, 0.15], "article": [5.0, 0.3, 0.3, 0.31]}
    seen = {"ticket": 0, "article": 0}

    def poster(body):
        name = body["name"]
        clock.t += costs[name][seen[name]]
        seen[name] += 1
        return 200

    r = fr.measure(poster, {"ticket": {"name": "ticket"}, "article": {"name": "article"}},
                   repeat=4, clock=clock)  # fmt: skip
    assert r["ticket"]["ms"] == [2000.0, 140.0, 130.0, 150.0]
    assert r["ticket"]["first_minus_steady_ms"] == 2000.0 - 140.0  # median of the rest
    assert r["article"]["first_minus_steady_ms"] == 5000.0 - 300.0
    assert list(r) == ["ticket", "article"]  # workloads in order: first-shape costs stay visible
    assert "| ticket | 2000.0, 140.0, 130.0, 150.0 | 1860.0 ms |" in fr.render("x", r)


def test_a_single_request_has_no_steady_state():
    r = fr.measure(lambda body: 200, {"t": {}}, repeat=1)
    assert r["t"]["first_minus_steady_ms"] is None


def test_steady_state_is_the_median_of_the_rest_at_the_default_repeat():
    clock = Clock()
    costs = iter([3.0, 0.2, 0.1])

    def poster(body):
        clock.t += next(costs)
        return 200

    r = fr.measure(poster, {"t": {}}, repeat=3, clock=clock)
    assert r["t"]["first_minus_steady_ms"] == 3000.0 - 150.0  # median of [200, 100] is 150
    assert r["t"]["statuses"] == [200, 200, 200]


def test_connection_failures_are_status_zero_and_keep_their_time():
    import urllib.error

    clock = Clock()
    calls = iter([None, urllib.error.URLError("refused"), TimeoutError()])

    def poster(body):
        clock.t += 0.5
        exc = next(calls)
        if exc:
            raise exc
        return 200

    r = fr.measure(poster, {"t": {}}, repeat=3, clock=clock)
    assert r["t"]["statuses"] == [200, 0, 0]
    assert r["t"]["ms"] == [500.0, 500.0, 500.0]


def test_render_shows_a_dash_without_a_steady_state():
    r = fr.measure(lambda body: 200, {"t": {}}, repeat=1)
    assert "| - |" in fr.render("x", r)


def test_main_writes_both_files_and_fails_on_a_non_200(tmp_path, monkeypatch):
    import json

    import pytest

    statuses = iter([200, 200, 503, 200, 200, 200])
    monkeypatch.setattr(fr, "http_poster", lambda *a: lambda body: next(statuses))
    monkeypatch.setattr(fr, "load_line", lambda rid, glob: {"id": rid})
    monkeypatch.setattr(fr, "wire_body", lambda line: line)
    out = tmp_path / "first"
    with pytest.raises(SystemExit) as e:
        fr.main(["http://x", str(out), "--workload", "a=r1", "--workload", "b=r2"])
    assert "not answered 200" in str(e.value)
    data = json.loads((out / "first_requests.json").read_text())
    assert data["a"]["statuses"] == [200, 200, 503]
    assert "## First requests: first" in (out / "first_requests.md").read_text()


def test_main_succeeds_when_every_request_is_200(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "http_poster", lambda *a: lambda body: 200)
    monkeypatch.setattr(fr, "load_line", lambda rid, glob: {"id": rid})
    monkeypatch.setattr(fr, "wire_body", lambda line: line)
    fr.main(["http://x", str(tmp_path / "ok"), "--workload", "a=r1"])
    assert (tmp_path / "ok" / "first_requests.json").exists()
