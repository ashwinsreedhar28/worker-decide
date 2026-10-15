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
    assert "| ticket | 2000.0, 140.0, 130.0, 150.0 | 1860.0 ms | all 200 |" in fr.render("x", r)


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


def test_failed_requests_are_left_out_of_the_steady_state_and_shown():
    clock = Clock()
    replies = iter([(3.0, 200), (0.1, 0), (0.2, 200), (0.3, 200)])

    def poster(body):
        dt, status = next(replies)
        clock.t += dt
        return status

    r = fr.measure(poster, {"t": {}}, repeat=4, clock=clock)
    assert r["t"]["first_minus_steady_ms"] == 3000.0 - 250.0  # median of [200, 300], not 100
    assert "| 200, 0, 200, 200 |" in fr.render("x", r)


def test_an_http_protocol_error_is_status_zero():
    import http.client

    def poster(body):
        raise http.client.IncompleteRead(b"")  # an HTTPException that is not an OSError

    assert fr.measure(poster, {"t": {}}, repeat=2)["t"]["statuses"] == [0, 0]


def test_main_fails_on_a_connection_failure(tmp_path, monkeypatch):
    import pytest

    def poster(body):
        raise ConnectionRefusedError()

    monkeypatch.setattr(fr, "http_poster", lambda *a: poster)
    monkeypatch.setattr(fr, "load_line", lambda rid, glob: {"id": rid})
    monkeypatch.setattr(fr, "wire_body", lambda line: line)
    with pytest.raises(SystemExit):
        fr.main(["http://x", str(tmp_path / "f"), "--workload", "a=r1", "--repeat", "1"])


def test_main_defaults_send_the_three_load_test_shapes_from_the_fixtures(tmp_path, monkeypatch):
    seen, args = [], []

    def make_poster(*a):
        args.append(a)
        return lambda body: seen.append(body) or 200

    monkeypatch.setattr(fr, "http_poster", make_poster)
    fr.main(["http://x", str(tmp_path / "d"), "--repeat", "1"])
    assert args == [("http://x", "/v1/systemone", None, 600.0)]  # a cold first request takes 34 s
    data = __import__("json").loads((tmp_path / "d" / "first_requests.json").read_text())
    assert list(data) == ["ticket", "pulse", "article"]
    assert len(seen) == 3 and all(b.get("questions") for b in seen)
