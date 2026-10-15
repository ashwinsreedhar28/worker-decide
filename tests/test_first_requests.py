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
