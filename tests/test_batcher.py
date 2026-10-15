"""worker/batcher.py: batch selection, result routing, failure isolation, the thread."""

import threading
import time

import pytest

from worker.batcher import Batcher, select_batch


def test_the_oldest_always_goes_even_alone():
    assert select_batch([16_384], max_batch=8, max_padded_tokens=1_000, max_ratio=2.0) == [0]
    assert select_batch([], 8, 1_000, 2.0) == []


def test_lengths_outside_the_ratio_wait_for_a_later_batch():
    # oldest 800: 300 and 2,486 are outside 2x, 843 and 700 are in
    assert select_batch([800, 300, 843, 2_486, 700], 8, 100_000, 2.0) == [0, 2, 4]


def test_padded_size_and_count_bound_the_batch_closest_lengths_first():
    # oldest 1,000; budget 3,300 padded tokens: 1,000 + 1,010 (x2 = 2,020), + 1,100 (x3 = 3,300)
    # fits, + 1,200 (x4 = 4,800) does not
    assert select_batch([1_000, 1_200, 1_010, 1_100], 8, 3_300, 2.0) == [0, 2, 3]
    assert select_batch([1_000, 1_010, 1_020, 1_030], 2, 100_000, 2.0) == [0, 1]
    # a longer request raises the padded size for everyone already chosen
    assert select_batch([1_000, 1_900, 1_000], 8, 3_000, 2.0) == [0, 2]


def test_ties_on_distance_keep_arrival_order():
    assert select_batch([500, 510, 490, 510], 3, 100_000, 2.0) == [0, 1, 2]


def unstarted(run_batch, **kw):
    return Batcher(run_batch, start=False, **kw)


def test_one_run_for_compatible_requests_and_each_gets_its_own_result():
    calls = []

    def run_batch(payloads):
        calls.append(list(payloads))
        return [p.upper() for p in payloads]

    b = unstarted(run_batch, max_batch=8)
    futs = [b.submit(p, t) for p, t in (("a", 800), ("b", 300), ("c", 843))]
    b.run(b.take())  # a and c (300 is outside 2x of 800)
    b.run(b.take())  # b alone
    assert calls == [["a", "c"], ["b"]]
    assert [f.result(timeout=1) for f in futs] == ["A", "B", "C"]
    assert b.snapshot() == {"batches": 2, "requests": 3, "batched_ok": 1, "max_ok_size": 2,
                            "retried_alone": 0, "stopped": None}  # fmt: skip


def test_a_failed_batch_is_retried_alone_so_one_bad_request_fails_only_itself():
    def run_batch(payloads):
        if len(payloads) > 1:
            raise RuntimeError("CUDA out of memory")
        if payloads[0] == "bad":
            raise RuntimeError("kernel failed")
        return [payloads[0] + "!"]

    b = unstarted(run_batch)
    good, bad = b.submit("ok", 100), b.submit("bad", 100)
    b.run(b.take())
    assert good.result(timeout=1) == "ok!"
    with pytest.raises(RuntimeError, match="kernel failed"):
        bad.result(timeout=1)
    assert b.snapshot()["retried_alone"] == 2
    assert b.snapshot()["batched_ok"] == 0  # the failed batch does not count as batched


def test_exception_results_and_wrong_result_counts():
    b = unstarted(lambda ps: [ValueError("rejected") if p == "x" else p for p in ps])
    fx, fy = b.submit("x", 10), b.submit("y", 10)
    b.run(b.take())
    with pytest.raises(ValueError, match="rejected"):
        fx.result(timeout=1)
    assert fy.result(timeout=1) == "y"

    short = unstarted(lambda ps: ps[:1])  # loses a result when batched: retried alone
    f1, f2 = short.submit("p", 10), short.submit("q", 10)
    short.run(short.take())
    assert (f1.result(timeout=1), f2.result(timeout=1)) == ("p", "q")

    empty = unstarted(lambda ps: [])
    f = empty.submit("p", 10)
    empty.run(empty.take())
    with pytest.raises(RuntimeError, match="0 results for 1"):
        f.result(timeout=1)


def test_a_single_request_failure_is_not_retried():
    calls = []

    def run_batch(ps):
        calls.append(ps)
        raise RuntimeError("boom")

    b = unstarted(run_batch)
    f = b.submit("p", 10)
    b.run(b.take())
    with pytest.raises(RuntimeError, match="boom"):
        f.result(timeout=1)
    assert len(calls) == 1


def test_closed_batcher_refuses_new_work_and_its_thread_exits():
    b = Batcher(lambda ps: ps)
    assert b.submit("p", 1).result(timeout=2) == "p"
    b.close()
    assert not b.thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        b.submit("q", 1)
    with pytest.raises(ValueError):
        Batcher(lambda ps: ps, max_batch=0, start=False)
    with pytest.raises(ValueError, match="max_ratio"):
        Batcher(lambda ps: ps, max_ratio=0.5, start=False)  # 0 would divide by zero in take()
    with pytest.raises(ValueError, match="max_padded_tokens"):
        Batcher(lambda ps: ps, max_padded_tokens=0, start=False)


def test_requests_arriving_during_a_forward_share_the_next_one():
    started, release = threading.Event(), threading.Event()
    sizes = []

    def run_batch(ps):
        sizes.append(len(ps))
        if len(sizes) == 1:  # hold the first forward until the others are queued
            started.set()
            release.wait(2)
        return [p * 2 for p in ps]

    b = Batcher(run_batch, max_batch=8)
    first = b.submit(1, 300)
    assert started.wait(2)
    rest = [b.submit(i, 300) for i in range(2, 7)]
    release.set()
    assert first.result(timeout=2) == 2
    assert [f.result(timeout=2) for f in rest] == [4, 6, 8, 10, 12]
    assert sizes == [1, 5]  # no artificial wait: the first went alone, the rest together
    b.close()


def test_max_wait_collects_a_burst_before_running():
    sizes = []
    b = Batcher(lambda ps: sizes.append(len(ps)) or ps, max_batch=4, max_wait_ms=200)
    futs = [b.submit(i, 100) for i in range(3)]
    assert [f.result(timeout=2) for f in futs] == [0, 1, 2]
    assert sizes == [3]
    b.close()


def test_max_wait_stops_early_when_the_batch_is_full():
    sizes = []
    b = Batcher(lambda ps: sizes.append(len(ps)) or ps, max_batch=2, max_wait_ms=5_000)
    t0 = time.monotonic()
    futs = [b.submit(i, 100) for i in range(2)]
    assert [f.result(timeout=2) for f in futs] == [0, 1]
    assert time.monotonic() - t0 < 2  # did not sit out the 5 s window
    assert sizes == [2]
    b.close()


def test_a_longer_request_raises_the_padding_of_later_choices():
    # 1,100 joins (2 x 1,100 = 2,200); 850 would make 3 x 1,100 = 3,300 > 3,200, though
    # 3 x 1,000 (the oldest's length) would have fit
    assert select_batch([1_000, 1_100, 850], 8, 3_200, 2.0) == [0, 1]


def test_max_wait_counts_from_the_oldest_requests_arrival():
    b = unstarted(lambda ps: ps, max_batch=4, max_wait_ms=300)
    b.submit("p", 10)
    time.sleep(0.2)
    t0 = time.monotonic()
    assert [p.payload for p in b.take()] == ["p"]
    assert time.monotonic() - t0 < 0.25  # ~0.1 s left of the window, not a fresh 0.3 s


def test_close_interrupts_a_max_wait_and_pending_work_is_still_answered():
    b = unstarted(lambda ps: ps, max_batch=4, max_wait_ms=5_000)
    b.submit("p", 10)
    got = []
    t = threading.Thread(target=lambda: got.append(b.take()))
    t.start()
    time.sleep(0.05)
    with b.cv:
        b.closed = True
        b.cv.notify()
    t.join(1)
    assert not t.is_alive() and [p.payload for p in got[0]] == ["p"]

    slow = Batcher(lambda ps: time.sleep(0.05) or ps, max_batch=1)
    futs = [slow.submit(i, 10) for i in range(4)]
    slow.close()  # drains what was submitted before close
    assert [f.result(timeout=2) for f in futs] == [0, 1, 2, 3]


def test_an_exception_result_on_the_retry_alone_path_is_raised():
    def run_batch(ps):
        if len(ps) > 1:
            raise RuntimeError("OOM")
        return [ValueError("rejected") if ps[0] == "x" else ps[0]]

    b = unstarted(run_batch)
    fx, fy = b.submit("x", 10), b.submit("y", 10)
    b.run(b.take())
    with pytest.raises(ValueError, match="rejected"):
        fx.result(timeout=1)
    assert fy.result(timeout=1) == "y"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # re-raised
def test_if_the_thread_dies_every_waiting_request_fails_and_new_work_is_refused():
    started = threading.Event()

    def run_batch(ps):
        started.set()
        time.sleep(0.05)
        raise KeyboardInterrupt  # a BaseException: not caught by the per-batch handler

    b = Batcher(run_batch, max_batch=1)
    first = b.submit("a", 10)
    assert started.wait(2)
    second = b.submit("b", 10)  # pending while the first batch runs
    for f in (first, second):
        with pytest.raises(RuntimeError, match="batcher stopped: KeyboardInterrupt"):
            f.result(timeout=2)
    b.thread.join(2)
    assert not b.thread.is_alive() and b.snapshot()["stopped"]
    with pytest.raises(RuntimeError, match="closed"):
        b.submit("c", 10)
