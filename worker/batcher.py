"""Dynamic batching for the worker: requests that arrive while the GPU is busy share a forward.

Tier 1 of the optimization plan. Tier 0 left our worker at 14 req/s at 32 concurrent requests
against SGLang's 96 on the same H100: SGLang batches concurrent requests, ours ran one forward
at a time under the engine lock. A batched forward costs about the same GPU time as a single
one for short inputs (the forward is CPU-bound there), so batching is the throughput lever.

How it works. Request threads call `submit(payload, tokens)` and block on the returned Future.
One background thread takes the oldest pending request, adds compatible ones (`select_batch`),
calls `run_batch(payloads)` once, and resolves each Future with its own result. There is no
artificial wait by default (max_wait_ms=0): requests that arrive while a forward runs form the
next batch, so one request alone pays nothing extra.

Compatibility. The reference collate pads every record to the longest in the batch, so a
300-token request batched with a 16,000-token one would cost a 16,000-token forward per record.
`select_batch` bounds that: lengths within `max_ratio` of the oldest request's, and the padded
size (longest × count) within `max_padded_tokens`. The oldest request always goes, so nothing
starves.

Failures. If a batched run raises (an OOM on a large batch, say), each request is retried alone,
so one bad request or one oversized batch does not fail its neighbours; a request that also
fails alone gets its own exception. If the batching thread itself fails (anything the run does
not catch), every pending request gets an exception and the batcher closes, so no caller waits
forever; callers should still wait with a timeout.

Stats: "batches"/"requests" count runs attempted; "batched_ok" and "max_ok_size" count only
multi-request runs that succeeded as a batch, so a run whose batches all failed and were
retried alone shows max_ok_size 1 (or 0) and retried_alone > 0.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Pending:
    payload: Any
    tokens: int
    future: Future = field(default_factory=Future)
    enqueued: float = 0.0


def select_batch(
    tokens: Sequence[int], max_batch: int, max_padded_tokens: int, max_ratio: float
) -> list[int]:
    """Indices (into `tokens`, oldest first) of the next batch.

    The oldest request (index 0) is always taken. Others are added closest-length first while
    the batch stays within `max_batch` requests, every length is within `max_ratio` of the
    oldest's, and longest × count stays within `max_padded_tokens`. The result is sorted, so
    the batch keeps arrival order.
    """
    if not tokens:
        return []
    first = tokens[0]
    chosen = [0]
    longest = first
    lo, hi = first / max_ratio, first * max_ratio
    candidates = sorted(range(1, len(tokens)), key=lambda i: (abs(tokens[i] - first), i))
    for i in candidates:
        if len(chosen) >= max_batch:
            break
        t = tokens[i]
        if not lo <= t <= hi:
            continue
        new_longest = max(longest, t)
        if new_longest * (len(chosen) + 1) > max_padded_tokens:
            continue
        chosen.append(i)
        longest = new_longest
    return sorted(chosen)


class Batcher:
    """One background thread turning concurrent `submit` calls into `run_batch` calls.

    `run_batch(payloads) -> list` returns one result per payload, in order; it may raise. A
    result that is an Exception instance is set as that request's exception.
    """

    def __init__(
        self,
        run_batch: Callable[[list[Any]], list[Any]],
        max_batch: int = 8,
        max_padded_tokens: int = 32_768,
        max_ratio: float = 2.0,
        max_wait_ms: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        start: bool = True,
    ):
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        if max_ratio < 1:
            raise ValueError(
                "max_ratio must be at least 1 (lengths within max_ratio of each other)"
            )
        if max_padded_tokens < 1:
            raise ValueError("max_padded_tokens must be at least 1")
        self.run_batch = run_batch
        self.max_batch = max_batch
        self.max_padded_tokens = max_padded_tokens
        self.max_ratio = max_ratio
        self.max_wait_s = max_wait_ms / 1000
        self.clock = clock
        self.pending: list[Pending] = []
        self.cv = threading.Condition()
        self.closed = False
        self.stats = {"batches": 0, "requests": 0, "batched_ok": 0, "max_ok_size": 0,
                      "retried_alone": 0, "stopped": None}  # fmt: skip
        self.stats_lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="batcher", daemon=True)
        if start:
            self.thread.start()

    def submit(self, payload: Any, tokens: int) -> Future:
        item = Pending(payload, int(tokens), enqueued=self.clock())
        with self.cv:
            if self.closed:
                raise RuntimeError("batcher is closed")
            self.pending.append(item)
            self.cv.notify()
        return item.future

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify()
        if self.thread.is_alive():
            self.thread.join(timeout=5)

    # -- the batching thread ---------------------------------------------------------------

    def take(self) -> list[Pending]:
        """Block until something is pending (or closed); pick the next batch. Waits up to
        max_wait for more requests when the batch is not full."""
        with self.cv:
            while not self.pending and not self.closed:
                self.cv.wait()
            if not self.pending:
                return []
            if self.max_wait_s > 0 and len(self.pending) < self.max_batch:
                deadline = self.pending[0].enqueued + self.max_wait_s
                while len(self.pending) < self.max_batch and not self.closed:
                    left = deadline - self.clock()
                    if left <= 0:
                        break
                    self.cv.wait(left)
            idx = select_batch([p.tokens for p in self.pending], self.max_batch,
                               self.max_padded_tokens, self.max_ratio)  # fmt: skip
            batch = [self.pending[i] for i in idx]
            keep = set(idx)
            self.pending = [p for i, p in enumerate(self.pending) if i not in keep]
            return batch

    def run(self, batch: list[Pending]) -> None:
        """Run one batch and resolve its futures; retry alone on a batch-level failure."""
        if not batch:
            return
        self.bump(batches=1, requests=len(batch))
        try:
            results = self.run_batch([p.payload for p in batch])
            if len(results) != len(batch):
                raise RuntimeError(f"run_batch returned {len(results)} results for {len(batch)}")
        except Exception as e:  # noqa: BLE001 - forwarded to the waiting requests
            if len(batch) == 1:
                batch[0].future.set_exception(e)
                return
            self.bump(retried_alone=len(batch))
            for p in batch:
                self.run_alone(p)
            return
        if len(batch) > 1:
            self.bump(batched_ok=1, max_ok_size=len(batch))
        else:
            self.bump(max_ok_size=1)
        for p, r in zip(batch, results, strict=True):
            if isinstance(r, Exception):
                p.future.set_exception(r)
            else:
                p.future.set_result(r)

    def run_alone(self, p: Pending) -> None:
        try:
            r = self.run_batch([p.payload])
            if len(r) != 1:
                raise RuntimeError(f"run_batch returned {len(r)} results for 1")
            r = r[0]
        except Exception as e:  # noqa: BLE001
            p.future.set_exception(e)
            return
        if isinstance(r, Exception):
            p.future.set_exception(r)
        else:
            p.future.set_result(r)

    def bump(self, **delta: int) -> None:
        """Counters add; max_ok_size keeps the maximum. Read through snapshot()."""
        with self.stats_lock:
            for k, v in delta.items():
                if k == "max_ok_size":
                    self.stats[k] = max(self.stats[k], v)
                else:
                    self.stats[k] += v

    def snapshot(self) -> dict[str, Any]:
        with self.stats_lock:
            return dict(self.stats)

    def stop(self, reason: BaseException, batch: list[Pending]) -> None:
        """The thread is going down: fail its batch and everything pending, refuse new work."""
        err = RuntimeError(f"batcher stopped: {type(reason).__name__}: {reason}")
        with self.cv:
            self.closed = True
            orphans, self.pending = list(self.pending), []
        with self.stats_lock:
            self.stats["stopped"] = str(err)
        for p in [*batch, *orphans]:
            if not p.future.done():
                p.future.set_exception(err)

    def _loop(self) -> None:
        batch: list[Pending] = []
        try:
            while True:
                batch = self.take()
                if not batch:
                    if self.closed:
                        return
                    continue
                self.run(batch)
                batch = []
        except BaseException as e:  # noqa: BLE001 - nobody else would resolve the futures
            self.stop(e, batch)
            if not isinstance(e, Exception):
                raise
