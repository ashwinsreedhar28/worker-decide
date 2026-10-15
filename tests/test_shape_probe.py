"""shape_probe.py: variants with distinct token counts, per-kind medians, op diffs."""

from types import SimpleNamespace

import shape_probe as sp


def test_variants_have_distinct_counts_and_same_length_siblings():
    # A fake tokenizer: one token per word, except that a run of r "x" words is ceil(r / 2)
    # tokens, so every even k repeats the previous count and must be skipped.
    def count(s):
        words, n, run = s.split(), 0, 0
        for w in [*words, ""]:
            if w == "x":
                run += 1
                continue
            n += (run + 1) // 2 + (1 if w else 0)
            run = 0
        return n

    vs = sp.make_variants("a b c", 3, count)
    assert [v["k"] for v in vs] == [1, 3, 5]
    assert [v["tokens"] for v in vs] == [4, 5, 6]
    # " y" * k never merges: same count as " x" * k only at k = 1
    assert [v["sibling"] is not None for v in vs] == [True, False, False]
    assert vs[0]["sibling"] == "a b c y"


def test_variants_stop_at_n_and_give_up_after_max_tries():
    vs = sp.make_variants("a", 4, lambda s: len(s.split()))
    assert [v["k"] for v in vs] == [1, 2, 3, 4]
    assert [v["tokens"] for v in vs] == [2, 3, 4, 5]
    assert all(v["sibling"] is not None for v in vs)  # " y" * k has the same word count
    assert sp.make_variants("a", 4, lambda s: 7, max_tries=10) == []  # never a new count


class T:
    """Just enough of a 1-D tensor for compare_outputs: shape, float, softmax, -, abs, max."""

    def __init__(self, xs):
        self.xs = [float(x) for x in xs]
        self.shape = (len(self.xs),)
        self.dtype = "float32"

    def float(self):
        return self

    def softmax(self, dim):
        import math

        m = max(self.xs)
        e = [math.exp(x - m) for x in self.xs]
        return T([x / sum(e) for x in e])

    def __sub__(self, other):
        return T([a - b for a, b in zip(self.xs, other.xs, strict=True)])

    def abs(self):
        return T([abs(x) for x in self.xs])

    def max(self):
        return max(self.xs)


def _pass(kind, backbone, head=4.0, allocs=0):
    seg = {"encode": 2.0, "collate": 0.3, "pre": 0.1, "backbone": backbone, "head": head,
           "post": 0.1}  # fmt: skip
    return {"kind": kind, **seg, "total": sum(seg.values()), "device_allocs": allocs}


def test_summary_medians_per_kind_and_new_minus_repeat_ignore_traced_passes():
    passes = [_pass("new", 160.0, allocs=5), _pass("new", 170.0, allocs=7), _pass("new", 150.0),
              _pass("repeat", 50.0), _pass("repeat", 52.0), _pass("sibling", 51.0),
              _pass("new-traced", 900.0), _pass("repeat-traced", 800.0)]  # fmt: skip
    s = sp.summarize(passes)
    assert s["new"]["n"] == 3 and s["new"]["backbone"] == 160.0
    assert s["new"]["device_allocs"] == 5
    assert s["repeat"]["backbone"] == 51.0 and s["sibling"]["n"] == 1
    assert s["new_minus_repeat"]["backbone"] == 109.0
    assert s["new_minus_repeat"]["head"] == 0.0
    assert "new-traced" not in s


def test_summary_without_repeats_has_no_difference():
    assert "new_minus_repeat" not in sp.summarize([_pass("new", 1.0)])


def _event(key, cpu, cuda, count, attr="self_device_time_total"):
    return SimpleNamespace(key=key, self_cpu_time_total=cpu, count=count, **{attr: cuda})


def test_op_rows_merge_duplicate_keys_and_read_either_cuda_attribute():
    malloc = _event("cudaMalloc", 300.0, 0.0, 3, attr="self_cuda_time_total")
    mm = [_event("aten::mm", 100.0, 50.0, 2), _event("aten::mm", 10.0, 5.0, 1)]
    rows = sp.op_rows([*mm, malloc])
    assert rows["aten::mm"] == {"cpu_us": 110.0, "cuda_us": 55.0, "count": 3}
    assert rows["cudaMalloc"]["cpu_us"] == 300.0


def test_diff_ranks_the_op_that_grew_most_first_and_keeps_ops_seen_in_one_pass():
    new = {"aten::mm": {"cpu_us": 90_000.0, "cuda_us": 40_000.0, "count": 257},
           "cudaMalloc": {"cpu_us": 5_000.0, "cuda_us": 0.0, "count": 12},
           "aten::add": {"cpu_us": 1_000.0, "cuda_us": 900.0, "count": 296}}  # fmt: skip
    rep = {"aten::mm": {"cpu_us": 6_000.0, "cuda_us": 40_000.0, "count": 257},
           "aten::add": {"cpu_us": 1_100.0, "cuda_us": 900.0, "count": 296}}  # fmt: skip
    d = sp.diff_ops(new, rep, top=2)
    assert [r["op"] for r in d] == ["aten::mm", "cudaMalloc"]
    assert d[0]["d_cpu_ms"] == 84.0 and d[0]["d_cuda_ms"] == 0.0
    assert d[1]["count_new"] == 12 and d[1]["count_repeat"] == 0


def test_describe_reports_shapes_not_values():
    t = SimpleNamespace(shape=(1, 843), dtype="torch.int64")
    d = sp.describe({"input_ids": t, "spans": [[(0, 5)]], "meta": SimpleNamespace(n=2)})
    assert d["input_ids"] == {"tensor": [1, 843], "dtype": "torch.int64"}
    assert d["spans"]["type"] == "list" and d["spans"]["len"] == 1
    assert d["meta"]["fields"]["n"] == "int"


def test_render_has_the_difference_row_and_the_op_table():
    s = sp.summarize([_pass("new", 160.0), _pass("repeat", 50.0)])
    text = sp.render("x", s, [{"op": "aten::mm", "d_cpu_ms": 84.0, "d_cuda_ms": 0.0,
                               "count_new": 257, "count_repeat": 257}])  # fmt: skip
    assert "| new - repeat | | +0.0 | +0.0 | +0.0 | +110.0 | +0.0 | +0.0 | +110.0 | |" in text
    assert "| `aten::mm` | +84.000 | +0.000 | 257 / 257 |" in text


def _fake_world(monkeypatch, cost_ms=110.0):
    """A fake torch, engine and splitter: a pass is cost_ms slower on a token count not seen."""
    import contextlib
    import sys
    import types

    import profile_engine

    import deciders.clef.engine as eng_mod
    import worker.core as core_mod

    stats = {"num_device_alloc": 0}
    torch = types.ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext
    torch.cuda = SimpleNamespace(memory_stats=lambda: dict(stats), synchronize=lambda: None)
    prof_mod = types.ModuleType("torch.profiler")

    class Profile:
        def __init__(self, activities):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def key_averages(self):
            return [SimpleNamespace(key="aten::mm", self_cpu_time_total=1.0, count=1,
                                    self_device_time_total=1.0)]  # fmt: skip

        def export_chrome_trace(self, path):
            open(path, "w").write("{}")

    prof_mod.ProfilerActivity = SimpleNamespace(CPU="cpu", CUDA="cuda")
    prof_mod.profile = Profile
    torch.profiler = prof_mod
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.profiler", prof_mod)

    class Engine:
        def __init__(self, cfg):
            self.device = SimpleNamespace(type="cuda")
            self.model = object()
            self.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
            self.jsm = SimpleNamespace(
                collate_records=lambda recs, pad, dev: {
                    "input_ids": SimpleNamespace(
                        shape=(len(recs), max(len(r.input_ids) for r in recs)), dtype="int64"
                    ),
                    "n": len(recs),
                }
            )

        def _forward(self, batch):  # one 2-option question per record
            return [T([1.0, 0.0]) for _ in range(batch["n"])]

        def encode(self, rec):
            return SimpleNamespace(input_ids=rec["state"].split())

        def manifest(self):
            return {"gpu": "fake"}

    seen = set()

    class Splitter:
        def __init__(self, torch, chain):
            pass

        def one_pass(self, engine, rec):
            n = len(engine.encode(rec).input_ids)
            backbone = 50.0 + (0.0 if n in seen else cost_ms)
            if n not in seen:
                stats["num_device_alloc"] += 3
            seen.add(n)
            seg = {"encode": 2.0, "collate": 0.3, "pre": 0.1, "backbone": backbone, "head": 4.0,
                   "post": 0.1}  # fmt: skip
            return {**seg, "total": sum(seg.values()), "input_tokens": n, "backbone_path": "bb"}

    monkeypatch.setattr(eng_mod, "ClefEngine", Engine)
    monkeypatch.setattr(core_mod, "config_from_env", lambda: None)
    monkeypatch.setattr(profile_engine, "Splitter", Splitter)
    monkeypatch.setattr(profile_engine, "backbone_chain", lambda model: [])

    def record_for(rid, line):
        return {"id": rid, "state": "w " * 40, "questions": {}}

    monkeypatch.setattr(profile_engine, "record_for", record_for)


def test_main_end_to_end_with_a_fake_engine_finds_the_per_shape_cost(tmp_path, monkeypatch):
    import json

    _fake_world(monkeypatch)
    out = tmp_path / "probe"
    sp.main([str(out), "--variants", "4", "--warmup", "2"])
    data = json.loads((out / "shape_probe.json").read_text())
    s = data["summary"]
    assert s["new"]["n"] == 4 and s["repeat"]["n"] == 4 and s["sibling"]["n"] == 4
    assert s["new_minus_repeat"]["backbone"] == 110.0
    assert s["sibling"]["backbone"] == 50.0  # same count, other words: no cost
    assert s["new"]["device_allocs"] == 3 and s["repeat"]["device_allocs"] == 0
    assert data["base_tokens"] == 40
    assert [v["tokens"] for v in data["variants"]] == [41, 42, 43, 44]
    assert json.loads((out / "batch_structure.json").read_text())["input_ids"]["tensor"] == [1, 40]
    assert "| new - repeat |" in (out / "shape_probe.md").read_text()
    assert data["op_diff"] is None
    assert data["env"]["preferred_blas"] is None  # the fake torch has no backends
    bp = data["batch_preview"]
    assert bp["records"] == ["pulse", "card_trio", "long_2k"] and bp["comparable"] is True
    assert bp["max_abs_dprob"] == 0.0 and bp["batch"]["input_ids"]["tensor"] == [3, 40]
    assert json.loads((out / "batch_preview.json").read_text())["lengths"] == [40, 40, 40]
    assert "Batch preview, lengths [40, 40, 40]" in (out / "shape_probe.md").read_text()


def test_main_trace_profiles_one_pair_and_keeps_it_out_of_the_medians(tmp_path, monkeypatch):
    import json

    _fake_world(monkeypatch)
    out = tmp_path / "probe"
    sp.main([str(out), "--variants", "4", "--trace"])
    data = json.loads((out / "shape_probe.json").read_text())
    kinds = [p["kind"] for p in data["passes"]]
    assert kinds.count("new-traced") == 1 and kinds.count("repeat-traced") == 1
    traced_k = [p["k"] for p in data["passes"] if p["kind"] == "new-traced"]
    assert traced_k == [data["variants"][2]["k"]]  # mid-run, past first-use costs
    assert data["summary"]["new"]["n"] == 3  # the traced pair is not in the medians
    assert data["op_diff"][0]["op"] == "aten::mm"
    assert (out / "trace-new.json.gz").exists() and (out / "trace-repeat.json.gz").exists()


def test_main_refuses_a_record_whose_state_is_not_a_string(tmp_path, monkeypatch):
    import profile_engine
    import pytest

    _fake_world(monkeypatch)
    monkeypatch.setattr(profile_engine, "record_for",
                        lambda rid, line: {"id": rid, "state": {"a": 1}})  # fmt: skip
    with pytest.raises(SystemExit, match="not a string"):
        sp.main([str(tmp_path / "p")])


def test_blas_library_reads_torch_or_says_unknown():
    fake = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(
        preferred_blas_library=lambda: "_BlasBackend.Cublaslt")))  # fmt: skip
    assert sp.blas_library(fake) == "_BlasBackend.Cublaslt"
    assert sp.blas_library(SimpleNamespace()) is None


def test_main_without_batch_records_skips_the_preview(tmp_path, monkeypatch):
    import json

    _fake_world(monkeypatch)
    sp.main([str(tmp_path / "p"), "--variants", "2", "--batch-records"])
    assert json.loads((tmp_path / "p" / "shape_probe.json").read_text())["batch_preview"] is None
    assert not (tmp_path / "p" / "batch_preview.json").exists()


def test_flatten_tensors_walks_lists_tuples_and_dicts_in_order():
    a, b, c = T([1]), T([2]), T([3])
    assert sp.flatten_tensors([a, (b, {"x": c, "y": "label"})]) == [a, b, c]
    assert sp.flatten_tensors("not a tensor") == []


def test_compare_outputs_pairs_in_order_and_refuses_misaligned_shapes():
    batched = [T([2.0, 0.0]), T([0.0, 1.0, 0.0])]
    singles = [[T([2.0, 0.0])], [T([0.0, 1.1, 0.0])]]
    r = sp.compare_outputs(batched, singles)
    assert r["comparable"] is True and r["per_tensor_max_abs_dprob"][0] == 0.0
    assert 0.0 < r["max_abs_dprob"] < 0.05
    swapped = sp.compare_outputs(batched, [singles[1], singles[0]])  # wrong order: shapes differ
    assert swapped["comparable"] is False and "max_abs_dprob" not in swapped


def _preview_engine(fail=False, diff=0.0, now=None):
    calls, seen = [], set()

    def collate(recs, pad, dev):
        if fail:
            raise RuntimeError("collate supports one record")
        calls.append(len(recs))
        return {"n": len(recs), "lens": tuple(len(r.input_ids) for r in recs)}

    def forward(batch):  # batched runs are off by `diff` on the first logit
        if now is not None:  # a first-seen shape costs 50 ms, a repeat 5 ms
            now[0] += 0.005 if batch["lens"] in seen else 0.050
            seen.add(batch["lens"])
        bump = diff if batch["n"] > 1 else 0.0
        return [T([1.0 + bump, 0.0]) for _ in range(batch["n"])]

    eng = SimpleNamespace(processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
                          device="cuda", jsm=SimpleNamespace(collate_records=collate),
                          _forward=forward)  # fmt: skip
    return eng, calls


def _preview_torch():
    import contextlib

    return SimpleNamespace(inference_mode=contextlib.nullcontext,
                           cuda=SimpleNamespace(synchronize=lambda: None))  # fmt: skip


def test_batch_preview_times_second_runs_and_reports_the_batched_difference():
    now = [0.0]
    eng, calls = _preview_engine(diff=0.5, now=now)
    recs = [SimpleNamespace(input_ids=[0] * n) for n in (843, 300)]
    r = sp.batch_preview(eng, _preview_torch(), recs, clock=lambda: now[0])
    assert r["lengths"] == [843, 300] and r["comparable"] is True
    assert calls == [2, 2, 2, 1, 1, 1, 1]  # describe, batched x2, each single x2
    assert r["batched_ms"] == 5.0 and r["singles_ms"] == [5.0, 5.0] and r["singles_ms_sum"] == 10.0
    assert r["max_abs_dprob"] > 0.05  # a 0.5 logit shift shows up


def test_batch_preview_records_an_error_instead_of_failing():
    eng, _ = _preview_engine(fail=True)
    r = sp.batch_preview(eng, _preview_torch(), [SimpleNamespace(input_ids=[0])])
    assert r["error"].startswith("RuntimeError: collate supports one record")
    assert "comparable" not in r
