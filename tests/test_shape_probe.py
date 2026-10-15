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
    torch.cuda = SimpleNamespace(memory_stats=lambda: dict(stats))
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
                    "input_ids": SimpleNamespace(shape=(1, len(recs[0].input_ids)), dtype="int64")
                }
            )

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
