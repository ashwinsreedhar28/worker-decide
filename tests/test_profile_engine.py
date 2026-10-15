"""profile_engine.py's torch-free parts: backbone discovery, choice of the timed module, stats."""

import profile_engine as pe
import pytest


class P:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


class M:
    """Duck-typed nn.Module: named_children and parameters (own + descendants)."""

    def __init__(self, own=0, **children):
        self.own, self.children = own, children
        for k, v in children.items():
            setattr(self, k, v)

    def named_children(self):
        return list(self.children.items())

    def parameters(self):
        yield P(self.own)
        for c in self.children.values():
            yield from c.parameters()


class ModuleList(M):
    pass


def clef_like():
    """The shape Clef-flash has: a 9B backbone (vision + language model + 1B lm_head table)
    next to a 122M joint schema head."""
    layers = ModuleList(**{str(i): M(own=200) for i in range(32)})  # 6.4B in layers
    language = M(embed=M(own=1000), layers=layers)  # 7.4B
    backbone = M(model=M(visual=M(own=400), language_model=language), lm_head=M(own=1000))
    return M(backbone=backbone, head=M(own=122))


def test_the_chain_descends_the_backbone_and_ends_with_a_layer_span():
    model = clef_like()
    chain = pe.backbone_chain(model)
    assert [p for p, _, _ in chain] == [
        "backbone",
        "backbone.model",
        "backbone.model.language_model",
        "backbone.model.language_model.layers[first..last]",
    ]
    _, first, last = chain[-1]
    layers = model.backbone.model.language_model.layers
    assert first is layers.children["0"] and last is layers.children["31"]


def test_override_replaces_the_chain():
    model = clef_like()
    ((path, start, end),) = pe.backbone_chain(model, "backbone.model")
    assert path == "backbone.model" and start is end is model.backbone.model


def test_a_model_without_a_dominant_child_needs_an_override():
    with pytest.raises(SystemExit):
        pe.backbone_chain(M(a=M(own=10), b=M(own=10)))


def test_the_outermost_module_that_ran_is_timed():
    chain = ["backbone", "backbone.model", "backbone.model.language_model"]
    # A wrapper calling the inner model directly never runs the outer module's forward.
    assert pe.outermost_fired(chain, {"backbone.model", "backbone.model.language_model"}) == (
        "backbone.model"
    )
    assert pe.outermost_fired(chain, set(chain)) == "backbone"
    assert pe.outermost_fired(chain, set()) is None


def test_aggregate_medians_and_shares():
    samples = [
        {"encode": 1, "collate": 1, "pre": 0, "backbone": 80, "head": 15, "post": 3, "total": 100},
        {"encode": 3, "collate": 1, "pre": 0, "backbone": 90, "head": 25, "post": 1, "total": 120},
        {"encode": 2, "collate": 1, "pre": 0, "backbone": 85, "head": 20, "post": 2, "total": 110},
    ]
    a = pe.aggregate(samples)
    assert a["median_ms"]["backbone"] == 85 and a["median_ms"]["total"] == 110
    assert a["share_of_total"]["head"] == round(20 / 110, 3)
    assert a["head_plus_overhead_share"] == round(1 - 85 / 110, 3)


def test_module_tree_lists_params():
    tree = pe.module_tree(clef_like(), depth=2)
    by = {t["path"]: t for t in tree}
    assert by["head"]["params"] == 122 and by["backbone.lm_head"]["params"] == 1000


# -- Splitter with a fake torch: event arithmetic, first-span guard, outermost choice --------


class FakeClock:
    t = 0.0


class FakeEvent:
    def __init__(self, enable_timing=True):
        self.t = None

    def record(self):
        self.t = FakeClock.t

    def elapsed_time(self, other):  # ms from self to other, like torch.cuda.Event
        return other.t - self.t


class FakeTorch:
    class cuda:
        Event = FakeEvent

        @staticmethod
        def synchronize():
            pass

    class inference_mode:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False


class Hooked:
    """A module that fires forward pre/post hooks and advances the fake clock by `cost`."""

    def __init__(self, cost, inner=None):
        self.cost, self.inner, self.pre, self.post = cost, inner, [], []

    def register_forward_pre_hook(self, fn):
        self.pre.append(fn)

    def register_forward_hook(self, fn):
        self.post.append(fn)

    def __call__(self, x):
        for f in self.pre:
            f(self, (x,))
        out = self.inner(x) if self.inner else x
        FakeClock.t += self.cost
        for f in self.post:
            f(self, (x,), out)
        return out


class Logit:
    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [0.0, 1.0]

    def softmax(self, dim):
        return self


def fake_engine(backbone_call, head_cost=3.0, pre_cost=1.0):
    class E:
        device = "cuda"

        class processor:
            class tokenizer:
                pad_token_id = 0

        class jsm:
            @staticmethod
            def collate_records(recs, pad, dev):
                return recs

        @staticmethod
        def encode(rec):
            class Enc:
                input_ids = [1] * 300

            return Enc()

        @staticmethod
        def _forward(batch):
            FakeClock.t += pre_cost
            backbone_call(batch)
            FakeClock.t += head_cost
            return [Logit()]

    return E


def test_splitter_times_the_outermost_module_that_ran():
    inner = Hooked(cost=10.0)
    outer = Hooked(cost=0.0, inner=inner)  # never called: the wrapper calls `inner` directly
    split = pe.Splitter(FakeTorch, [("outer", outer, outer), ("inner", inner, inner)])
    r = split.one_pass(fake_engine(inner), {})
    assert r["backbone_path"] == "inner"
    assert r["pre"] == 1.0 and r["backbone"] == 10.0 and r["head"] == 3.0  # signs and order


def test_splitter_keeps_the_first_span_of_a_module_called_twice():
    bb = Hooked(cost=10.0)

    def twice(batch):
        bb(batch)
        FakeClock.t += 5.0
        bb(batch)  # a second call must not stretch the backbone span over the gap

    split = pe.Splitter(FakeTorch, [("bb", bb, bb)])
    r = split.one_pass(fake_engine(twice), {})
    assert r["backbone"] == 10.0 and r["head"] == 3.0 + 5.0 + 10.0


def test_splitter_falls_back_to_the_layer_span():
    first, last = Hooked(cost=4.0), Hooked(cost=4.0)
    wrapper = Hooked(cost=0.0)  # hooked but never run

    def run_layers(batch):
        first(batch)
        last(batch)

    split = pe.Splitter(FakeTorch, [("w", wrapper, wrapper), ("w.layers", first, last)])
    r = split.one_pass(fake_engine(run_layers), {})
    assert r["backbone_path"] == "w.layers" and r["backbone"] == 8.0
