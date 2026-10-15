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


def test_the_chain_descends_the_backbone_and_stops_above_the_layer_list():
    chain = [p for p, _ in pe.backbone_chain(clef_like())]
    assert chain == ["backbone", "backbone.model", "backbone.model.language_model"]


def test_override_replaces_the_chain():
    model = clef_like()
    ((path, mod),) = pe.backbone_chain(model, "backbone.model")
    assert path == "backbone.model" and mod is model.backbone.model


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
