from __future__ import annotations

import pytest


def test_slice_for_tp_refuses_bad_rank():
    from tensorfold.rocm.qwen import Spec
    from tensorfold.rocm.qwen_tp import TextModel

    model = TextModel(
        spec=Spec(hidden=4096, intermediate=14336, n_layers=2, heads=24, kv_heads=4, head_dim=128,
                  key_heads=4, value_heads=4, key_dim=128, value_dim=128, conv=4, vocab=100000,
                  eps=1e-6, rope_theta=1000000.0, rotary_dim=32, full_every=4, bits=4, group=64),
        embed=None, layers=[], final_norm=None, head=None,
    )
    from tensorfold.rocm.qwen import slice_for_tp
    with pytest.raises(ValueError, match="rank 2 not in"):
        slice_for_tp(model, rank=2, world=2)


def test_slice_for_tp_refuses_indivisible_heads():
    from tensorfold.rocm.qwen import Spec
    from tensorfold.rocm.qwen import slice_for_tp
    from tensorfold.rocm.qwen_tp import TextModel

    model = TextModel(
        spec=Spec(hidden=4096, intermediate=14336, n_layers=2, heads=20, kv_heads=4, head_dim=128,
                  key_heads=4, value_heads=4, key_dim=128, value_dim=128, conv=4, vocab=100000,
                  eps=1e-6, rope_theta=1000000.0, rotary_dim=32, full_every=4, bits=4, group=64),
        embed=None, layers=[], final_norm=None, head=None,
    )
    with pytest.raises(ValueError, match="heads 20 not divisible"):
        slice_for_tp(model, rank=0, world=4)


def test_slice_for_tp_refuses_indivisible_kv_heads():
    from tensorfold.rocm.qwen import Spec
    from tensorfold.rocm.qwen import slice_for_tp
    from tensorfold.rocm.qwen_tp import TextModel

    model = TextModel(
        spec=Spec(hidden=4096, intermediate=14336, n_layers=2, heads=24, kv_heads=3, head_dim=128,
                  key_heads=4, value_heads=4, key_dim=128, value_dim=128, conv=4, vocab=100000,
                  eps=1e-6, rope_theta=1000000.0, rotary_dim=32, full_every=4, bits=4, group=64),
        embed=None, layers=[], final_norm=None, head=None,
    )
    with pytest.raises(ValueError, match="kv_heads 3 not divisible"):
        slice_for_tp(model, rank=0, world=4)


def test_tp_forward_uses_forward_hidden_when_world_one(monkeypatch):
    from tensorfold.rocm import qwen_tp

    seen = {}

    class _FakeRccl:
        world = 1
        rank = 0

    def fake_forward_hidden(model, tokens, caches, linear, pos0, act_dtype, *, exact_short):
        seen["called"] = True
        return "hidden", "caches"

    monkeypatch.setattr(qwen_tp.qwen_math, "forward_hidden", fake_forward_hidden)
    out = qwen_tp.tp_forward_hidden(object(), object(), None, lambda x, y: x, 0, _FakeRccl())
    assert out == ("hidden", "caches")
    assert seen.get("called")


def test_all_reduce_local_returns_local_when_world_one():
    from tensorfold.rocm.qwen_tp import all_reduce_local

    class _LocalRccl:
        world = 1

    tensor = object()
    assert all_reduce_local(_LocalRccl(), tensor) is tensor


def test_vocab_gather_returns_local_when_world_one():
    from tensorfold.rocm.qwen_tp import vocab_gather

    class _LocalRccl:
        world = 1

    tensor = object()
    assert vocab_gather(_LocalRccl(), tensor) is tensor


def test_qwen_engine_sample_skips_logit_compute_on_follower():
    from tensorfold.rocm import engine as engine_mod

    class _FakeEngine:
        tp = 2
        rank = 1
        rccl = object()

        def _follow_token(self):
            self.follow_called = True
            return 7

    inst = _FakeEngine()
    seen = {}

    def fake_project(self, x, packed, linear):
        seen["project"] = True
        raise AssertionError("follower must not project the head")

    engine_mod._project = fake_project
    out = engine_mod.QwenEngine._sample(inst, object(), None, 0, None)
    assert out == 7
    assert inst.follow_called
    assert "project" not in seen