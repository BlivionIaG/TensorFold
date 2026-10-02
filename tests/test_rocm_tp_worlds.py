from __future__ import annotations

import pytest


def _spec(*, hidden=4096, intermediate=14336, n_layers=2, heads=24, kv_heads=4, head_dim=128,
          key_heads=4, value_heads=4, key_dim=128, value_dim=128, conv=4, vocab=100000,
          eps=1e-6, rope_theta=1000000.0, rotary_dim=32, full_every=4, bits=4, group=64):
    from tensorfold.rocm.qwen import Spec
    return Spec(hidden=hidden, intermediate=intermediate, n_layers=n_layers,
                heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                key_heads=key_heads, value_heads=value_heads,
                key_dim=key_dim, value_dim=value_dim,
                conv=conv, vocab=vocab, eps=eps, rope_theta=rope_theta,
                rotary_dim=rotary_dim, full_every=full_every, bits=bits, group=group)


def _model_with_spec(spec):
    from tensorfold.rocm.qwen_tp import TextModel
    return TextModel(spec=spec, embed=None, layers=[], final_norm=None, head=None)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_qwen3_5_text_default_spec_splits_under_tp_world(world):
    from tensorfold.rocm.qwen import slice_for_tp

    model = _model_with_spec(_spec())
    sliced = slice_for_tp(model, rank=0, world=world)
    assert sliced.spec.heads == 24 // world
    assert sliced.spec.kv_heads == 4 // world
    assert sliced.spec.vocab == 100000 // world


@pytest.mark.parametrize("world", [3, 5, 6, 7, 9])
def test_qwen3_5_text_default_spec_refuses_non_power_of_two(world):
    from tensorfold.rocm.qwen import slice_for_tp

    model = _model_with_spec(_spec())
    if world == 1:
        slice_for_tp(model, rank=0, world=1)
    else:
        with pytest.raises(ValueError):
            slice_for_tp(model, rank=0, world=world)


def test_qwen3_5_8_kv_heads_does_not_admit_world_8_if_spec_says_4():
    from tensorfold.rocm.qwen import slice_for_tp

    spec = _spec(heads=24, kv_heads=4)
    for world in (1, 2, 4):
        model = _model_with_spec(spec)
        slice_for_tp(model, rank=0, world=world)
    model = _model_with_spec(spec)
    with pytest.raises(ValueError, match="kv_heads 4 not divisible"):
        slice_for_tp(model, rank=0, world=8)


def test_qwen3_5_32_kv_heads_admits_world_8():
    from tensorfold.rocm.qwen import slice_for_tp

    spec = _spec(heads=32, kv_heads=32, value_heads=32, key_heads=32)
    for world in (1, 2, 4, 8):
        model = _model_with_spec(spec)
        sliced = slice_for_tp(model, rank=0, world=world)
        assert sliced.spec.kv_heads == 32 // world


def test_slice_for_tp_refuses_rank_out_of_bounds_for_world_8():
    from tensorfold.rocm.qwen import slice_for_tp

    model = _model_with_spec(_spec(heads=32, kv_heads=32, value_heads=32, key_heads=32))
    slice_for_tp(model, rank=7, world=8)
    model = _model_with_spec(_spec(heads=32, kv_heads=32, value_heads=32, key_heads=32))
    with pytest.raises(ValueError, match="rank 8 not in"):
        slice_for_tp(model, rank=8, world=8)