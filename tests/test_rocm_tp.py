"""Tensor-parallel slicing on the host: two ranks in threads, joined by an in-process ring, give the one-rank model.

No GPU. The projection is a dense fp64 dequantization, so the check is the slicing, the K split, where the
ranks' shares are summed, and the order of the vocabulary slices, not the kernels.
"""

from __future__ import annotations

import copy
import threading

import pytest

torch = pytest.importorskip("torch")

from tensorfold.rocm import (
    forward,  # noqa: E402
    qwen_math,  # noqa: E402
)
from tensorfold.rocm.qwen import FullLayer, LinearLayer, TextModel, slice_for_tp  # noqa: E402
from tensorfold.rocm.qwen_math import Packed, Spec  # noqa: E402
from tensorfold.rocm.qwen_tp import all_reduce_local, tp_forward_hidden, vocab_gather  # noqa: E402

BITS, GROUP = 8, 32


def _packed(n: int, k: int, g: torch.Generator) -> Packed:
    words = torch.randint(-2**31, 2**31 - 1, (n, k * BITS // 32), generator=g, dtype=torch.int64).to(torch.int32)
    scale = torch.rand((n, k // GROUP), generator=g) * 0.02 + 0.005
    bias = torch.randn((n, k // GROUP), generator=g) * 0.01 - 1.3
    return Packed(words, scale, bias, BITS, GROUP)


def _model(tied: bool, kv_heads: int = 4) -> TextModel:
    """One linear-attention layer and one full-attention layer. Every K splits into whole groups 4 ways."""

    spec = Spec(hidden=64, intermediate=128, n_layers=2, heads=4, kv_heads=kv_heads, head_dim=32, key_heads=4,
                value_heads=4, key_dim=16, value_dim=32, conv=4, vocab=64, eps=1e-6, rope_theta=10000.0,
                rotary_dim=8, full_every=2, bits=BITS, group=GROUP)
    g = torch.Generator().manual_seed(3)

    def vec(n):
        return torch.randn(n, generator=g).mul(0.1).add(1)

    def mlp():
        return (_packed(spec.intermediate, spec.hidden, g), _packed(spec.intermediate, spec.hidden, g),
                _packed(spec.hidden, spec.intermediate, g))

    width = spec.key_width * 2 + spec.value_width
    linear = LinearLayer(vec(spec.hidden), vec(spec.hidden), _packed(width, spec.hidden, g),
                         _packed(spec.value_width, spec.hidden, g), _packed(spec.value_heads, spec.hidden, g),
                         _packed(spec.value_heads, spec.hidden, g), torch.randn(width, spec.conv, generator=g) * 0.3,
                         torch.randn(spec.value_heads, generator=g), torch.randn(spec.value_heads, generator=g),
                         vec(spec.value_dim), _packed(spec.hidden, spec.value_width, g), *mlp())
    full = FullLayer(vec(spec.hidden), vec(spec.hidden), _packed(spec.heads * spec.head_dim * 2, spec.hidden, g),
                     _packed(spec.kv_heads * spec.head_dim, spec.hidden, g),
                     _packed(spec.kv_heads * spec.head_dim, spec.hidden, g),
                     _packed(spec.hidden, spec.heads * spec.head_dim, g), vec(spec.head_dim), vec(spec.head_dim),
                     *mlp())
    embed = _packed(spec.vocab, spec.hidden, g)
    head = None if tied else _packed(spec.vocab, spec.hidden, g)
    return TextModel(spec, embed, [linear, full], vec(spec.hidden), head)


def _linear(flat: torch.Tensor, packed: Packed) -> torch.Tensor:
    n, groups = packed.scale.shape
    codes = qwen_math._codes(packed.words, packed.bits, groups * packed.group).double().view(n, groups, packed.group)
    weight = (codes * packed.scale.double()[..., None] + packed.bias.double()[..., None]).view(n, -1)
    return (flat.double() @ weight.T).float()


class _Ring:
    """An in-process ring for ``world`` threads: the sum and the gather in rank order."""

    def __init__(self, world: int):
        self.world, self.slots, self.barrier = world, [None] * world, threading.Barrier(world)

    def rank(self, rank: int):
        ring = self

        class Rank:
            world = ring.world

            def all_reduce(self, send, recv, *, op="sum"):
                ring.slots[rank] = send.clone()
                ring.barrier.wait()
                total = ring.slots[0].clone()
                for other in ring.slots[1:]:
                    total += other
                recv.copy_(total)
                ring.barrier.wait()

            def all_gather(self, send, recv):
                ring.slots[rank] = send.clone()
                ring.barrier.wait()
                recv.copy_(torch.stack(ring.slots).view(recv.shape))
                ring.barrier.wait()

        return Rank()


def _run_ranks(model: TextModel, world: int, tokens: torch.Tensor):
    ring, out, errors = _Ring(world), [None] * world, []

    def work(rank):
        try:
            mine = slice_for_tp(copy.deepcopy(model), rank, world)
            comm = ring.rank(rank)
            hidden, _ = tp_forward_hidden(mine, tokens, None, _linear, 0, comm, act_dtype=torch.float32)
            local = forward._project(hidden[:, -1], mine.output_head(), _linear)
            out[rank] = (hidden, vocab_gather(comm, local))
        except Exception as exc:  # noqa: BLE001 - surfaced below, the barrier is broken for the others
            errors.append(exc)
            ring.barrier.abort()

    threads = [threading.Thread(target=work, args=(rank,)) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return out


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("world", [2, 4])
def test_ranks_together_give_the_one_rank_model(world, tied, kv_heads):
    model = _model(tied, kv_heads)
    tokens = torch.tensor([[3, 17, 41, 8, 60, 2, 33]])
    hidden, _ = forward.forward_hidden(copy.deepcopy(model), tokens, None, _linear, 0, torch.float32)
    logits = forward._project(hidden[:, -1], model.output_head(), _linear)
    for rank_hidden, rank_logits in _run_ranks(model, world, tokens):
        assert torch.allclose(rank_hidden, hidden, rtol=1e-4, atol=1e-4)
        assert torch.allclose(rank_logits, logits, rtol=1e-4, atol=1e-4)
        assert torch.equal(rank_logits.argmax(-1), logits.argmax(-1))


def test_each_segment_of_qkv_is_split_by_heads():
    model = _model(tied=False)
    spec = model.spec
    full = model.layers[0].qkv.words
    halves = [slice_for_tp(copy.deepcopy(model), rank, 2).layers[0].qkv.words for rank in range(2)]
    kw, vw = spec.key_width, spec.value_width
    for rank, half in enumerate(halves):
        q, k, v = half.split((kw // 2, kw // 2, vw // 2))
        assert torch.equal(q, full[rank * kw // 2:(rank + 1) * kw // 2])
        assert torch.equal(k, full[kw + rank * kw // 2:kw + (rank + 1) * kw // 2])
        assert torch.equal(v, full[2 * kw + rank * vw // 2:2 * kw + (rank + 1) * vw // 2])


def test_residual_writers_split_whole_groups_and_return_shares():
    model = _model(tied=False)
    parts = [slice_for_tp(copy.deepcopy(model), rank, 2) for rank in range(2)]
    for name, layer in (("out", 0), ("o", 1), ("down", 0), ("down", 1)):
        full = getattr(model.layers[layer], name)
        halves = [getattr(part.layers[layer], name) for part in parts]
        assert all(half.partial for half in halves)
        assert torch.equal(torch.cat([h.words for h in halves], dim=1), full.words)
        assert torch.equal(torch.cat([h.scale for h in halves], dim=1), full.scale)


def test_a_tied_head_keeps_the_full_embedding_for_the_lookup():
    model = _model(tied=True)
    rank1 = slice_for_tp(copy.deepcopy(model), 1, 2)
    assert torch.equal(rank1.embed.words, model.embed.words)
    assert torch.equal(rank1.output_head().words, model.embed.words[32:])
    assert rank1.spec.vocab == model.spec.vocab


@pytest.mark.parametrize("field,value,world", [("heads", 6, 4), ("vocab", 63, 2)])
def test_a_shape_that_does_not_split_is_refused(field, value, world):
    model = _model(tied=False)
    setattr(model.spec, field, value)
    with pytest.raises(ValueError, match=f"{field}: {value} does not split into {world}"):
        slice_for_tp(model, 0, world)


def test_fewer_kv_heads_than_ranks_replicate_each_head_on_its_query_ranks():
    model = _model(tied=False, kv_heads=2)
    full = model.layers[1].k.words
    for rank in range(4):
        mine = slice_for_tp(copy.deepcopy(model), rank, 4)
        assert mine.spec.kv_heads == 1
        assert torch.equal(mine.layers[1].k.words, full[(rank // 2) * 32:(rank // 2 + 1) * 32])


def test_kv_heads_that_do_not_divide_the_ranks_are_refused():
    model = _model(tied=False, kv_heads=3)
    with pytest.raises(ValueError, match="3 KV heads do not divide 4 ranks"):
        slice_for_tp(model, 0, 4)


def test_a_rank_outside_the_world_is_refused():
    with pytest.raises(ValueError, match="rank 2 not in"):
        slice_for_tp(_model(tied=False), 2, 2)


def test_one_rank_keeps_its_tensors():
    class One:
        world = 1

    tensor = torch.ones(2)
    assert all_reduce_local(One(), tensor) is tensor
    assert vocab_gather(One(), tensor) is tensor
