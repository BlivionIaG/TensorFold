"""Tensor-parallel slicing: each rank keeps its heads, columns, input groups, experts and vocabulary rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorfold.rocm.model.model import FullLayer, MTPHead, TextModel
from tensorfold.rocm.model.qwen_math import Packed, Spec

if TYPE_CHECKING:
    from tensorfold.rocm.model.moe import Routed


def _even(size: int, world: int, name: str) -> int:
    if size % world:
        raise ValueError(f"{name}: {size} does not split into {world} equal parts")
    return size // world


def _rows(packed: Packed, spans: list[tuple[int, int]]) -> Packed:
    """Output rows ``spans`` of ``packed``, in order: one rank's share of a column-split projection."""

    if not isinstance(packed, Packed):
        raise ValueError("only MLX affine projections slice across ranks; run one rank")
    pick = lambda t: torch.cat([t[a:b] for a, b in spans]).contiguous()  # noqa: E731
    return Packed(pick(packed.words), pick(packed.scale), pick(packed.bias), packed.bits, packed.group)


def _row_spans(widths: list[int], rank: int, world: int, name: str) -> list[tuple[int, int]]:
    """This rank's rows of each segment of a concatenated output (q | k | v): each segment split evenly."""

    spans, base = [], 0
    for width in widths:
        part = _even(width, world, name)
        spans.append((base + rank * part, base + (rank + 1) * part))
        base += width
    return spans


def _cols(packed: Packed, rank: int, world: int, name: str) -> Packed:
    """Input columns of one rank, whole groups: a row-split projection whose fp32 outputs the ranks sum."""

    if not isinstance(packed, Packed):
        raise ValueError("only MLX affine projections slice across ranks; run one rank")
    groups = _even(packed.scale.shape[1], world, f"{name} groups")
    words = groups * packed.group * packed.bits // 32
    return Packed(packed.words[:, rank * words:(rank + 1) * words].contiguous(),
                  packed.scale[:, rank * groups:(rank + 1) * groups].contiguous(),
                  packed.bias[:, rank * groups:(rank + 1) * groups].contiguous(), packed.bits, packed.group,
                  partial=True)


def _kv_spans(width: int, kv_heads: int, rank: int, world: int, name: str) -> list[tuple[int, int]]:
    """This rank's k or v rows; with fewer KV heads than ranks each head is kept by its query ranks."""

    if kv_heads >= world:
        return _row_spans([width], rank, world, name)
    if world % kv_heads:
        raise ValueError(f"{name}: {kv_heads} KV heads do not divide {world} ranks")
    head = rank // (world // kv_heads)
    part = width // kv_heads
    return [(head * part, (head + 1) * part)]


def _heads(tensor: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([tensor[a:b] for a, b in spans]).contiguous()


def slice_for_tp(model: TextModel, rank: int, world: int) -> TextModel:
    """Keep this rank's heads, columns, input groups, experts and vocabulary rows (mutates ``model``)."""

    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} not in [0, {world})")
    spec = model.spec
    for name in ("heads", "key_heads", "value_heads"):
        _even(getattr(spec, name), world, name)
    _even(spec.vocab, world, "vocab")
    key_width, value_width = spec.key_width, spec.value_width
    for index, layer in enumerate(model.layers):
        base = f"layer {index}"
        if isinstance(layer, FullLayer):
            layer.q = _rows(layer.q, _row_spans([spec.heads * spec.head_dim * 2], rank, world, f"{base} q_proj"))
            kv = spec.kv_heads * spec.head_dim
            layer.k = _rows(layer.k, _kv_spans(kv, spec.kv_heads, rank, world, f"{base} k_proj"))
            layer.v = _rows(layer.v, _kv_spans(kv, spec.kv_heads, rank, world, f"{base} v_proj"))
            layer.o = _cols(layer.o, rank, world, f"{base} o_proj")
        else:
            qkv = _row_spans([key_width, key_width, value_width], rank, world, f"{base} in_proj_qkv")
            layer.qkv = _rows(layer.qkv, qkv)
            layer.conv = _heads(layer.conv, qkv)
            layer.z = _rows(layer.z, _row_spans([value_width], rank, world, f"{base} in_proj_z"))
            heads = _row_spans([spec.value_heads], rank, world, f"{base} value heads")
            layer.a = _rows(layer.a, heads)
            layer.b = _rows(layer.b, heads)
            layer.a_log = _heads(layer.a_log, heads)
            layer.dt_bias = _heads(layer.dt_bias, heads)
            layer.out = _cols(layer.out, rank, world, f"{base} out_proj")
        if getattr(layer, "moe", None) is not None:
            layer.moe = _experts_share(layer.moe, rank, world, f"{base} experts")
            continue
        mlp = _row_spans([layer.gate.words.shape[0]], rank, world, f"{base} mlp")
        layer.gate = _rows(layer.gate, mlp)
        layer.up = _rows(layer.up, mlp)
        layer.down = _cols(layer.down, rank, world, f"{base} mlp.down_proj")
    model.head = _rows(model.output_head(), _row_spans([spec.vocab], rank, world, "vocab"))
    if model.mtp is not None:
        _slice_mtp(model.mtp, spec, rank, world)
    spec.heads //= world
    spec.kv_heads = max(1, spec.kv_heads // world)
    spec.key_heads //= world
    spec.value_heads //= world
    return model


def _experts_share(routed: "Routed", rank: int, world: int, name: str) -> "Routed":
    """A rank's routed experts: a contiguous ``E / world`` of them, and the shared one on rank 0."""

    from tensorfold.rocm.model.experts import AffineExperts, GptqExperts
    from tensorfold.rocm.model.moe import Routed

    total = routed.count                               # routed experts; the shared one is id ``total``
    part = _even(total, world, name)
    ids = list(range(rank * part, (rank + 1) * part)) + ([total] if rank == 0 else [])
    device = routed.router.device
    keep = torch.tensor(ids, dtype=torch.long, device=device)
    remap = torch.full((total + 1,), -1, dtype=torch.int32, device=device)
    remap[keep] = torch.arange(len(ids), dtype=torch.int32, device=device)
    ex = routed.experts
    if isinstance(ex, AffineExperts):
        take = lambda triple: None if triple is None else tuple(t.index_select(0, keep).contiguous()  # noqa: E731
                                                                 for t in triple)
        experts = AffineExperts(take(ex.up), take(ex.down), take(ex.gate), ex.bits, ex.group, ex.limit,
                                ex.down_bits, ex.down_group)
    elif isinstance(ex, GptqExperts):
        take = lambda t: t.index_select(1, keep).contiguous()  # noqa: E731
        experts = GptqExperts(take(ex.up), take(ex.up_z), take(ex.up_s), take(ex.down), take(ex.down_z),
                              take(ex.down_s), ex.group, ex.v2, ex.limit)
    else:
        raise ValueError(f"{name}: expert kind {type(ex).__name__} has no tp split")
    return Routed(routed.router, experts, routed.top_k, remap=remap, partial=True)


def _slice_mtp(head: "MTPHead", spec: Spec, rank: int, world: int) -> None:
    """Under tp the MTP head stays whole on every rank but its logits rows, so ranks draft alike."""

    if head.head is not None:
        head.head = _rows(head.head, _row_spans([spec.vocab], rank, world, "mtp head"))
