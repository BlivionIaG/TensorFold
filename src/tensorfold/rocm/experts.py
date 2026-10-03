"""Grouped MoE experts on RDNA, grouped by TensorFold's plan: MLX affine or W4A16 GPTQ, dispatched per checkpoint.

Same plan contract as ``tensorfold.cuda.experts`` (``members`` and ``items``), so a router and a combine
built for the group path drive either weight kind. Affine experts run the packed affine matmul one expert
item at a time; W4A16 experts run the grouped fp16 kernels. An item owns its output rows, so no atomics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

TILE = 16
PREFILL_TILE = 64
SMALL = 1024


def max_items(pairs: int, experts: int, tile: int = TILE) -> int:
    """Items a plan of ``pairs`` can hold: an item per used expert, plus one per ``tile`` pairs past its first."""

    return min(pairs, experts) + pairs // tile


class Plan:
    """Scratch grouping pairs by expert: ``members`` (pair ids) and ``items`` (expert, first, count)."""

    def __init__(self, rows: int, slots: int, experts: int, device: torch.device | str, *,
                 prefill: bool = False) -> None:
        pairs = rows * slots
        self.rows, self.slots, self.experts, self.prefill = rows, slots, experts, prefill
        self.tile = PREFILL_TILE if prefill else TILE
        self.members = torch.empty((pairs,), dtype=torch.int32, device=device)
        self.items = torch.zeros((max_items(pairs, experts, TILE), 3), dtype=torch.int32, device=device)
        self.count = 0


def route(picks: torch.Tensor, plan: Plan, tile: int = PREFILL_TILE) -> None:
    """``picks`` [R, slots] int32, contiguous: each pair's expert; the runs land in ``plan.items``.

    ``members`` is the pair ids sorted by expert, so an item's rows are contiguous. Every slot is routed,
    so the items partition ``0 .. pairs - 1`` exactly once. The tail of ``items`` is zeroed, since the
    W4A16 kernel takes the whole tensor and reads count 0 as an item it skips.
    """

    rows, slots = picks.shape
    if slots != plan.slots or rows > plan.rows:
        raise ValueError(f"picks {tuple(picks.shape)} do not fit a plan of {plan.rows} x {plan.slots}")
    plan.tile = tile if plan.prefill else TILE
    flat = picks.reshape(-1)[: rows * slots].to(torch.int32)
    plan.members[: flat.numel()] = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat.to(torch.int64), minlength=plan.experts)
    starts = torch.cumsum(counts, 0) - counts
    reps = (counts + plan.tile - 1) // plan.tile
    total = int(reps.sum())
    index = torch.repeat_interleave(torch.arange(plan.experts, device=flat.device), reps)
    offset = torch.cumsum(reps, 0) - reps
    within = torch.arange(total, device=flat.device) - offset[index]
    first = starts[index] + within * plan.tile
    span = torch.minimum(counts[index] - within * plan.tile, torch.full_like(index, plan.tile))
    plan.items[:total, 0] = index
    plan.items[:total, 1] = first
    plan.items[:total, 2] = span
    plan.items[total:] = 0
    plan.count = total


@dataclass
class AffineExperts:
    """One layer's MLX affine experts, the shared expert last.

    Each side is a (words, scales, biases) triple whose leading dim is E + 1: ``up`` is the expert's up (or
    its single relu^2 projection), ``gate`` the gate of a SwiGLU pair or None, and ``down`` the output side.
    """

    up: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    down: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    gate: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    bits: int
    group: int
    limit: float = 0.0
    kind: str = field(default="affine", init=False)

    @property
    def count(self) -> int:
        return self.up[0].shape[0]

    @property
    def swiglu(self) -> bool:
        return self.gate is not None

    @property
    def width(self) -> int:
        return self.up[0].shape[1]

    @property
    def dims(self) -> int:
        return self.down[0].shape[1]


@dataclass
class GptqExperts:
    """One layer's W4A16 GPTQ experts, the shared expert last.

    ``up`` is (mats, E + 1, D / 8, NI) int32 with ``mats`` 2 for a fused gate+up and 1 for relu^2, and
    ``down`` is (1, E + 1, NI / 8, D). ``v2`` is True for AWQ's literal zeros and False for GPTQ's +1.
    """

    up: torch.Tensor
    up_z: torch.Tensor
    up_s: torch.Tensor
    down: torch.Tensor
    down_z: torch.Tensor
    down_s: torch.Tensor
    group: int
    v2: bool = False
    limit: float = 0.0
    kind: str = field(default="gptq", init=False)

    @property
    def count(self) -> int:
        return self.up.shape[1]

    @property
    def swiglu(self) -> bool:
        return self.up.shape[0] == 2

    @property
    def width(self) -> int:
        return self.up.shape[3]

    @property
    def dims(self) -> int:
        return self.up.shape[2] * 8


Experts = AffineExperts | GptqExperts


def _affine(triple, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return triple[0][expert], triple[1][expert], triple[2][expert]


def _items(ex: "AffineExperts | GptqExperts", plan: Plan) -> list[tuple[int, int, int]]:
    """The plan's items with work: a tp rank zeroes the count of pairs another rank's experts own."""

    return [item for item in plan.items[: plan.count].tolist() if item[2] > 0]


def gate_up(x: torch.Tensor, ex, plan: Plan, rows: int, *, block_m: int = 4) -> torch.Tensor:
    """``x`` [R, D] bf16 or fp16 -> [R * slots, NI], each pair's activated expert output.

    The W4A16 kernels take bf16, so that kind is normalized here; the affine kind keeps the caller's
    activation dtype, which is fp16 on RDNA2.
    """

    if isinstance(ex, GptqExperts):
        from tensorfold.rocm import qgemm

        return qgemm.moe(x.to(torch.bfloat16), ex.up.contiguous(), ex.up_z.contiguous(), ex.up_s.contiguous(),
                         plan.items, plan.members, rows, plan.slots, epi=2 if ex.swiglu else 1, block_m=block_m,
                         use_v2_format=ex.v2, limit=ex.limit)

    from tensorfold.rocm import affine as affine_mod

    out = torch.empty((rows * plan.slots, ex.width), dtype=torch.bfloat16, device=x.device)
    group = ex.group
    for expert, first, count in _items(ex, plan):
        idx = plan.members[first:first + count].to(torch.int64)
        xr = x.index_select(0, idx // plan.slots)
        u = affine_mod.matmul(xr, *_affine(ex.up, expert), bits=ex.bits, group=group, f32=True)
        if ex.gate is None:
            act = torch.nn.functional.relu(u).square()
        else:
            g = affine_mod.matmul(xr, *_affine(ex.gate, expert), bits=ex.bits, group=group, f32=True)
            if ex.limit > 0:
                g = g.clamp(max=ex.limit)
                u = u.clamp(-ex.limit, ex.limit)
            act = torch.nn.functional.silu(g) * u
        out.index_copy_(0, idx, act.to(torch.bfloat16))
    return out


def down(act: torch.Tensor, ex, plan: Plan, rows: int, *, block_m: int = 4) -> torch.Tensor:
    """``act`` [R * slots, NI] -> [R * slots, D] fp32, each pair's expert output."""

    if isinstance(ex, GptqExperts):
        from tensorfold.rocm import qgemm

        return qgemm.moe(act, ex.down.contiguous(), ex.down_z.contiguous(), ex.down_s.contiguous(), plan.items,
                         plan.members, rows, plan.slots, epi=0, block_m=block_m, use_v2_format=ex.v2)

    from tensorfold.rocm import affine as affine_mod

    out = torch.empty((rows * plan.slots, ex.dims), dtype=torch.float32, device=act.device)
    for expert, first, count in _items(ex, plan):
        idx = plan.members[first:first + count].to(torch.int64)
        out.index_copy_(0, idx, affine_mod.matmul(act.index_select(0, idx), *_affine(ex.down, expert),
                                                  bits=ex.bits, group=ex.group, f32=True))
    return out
