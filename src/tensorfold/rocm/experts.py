"""MoE experts over TensorFold's plan (``members``, ``items``): MLX affine or W4A16 GPTQ / AWQ."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

TILE = 16
PREFILL_TILE = 64
SMALL = 1024
LANE_ROWS = 8      # the affine decode tile's most rows
BLOCK_ROWS = 128   # the affine prefill GEMM tile's rows
BLOCK_FROM = 64    # rows from which an expert's run is long enough for the GEMM tile


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
        self.items = torch.zeros((max_items(pairs, experts, min(TILE, LANE_ROWS)), 3), dtype=torch.int32,
                                 device=device)
        self.count = 0


def tile_for(ex: "Experts", rows: int, prefill: bool) -> int:
    """Most pairs an item holds: the affine decode tile for short inputs, its GEMM tile past them, or W4A16's."""

    if isinstance(ex, AffineExperts):
        return LANE_ROWS if rows < BLOCK_FROM else BLOCK_ROWS
    return PREFILL_TILE if prefill else TILE


def route(picks: torch.Tensor, plan: Plan, tile: int = PREFILL_TILE) -> None:
    """Group the pairs of ``picks`` [R, slots] by expert into ``plan`` without a host sync."""

    rows, slots = picks.shape
    if slots != plan.slots or rows > plan.rows:
        raise ValueError(f"picks {tuple(picks.shape)} do not fit a plan of {plan.rows} x {plan.slots}")
    plan.tile = tile
    flat = picks.reshape(-1)[: rows * slots].to(torch.int32)
    plan.members[: flat.numel()] = torch.argsort(flat, stable=True).to(torch.int32)
    ids = flat.to(torch.int64)
    counts = torch.zeros(plan.experts, dtype=torch.int64, device=flat.device).index_add_(0, ids, torch.ones_like(ids))
    starts = torch.cumsum(counts, 0) - counts
    reps = (counts + tile - 1) // tile
    ends = torch.cumsum(reps, 0)
    slot = torch.arange(plan.items.shape[0], device=flat.device)
    expert = torch.searchsorted(ends, slot, right=True).clamp_(max=plan.experts - 1)
    within = slot - (ends[expert] - reps[expert])
    plan.items[:, 0] = expert
    plan.items[:, 1] = starts[expert] + within * tile
    plan.items[:, 2] = (counts[expert] - within * tile).clamp(0, tile)
    plan.count = plan.items.shape[0]


@dataclass
class AffineExperts:
    """A layer's MLX affine experts, shared expert last; gate and up stacked once at load."""

    up: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    down: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    gate: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    bits: int
    group: int
    limit: float = 0.0
    down_bits: int = 0                 # the down side's width and group when they differ (0: bits / group)
    down_group: int = 0
    kind: str = field(default="affine", init=False)
    fused: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Gate and up stacked as one (E + 1, 2 NI, ...) weight for one launch; ``gate`` and ``up`` view its halves."""

        self.down_bits, self.down_group = self.down_bits or self.bits, self.down_group or self.group

        if self.gate is None:
            self.fused = tuple(t.contiguous() for t in self.up)
            self.up = self.fused
            return
        width = self.up[0].shape[1]
        self.fused = tuple(torch.cat([g, u], dim=1) for g, u in zip(self.gate, self.up))
        self.gate = tuple(t[:, :width] for t in self.fused)
        self.up = tuple(t[:, width:] for t in self.fused)

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
    """A layer's W4A16 GPTQ / AWQ experts, shared expert last; ``v2`` is AWQ's literal zeros."""

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


def one_launch(ex, x: torch.Tensor) -> bool:
    """Affine experts run every item in one launch a projection: FP16 dot2 on RDNA2, BF16 dot2 on gfx11 / gfx12."""

    if not isinstance(ex, AffineExperts):
        return False
    if x.dtype == torch.float16:
        return True
    from tensorfold.rocm.build import WMMA, gfx_name

    return x.dtype == torch.bfloat16 and gfx_name() in WMMA


def _activate(u: torch.Tensor, g: torch.Tensor | None, limit: float) -> torch.Tensor:
    if g is None:
        return torch.nn.functional.relu(u).square()
    if limit > 0:
        g = g.clamp(max=limit)
        u = u.clamp(-limit, limit)
    return torch.nn.functional.silu(g) * u


def gate_up(x: torch.Tensor, ex, plan: Plan, rows: int, *, block_m: int = 4) -> torch.Tensor:
    """``x`` [R, D] -> [R * slots, NI], each pair's activated expert output."""

    if isinstance(ex, GptqExperts):
        from tensorfold.rocm import qgemm

        return qgemm.moe(x.to(torch.bfloat16), ex.up.contiguous(), ex.up_z.contiguous(), ex.up_s.contiguous(),
                         plan.items, plan.members, rows, plan.slots, epi=2 if ex.swiglu else 1, block_m=block_m,
                         use_v2_format=ex.v2, limit=ex.limit)

    from tensorfold.rocm import affine as affine_mod

    if one_launch(ex, x):
        both = affine_mod.matmul_routed(x, *ex.fused, plan.items[:plan.count], plan.members, pairs=rows * plan.slots,
                                        x_div=plan.slots, rows=min(plan.tile, rows), bits=ex.bits, group=ex.group)
        if ex.gate is None:
            return _activate(both, None, ex.limit).to(x.dtype)
        from tensorfold.rocm.act import moe_act

        return moe_act(both, x.dtype, ex.limit)
    out = torch.empty((rows * plan.slots, ex.width), dtype=x.dtype, device=x.device)
    group = ex.group
    for expert, first, count in _items(ex, plan):
        idx = plan.members[first:first + count].to(torch.int64)
        xr = x.index_select(0, idx // plan.slots)
        u = affine_mod.matmul(xr, *_affine(ex.up, expert), bits=ex.bits, group=group, f32=True)
        g = None
        if ex.gate is not None:
            g = affine_mod.matmul(xr, *_affine(ex.gate, expert), bits=ex.bits, group=group, f32=True)
        out.index_copy_(0, idx, _activate(u, g, ex.limit).to(x.dtype))
    return out


def down(act: torch.Tensor, ex, plan: Plan, rows: int, *, block_m: int = 4) -> torch.Tensor:
    """``act`` [R * slots, NI] -> [R * slots, D] fp32, each pair's expert output."""

    if isinstance(ex, GptqExperts):
        from tensorfold.rocm import qgemm

        return qgemm.moe(act, ex.down.contiguous(), ex.down_z.contiguous(), ex.down_s.contiguous(), plan.items,
                         plan.members, rows, plan.slots, epi=0, block_m=block_m, use_v2_format=ex.v2)

    from tensorfold.rocm import affine as affine_mod

    if one_launch(ex, act):
        return affine_mod.matmul_routed(act, *ex.down, plan.items[:plan.count], plan.members, pairs=rows * plan.slots,
                                        x_div=1, rows=min(plan.tile, rows), bits=ex.down_bits, group=ex.down_group)
    out = torch.empty((rows * plan.slots, ex.dims), dtype=torch.float32, device=act.device)
    for expert, first, count in _items(ex, plan):
        idx = plan.members[first:first + count].to(torch.int64)
        out.index_copy_(0, idx, affine_mod.matmul(act.index_select(0, idx), *_affine(ex.down, expert),
                                                  bits=ex.down_bits, group=ex.down_group, f32=True))
    return out
