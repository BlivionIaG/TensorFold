"""Softmax top-k MoE with a sigmoid-gated shared expert on RDNA2, eager torch control over W4A16 experts.

Parallel to ``tensorfold.cuda.moe``: same slots, same weights, same combine, but the control path is
eager torch on this backend (no triton), and the experts are the checkpoint's kind through
``tensorfold.rocm.experts``: the packed affine matmul or the grouped W4A16 kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.rocm import experts as grouped


def router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> None:
    """``out`` [R, E + 1] fp32 = ``x`` [R, D] bf16 . ``rows`` [E + 1, D] bf16, the shared gate row last."""

    torch.mm(x.float(), rows.float().t(), out=out)


def select_rows(logits: torch.Tensor, buf: "MoEBuffers", top_k: int, experts: int) -> None:
    """Each row's top-k experts and weights (buf.pick, buf.wts); slot ``top_k`` is the shared expert.

    Largest fp32 logit first, the lower id among equal logits (a stable descending argsort), the
    weights renormalized exp(l_k - l_0) rounded through bf16, and the shared slot's weight the
    sigmoid of the bf16-rounded shared gate logit.
    """

    rows = logits.shape[0]
    gates = logits[:rows, :experts]
    order = torch.argsort(gates, dim=1, descending=True, stable=True)[:, :top_k]
    picked = torch.gather(gates, 1, order)
    weights = torch.exp(picked - picked[:, :1])
    weights = (weights / weights.sum(dim=1, keepdim=True)).to(torch.bfloat16).to(torch.float32)
    shared = logits[:rows, experts].to(torch.bfloat16).to(torch.float32)
    sgw = torch.sigmoid(shared).to(torch.bfloat16).to(torch.float32)
    buf.pick[:rows, :top_k] = order
    buf.pick[:rows, top_k] = experts
    buf.wts[:rows, :top_k] = weights
    buf.wts[:rows, top_k] = sgw


def select(logits: torch.Tensor, buf: "MoEBuffers", top_k: int, experts: int,
           tile: int = grouped.PREFILL_TILE) -> None:
    """Each row's experts and weights, then the pairs grouped by expert, ``tile`` a prompt's item."""

    select_rows(logits, buf, top_k, experts)
    grouped.route(buf.pick[: logits.shape[0]], buf.plan, tile)


def moe(x: torch.Tensor, router_rows: torch.Tensor, ex: grouped.Experts, buf: "MoEBuffers", top_k: int,
        experts: int) -> "MoEBuffers":
    """Route ``x`` [R, D] and run its experts into ``buf.y`` [R, k + 1, D]; slot k is the shared expert."""

    rows = x.shape[0]
    router(x, router_rows, buf.logits[:rows])
    select(buf.logits[:rows], buf, top_k, experts)
    buf.act[:rows] = grouped.gate_up(x, ex, buf.plan, rows).view(rows, buf.slots, ex.width)
    buf.y[:rows] = grouped.down(buf.act[:rows].view(-1, ex.width), ex, buf.plan, rows).view(rows, buf.slots, ex.dims)
    return buf


def combine(y: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """``y`` [R, S, D] fp32 and ``wts`` [R, S] fp32 -> [R, D] bf16, the slots summed in order, rounded once."""

    rows, slots, _ = y.shape
    return (y.float() * wts[:rows, :slots].unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


@dataclass
class MoEBuffers:
    """Static scratch for up to ``rows`` rows; ``prefill`` picks the experts' prompt arithmetic."""

    rows: int
    slots: int
    logits: torch.Tensor
    pick: torch.Tensor
    wts: torch.Tensor
    plan: grouped.Plan
    act: torch.Tensor
    y: torch.Tensor

    def __init__(self, rows: int, cfg, device: torch.device | str, *, prefill: bool = False) -> None:
        # Normal tensors, not inference ones: the engine forwards inside inference_mode and these buffers
        # are cached process-wide, so a caller outside that mode could not update them in place otherwise.
        with torch.inference_mode(False):
            slots = cfg.num_experts_per_tok + 1
            self.rows, self.slots = rows, slots
            self.logits = torch.empty((rows, cfg.num_experts + 1), dtype=torch.float32, device=device)
            self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
            self.wts = torch.empty((rows, slots), dtype=torch.float32, device=device)
            self.plan = grouped.Plan(rows, slots, cfg.num_experts + 1, device, prefill=prefill)
            self.act = torch.empty((rows, slots, cfg.moe_intermediate_size), dtype=torch.bfloat16, device=device)
            self.y = torch.empty((rows, slots, cfg.hidden_size), dtype=torch.float32, device=device)


@dataclass
class Routed:
    """A layer's router rows [E + 1, D] bf16 (the shared expert's gate row last) and its E + 1 experts.

    Under tp a rank holds some of the experts: ``remap`` [E + 1] int32 maps a global expert id to its row in
    ``experts`` or -1, and ``partial`` makes :func:`run` return the rank's fp32 share for the all-reduce.
    """

    router: torch.Tensor
    experts: grouped.Experts
    top_k: int
    remap: torch.Tensor | None = None
    partial: bool = False

    @property
    def count(self) -> int:
        return self.router.shape[0] - 1


class _Shape:
    def __init__(self, m: Routed) -> None:
        self.num_experts_per_tok, self.num_experts = m.top_k, m.count
        self.moe_intermediate_size, self.hidden_size = m.experts.width, m.experts.dims


_scratch: dict[tuple, MoEBuffers] = {}


def run(x: torch.Tensor, m: Routed, *, prefill: bool = False) -> torch.Tensor:
    """``x`` [R, D] bf16 -> [R, D] bf16: its top-k experts by renormalized softmax weight, plus the shared one.

    With ``m.remap`` (a tp rank) only the rank's experts run, and the result is its fp32 share, unrounded.
    """

    rows = x.shape[0]
    size = 1 << max(4, (rows - 1).bit_length())
    key = (size, m.count, m.top_k, m.experts.width, m.experts.dims, prefill, x.device)
    buf = _scratch.get(key)
    if buf is None:
        buf = _scratch[key] = MoEBuffers(size, _Shape(m), x.device, prefill=prefill)
    if m.remap is None:
        moe(x.contiguous(), m.router, m.experts, buf, m.top_k, m.count)
        return combine(buf.y[:rows], buf.wts[:rows])
    return _run_share(x.contiguous(), m, buf, prefill)


def _run_share(x: torch.Tensor, m: Routed, buf: MoEBuffers, prefill: bool) -> torch.Tensor:
    """One rank's experts: every rank picks the same pairs, runs the ones it holds and sums them in slot order."""

    rows = x.shape[0]
    router(x, m.router, buf.logits[:rows])
    select_rows(buf.logits[:rows], buf, m.top_k, m.count)
    local = m.remap[buf.pick[:rows].long()]
    mine = local >= 0
    skip = m.experts.count                         # an id past the rank's experts: its items do no work
    picks = torch.where(mine, local, torch.full_like(local, skip)).to(torch.int32).contiguous()
    grouped.route(picks, buf.plan, grouped.PREFILL_TILE if prefill else grouped.TILE)
    items = buf.plan.items[:buf.plan.count]
    items[items[:, 0] == skip, 2] = 0
    buf.act[:rows] = grouped.gate_up(x, m.experts, buf.plan, rows).view(rows, buf.slots, m.experts.width)
    y = grouped.down(buf.act[:rows].view(-1, m.experts.width), m.experts, buf.plan, rows).view(rows, buf.slots, -1)
    y = torch.where(mine.unsqueeze(-1), y, torch.zeros((), dtype=y.dtype, device=y.device))
    return (y * buf.wts[:rows, :buf.slots].unsqueeze(-1)).sum(dim=1)
