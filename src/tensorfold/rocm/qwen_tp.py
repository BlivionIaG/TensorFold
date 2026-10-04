"""Tensor-parallel forward: fp32 shares summed before each residual add, vocabulary joined in rank order."""

from __future__ import annotations

from functools import partial

import torch

from tensorfold.rocm import forward
from tensorfold.rocm.comm import RCCL
from tensorfold.rocm.qwen import TextModel


def all_reduce_local(rccl: RCCL, local: torch.Tensor) -> torch.Tensor:
    """The ranks' shares summed in fp32. Every rank gets the same sum."""

    if rccl.world <= 1:
        return local
    local = local.float().contiguous()
    out = torch.empty_like(local)
    rccl.all_reduce(local, out, op="sum")
    return out


def ordered_sum(rccl: RCCL, local: torch.Tensor) -> torch.Tensor:
    """The ranks' fp32 shares added in rank order, so a row's sum does not depend on how many rows share the call."""

    if rccl.world <= 2:                     # one add: the same bits in any order
        return all_reduce_local(rccl, local)
    local = local.float().contiguous()
    parts = torch.empty((rccl.world, *local.shape), dtype=local.dtype, device=local.device)
    rccl.all_gather(local, parts)
    out = parts[0] + parts[1]
    for share in parts[2:]:
        out += share
    return out


def vocab_gather(rccl: RCCL, local_logits: torch.Tensor) -> torch.Tensor:
    """(rows, vocab / world) slices joined to (rows, vocab) in rank order, on every rank."""

    if rccl.world <= 1:
        return local_logits
    send = local_logits.contiguous()
    rows, width = send.shape
    recv = torch.empty((rccl.world, rows, width), dtype=send.dtype, device=send.device)
    rccl.all_gather(send, recv)
    return recv.permute(1, 0, 2).reshape(rows, rccl.world * width)


def tp_forward_hidden(model: TextModel, tokens: torch.Tensor, caches: list | None, linear,
                      pos0: int, rccl: RCCL, *, act_dtype: torch.dtype | None = None,
                      exact_short: bool = False, at=None) -> tuple[torch.Tensor, list]:
    """:func:`forward.forward_hidden` with the ranks' residual shares summed. ``at`` as there."""

    if rccl.world <= 1:
        return forward.forward_hidden(model, tokens, caches, linear, pos0, act_dtype,
                                        exact_short=exact_short, at=at)
    # Past two ranks RCCL's add order follows the message size: prefill sums in rank order, decode keeps RCCL's.
    reduce = partial(ordered_sum if exact_short else all_reduce_local, rccl)
    return forward.forward_hidden(model, tokens, caches, linear, pos0, act_dtype, exact_short=exact_short,
                                    reduce=reduce, at=at)


__all__ = ["RCCL", "all_reduce_local", "ordered_sum", "tp_forward_hidden", "vocab_gather"]
