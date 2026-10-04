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
    return forward.forward_hidden(model, tokens, caches, linear, pos0, act_dtype, exact_short=exact_short,
                                    reduce=partial(all_reduce_local, rccl), at=at)


__all__ = ["RCCL", "all_reduce_local", "tp_forward_hidden", "vocab_gather"]
