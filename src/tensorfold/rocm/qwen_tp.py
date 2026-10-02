"""Tensor-parallel forward for the ROCm Qwen engine.

Wraps :func:`tensorfold.rocm.qwen_math.forward_hidden` so each layer's per-rank attention / MLP
projection is column-split (already done at load time by :func:`tensorfold.rocm.qwen.slice_for_tp`)
and the per-rank residual addends are reduced across ranks before the addition. The hidden state
stays replicated on every rank; only the column-split projections and the output head are sliced.

Same model, same weights per rank, same activation dtype, same exactness contract: the per-rank fp32
partials are summed rank-order so a row's bits are independent of the world's window, mirroring
``tensorfold.cuda.families.nemotron_h.cuda.tp``'s fp32-then-cast path.
"""

from __future__ import annotations

import torch

from tensorfold.rocm import qwen_math
from tensorfold.rocm.comm import RCCL
from tensorfold.rocm.qwen import TextModel


def all_reduce_local(rccl: RCCL, local: torch.Tensor) -> torch.Tensor:
    """Sum ``local`` across ranks, fp32 partials summed rank-order; cast back to original dtype."""

    if rccl.world <= 1:
        return local
    out_dtype = local.dtype
    if local.dtype != torch.float32:
        local = local.to(torch.float32)
    out = torch.empty_like(local)
    rccl.all_reduce(local, out, op="sum")
    return out.to(out_dtype) if out_dtype != torch.float32 else out


def vocab_gather(rccl: RCCL, local_logits: torch.Tensor) -> torch.Tensor:
    """Concatenate ``local_logits`` across ranks in rank order along the last dim; only rank 0 needs the result."""

    if rccl.world <= 1:
        return local_logits
    if rccl.rank != 0:
        return local_logits
    send = local_logits.contiguous()
    recv = torch.empty(send.shape[:-1] + (send.shape[-1] * rccl.world,), dtype=send.dtype, device=send.device)
    rccl.all_gather(send, recv)
    return recv


def tp_forward_hidden(model: TextModel, tokens: torch.Tensor, caches: list | None, linear,
                      pos0: int, rccl: RCCL, *, act_dtype: torch.dtype | None = None,
                      exact_short: bool = False) -> tuple[torch.Tensor, list]:
    """TP-aware forward: per-layer all-reduce on the column-split O and down projection outputs.

    The replicated hidden state stays full on every rank; only the residual addends cross ranks.
    Same forward contract as :func:`qwen_math.forward_hidden` so the engine's KV cache and prefix
    handoff stay shape-compatible across ranks.
    """

    if rccl.world <= 1:
        return qwen_math.forward_hidden(model, tokens, caches, linear, pos0, act_dtype,
                                        exact_short=exact_short)

    spec = model.spec
    x = qwen_math.gather_rows(model.embed, tokens, dtype=act_dtype)
    if x.device != tokens.device:
        x = x.to(device=tokens.device)
    if not x.is_contiguous():
        x = x.contiguous()
    fresh = caches is None
    new_caches = []
    length = x.shape[1]
    for index, layer in enumerate(model.layers):
        cache = None if fresh else caches[index]
        for start in range(0, length, qwen_math.SPAN):
            stop = min(length, start + qwen_math.SPAN)
            normed = qwen_math.rms_norm(x[:, start:stop], layer.input_norm, spec.eps)
            if spec.full(index):
                y, cache = qwen_math._attention(spec, layer, normed, cache, linear, pos0 + start, exact_short)
            else:
                y, cache = qwen_math._linear_attn(spec, layer, normed, cache, linear, exact_short)
            y = all_reduce_local(rccl, y)
            x[:, start:stop] = x[:, start:stop] + y
            y = qwen_math._mlp(spec, layer,
                               qwen_math.rms_norm(x[:, start:stop], layer.post_norm, spec.eps), linear)
            y = all_reduce_local(rccl, y)
            x[:, start:stop] = x[:, start:stop] + y
        new_caches.append(cache)
    return qwen_math.rms_norm(x, model.final_norm, spec.eps), new_caches


__all__ = ["RCCL", "all_reduce_local", "tp_forward_hidden", "vocab_gather"]