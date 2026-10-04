"""The Qwen3.5 ROCm MTP drafter (Phase 2: forward + chain + absorb)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from tensorfold.rocm import forward, qwen_math
from tensorfold.rocm.qwen import MTPHead, Packed, TextModel


@dataclass
class MTPState:
    k: torch.Tensor = None
    v: torch.Tensor = None
    pos: int = 0
    capacity: int = 0

    def reset(self) -> None:
        if self.k is not None:
            self.k.zero_()
            self.v.zero_()
        self.pos = 0


def _alloc_cache(spec, batch: int, total: int, device: torch.device, dtype: torch.dtype,
                 kv_heads: int | None = None) -> dict:
    if kv_heads is None:
        kv_heads = spec.kv_heads
    shape = (batch, kv_heads, total, spec.head_dim)
    return {"k": torch.empty(shape, device=device, dtype=dtype),
            "v": torch.empty(shape, device=device, dtype=dtype),
            "len": 0}


def _grow_cache(cache: dict, needed: int, dtype: torch.dtype) -> None:
    """Extend the head's KV cache to at least ``needed`` rows; matches _blank_caches' growth policy."""

    if cache["k"].shape[2] >= needed:
        return
    new_total = max(needed, cache["k"].shape[2] * 2)
    new_shape = (cache["k"].shape[0], cache["k"].shape[1], new_total, cache["k"].shape[3])
    new_k = torch.empty(new_shape, device=cache["k"].device, dtype=dtype)
    new_v = torch.empty(new_shape, device=cache["k"].device, dtype=dtype)
    new_k[:, :, :cache["len"]].copy_(cache["k"][:, :, :cache["len"]])
    new_v[:, :, :cache["len"]].copy_(cache["v"][:, :, :cache["len"]])
    cache["k"], cache["v"] = new_k, new_v


def _attention(head: MTPHead, x: torch.Tensor, cache: dict | None, position: int, dtype: torch.dtype,
               linear: Callable[[torch.Tensor, Packed], torch.Tensor], spec,
               heads: int, kv_heads: int) -> tuple[torch.Tensor, dict]:
    batch, length, _ = x.shape
    qg = linear(x, head.q)
    keys = linear(x, head.k)
    values = linear(x, head.v)
    if head.gated:
        queries, gate = qg.view(batch, length, heads, spec.head_dim * 2).split(spec.head_dim, dim=-1)
    else:
        queries, gate = qg.view(batch, length, heads, spec.head_dim), None
    keys = keys.view(batch, length, kv_heads, spec.head_dim)
    values = values.view(batch, length, kv_heads, spec.head_dim)
    queries = qwen_math.rms_norm(queries, head.q_norm, spec.eps).permute(0, 2, 1, 3)
    keys = qwen_math.rms_norm(keys, head.k_norm, spec.eps).permute(0, 2, 1, 3)
    values = values.permute(0, 2, 1, 3)
    queries = qwen_math.apply_rope(queries, position, spec.rope_theta, spec.rotary_dim, exact=False)
    keys = qwen_math.apply_rope(keys, position, spec.rope_theta, spec.rotary_dim, exact=False)
    if cache is None:
        shape = (batch, kv_heads, length + position, spec.head_dim)
        cache = {"k": torch.empty(shape, device=x.device, dtype=dtype),
                 "v": torch.empty(shape, device=x.device, dtype=dtype),
                 "len": 0}
    end = cache["len"] + length
    if end > cache["k"].shape[2]:
        _grow_cache(cache, end, dtype)
    cache["k"][:, :, cache["len"]:end] = keys.to(dtype=dtype)
    cache["v"][:, :, cache["len"]:end] = values.to(dtype=dtype)
    cache["len"] = end
    kept_k = cache["k"][:, :, :end]
    kept_v = cache["v"][:, :, :end]
    scale = spec.head_dim ** -0.5
    if queries.dtype != torch.float32:
        queries_f = queries.float()
    else:
        queries_f = queries
    attended = forward._attend(queries_f, kept_k, kept_v, scale, position)
    attended = attended.permute(0, 2, 1, 3).reshape(batch, length, -1)
    if gate is not None:
        attended = attended * torch.sigmoid(gate.reshape(batch, length, -1).float())
    if attended.dtype != dtype:
        attended = attended.to(dtype=dtype)
    out = linear(attended, head.o)
    return out, cache


def _vocab_logits(head: MTPHead, model: TextModel, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    target = head.head if head.head is not None else model.output_head()
    flat = x.reshape(-1, x.shape[-1]).to(dtype=dtype).contiguous()
    from tensorfold.rocm import affine as affine_mod

    out = affine_mod.matmul(flat, target.words, target.scale, target.bias,
                            bits=target.bits, group=target.group, schedule="auto")
    return out.view(*x.shape[:-1], -1)


class MTPEngine:
    def __init__(self, model: TextModel, head: MTPHead, *,
                 linear: Callable[[torch.Tensor, Packed], torch.Tensor], rccl=None):
        self.model = model
        self.head = head
        self.linear = linear
        self.rccl = rccl
        # The head's own q/k/v shapes give its heads: spec may be sliced under tp.
        spec = model.spec
        self._heads = int(head.q.words.shape[0]) // int(spec.head_dim) // (2 if head.gated else 1)
        self._kv_heads = int(head.k.words.shape[0]) // int(spec.head_dim)

    def fresh_cache(self, *, batch: int, total: int, device: torch.device, dtype: torch.dtype) -> dict:
        # The head's own kv_heads; spec.kv_heads may be sliced under TP.
        return _alloc_cache(self.model.spec, batch, total, device, dtype, self._kv_heads)

    def _logits(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Draft logits; under tp the vocabulary slices are joined so every rank drafts the same ids."""

        target = self.head.head if self.head.head is not None else self.model.output_head()
        out = _vocab_logits(self.head, self.model, x, dtype)
        if self.rccl is None or self.rccl.world <= 1 or target.words.shape[0] == self.model.spec.vocab:
            return out
        from tensorfold.rocm.qwen_tp import vocab_gather

        rows = vocab_gather(self.rccl, out.reshape(-1, out.shape[-1]))
        return rows.view(*out.shape[:-1], -1)

    def forward(self, hidden: torch.Tensor, next_token: torch.Tensor, position: int, cache: dict, *,
                dtype: torch.dtype | None = None) -> torch.Tensor:
        spec = self.model.spec
        if dtype is None:
            dtype = hidden.dtype
        emb = qwen_math.gather_rows(self.model.embed, next_token, dtype=dtype)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        emb_e = qwen_math.rms_norm(emb, self.head.fc_e_norm, spec.eps)
        emb_h = qwen_math.rms_norm(hidden, self.head.fc_h_norm, spec.eps)
        e_proj = self.linear(emb_e.view(-1, emb_e.shape[-1]), self.head.fc_e).view(*emb_e.shape[:-1], -1)
        h_proj = self.linear(emb_h.view(-1, emb_h.shape[-1]), self.head.fc_h).view(*emb_h.shape[:-1], -1)
        x = e_proj + h_proj
        if self.head.input_norm is not None:
            x = qwen_math.rms_norm(x, self.head.input_norm, spec.eps)
        attn_out, cache = _attention(self.head, x, cache, position, dtype, self.linear, spec,
                                     self._heads, self._kv_heads)
        x = x + attn_out
        if self.head.post_norm is not None:
            x = x + forward._mlp(spec, self.head, qwen_math.rms_norm(x, self.head.post_norm, spec.eps),
                                   self.linear)
        residual = qwen_math.rms_norm(x, self.head.final_norm, spec.eps)
        return self._logits(residual, dtype), residual

    def draft_chain(self, hidden: torch.Tensor, last_token: int, position: int, depth: int, cache: dict, *,
                    sampling, dtype: torch.dtype) -> list[int]:
        """``depth`` drafts from ``position``; draft i is keyed by its slot, as the verifier keys it."""
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        from tensorfold.engine.exact_sampling import MARGIN, choose

        device, ids = hidden.device, []
        cur_hidden, cur_token = hidden, last_token
        for step in range(depth):
            tok_input = torch.as_tensor([cur_token], dtype=torch.long, device=device)
            logits, residual = self.forward(cur_hidden, tok_input, position + step, cache, dtype=dtype)
            row = logits.detach().float().reshape(-1)
            k = int(sampling.top_k)
            sample_key = position + step + 1
            if sampling is None or float(sampling.temperature) <= 0.0:
                nxt = int(torch.argmax(row).item())
            elif k:
                count = min(int(row.shape[0]), k + MARGIN)
                values, index = torch.topk(row, count, sorted=False)
                nxt = int(choose(values.cpu().numpy(),
                                 index.cpu().numpy().astype("int64"),
                                 sample_key, sampling))
            else:
                nxt = int(choose(row.cpu().numpy(),
                                 torch.arange(int(row.shape[0]), dtype=torch.int64).numpy(),
                                 sample_key, sampling))
            ids.append(nxt)
            cur_token = nxt
            cur_hidden = residual[:, -1:, :]
        return ids

    def absorb(self, hidden_per_step: Sequence[torch.Tensor], tokens: Sequence[int], cache: dict, *,
               dtype: torch.dtype) -> dict:
        device = hidden_per_step[0].device
        for step, t in enumerate(tokens):
            cur = hidden_per_step[step] if step < len(hidden_per_step) else hidden_per_step[-1]
            if cur.dim() == 1:
                cur = cur.unsqueeze(0)
            tok = torch.as_tensor([t], dtype=torch.long, device=device)
            self.forward(cur, tok, cache["len"], cache, dtype=dtype)
        return cache


__all__ = ["MTPEngine", "MTPState"]