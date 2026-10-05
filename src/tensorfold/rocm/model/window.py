"""A lane round's forward: every stream's window in one pass, each row with its serial decode step's bits."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tensorfold.rocm.model.forward import _attend, _mlp, _project, _project_group, _project_pair, _residual
from tensorfold.rocm.model.qwen_math import apply_rope, causal_conv, gated_delta, gather_rows, normalize_qk, rms_norm


@dataclass
class Window:
    """One stream's rows this round: its last token and drafts from slot ``pos``, over the stream's caches."""

    tokens: list[int]
    caches: list[dict]
    pos: int
    states: dict = field(default_factory=dict)   # linear layer index -> [(conv, state) after each row]


def window_forward(model, windows: list[Window], linear, act_dtype: torch.dtype, *, reduce=None) -> torch.Tensor:
    """The final-normed hidden rows of every window, concatenated in order; ``reduce`` sums tp shares."""

    spec = model.spec
    device = model.embed.words.device if hasattr(model.embed, "words") else model.final_norm.device
    ids = torch.tensor([[t for w in windows for t in w.tokens]], dtype=torch.long, device=device)
    x = gather_rows(model.embed, ids, dtype=act_dtype)
    if x.device != device:
        x = x.to(device=device)
    x = x.contiguous()
    starts = [0]
    for w in windows:
        w.states = {}
        starts.append(starts[-1] + len(w.tokens))
    for index, layer in enumerate(model.layers):
        normed = rms_norm(x, layer.input_norm, spec.eps)
        if spec.full(index):
            y = _attention_rows(spec, layer, normed, windows, starts, index, linear)
        else:
            y = _linear_rows(spec, layer, normed, windows, starts, index, linear)
        if reduce is not None:
            y = reduce(y)
        x[:] = _residual(x, y)
        y = _mlp(spec, layer, rms_norm(x, layer.post_norm, spec.eps), linear)
        if reduce is not None:
            y = reduce(y)
        x[:] = _residual(x, y)
    return rms_norm(x, model.final_norm, spec.eps)


def commit(model, window: Window, rows: int) -> None:
    """Keep the window's first ``rows`` rows: attention lengths and linear states as after the serial steps."""

    for index, cache in enumerate(window.caches):
        if model.spec.full(index):
            cache["len"] = window.pos + rows
        else:
            cache["conv"], cache["state"] = window.states[index][rows - 1]
    window.states = {}


def _attention_rows(spec, layer, x: torch.Tensor, windows: list[Window], starts: list[int], index: int, linear):
    batch, length, _ = x.shape
    grouped = _project_group(x, (layer.q, layer.k, layer.v), linear)
    if grouped is None:
        qg = _project(x, layer.q, linear)
        keys, values = _project_pair(x, layer.k, layer.v, linear)
    else:
        qg, keys, values = grouped
    qg = qg.view(batch, length, spec.heads, spec.head_dim * 2)
    queries, gate = qg.split(spec.head_dim, dim=-1)
    keys = keys.view(batch, length, spec.kv_heads, spec.head_dim)
    values = values.view(batch, length, spec.kv_heads, spec.head_dim)
    queries = rms_norm(queries, layer.q_norm, spec.eps).permute(0, 2, 1, 3)
    keys = rms_norm(keys, layer.k_norm, spec.eps).permute(0, 2, 1, 3)
    values = values.permute(0, 2, 1, 3)
    attended = []
    scale = spec.head_dim ** -0.5
    for w, start in zip(windows, starts):
        cache = w.caches[index]
        for i in range(len(w.tokens)):
            # The serial step's calls, one row at a time: rope, the cache write, the decode walk over its keys.
            row, pos = start + i, w.pos + i
            q = apply_rope(queries[:, :, row:row + 1], pos, spec.rope_theta, spec.rotary_dim, exact=False)
            k = apply_rope(keys[:, :, row:row + 1], pos, spec.rope_theta, spec.rotary_dim, exact=False)
            if pos + 1 > cache["k"].shape[2]:
                raise RuntimeError("kv cache is shorter than the tokens written into it")
            cache["k"][:, :, pos:pos + 1] = k.to(dtype=cache["k"].dtype)
            cache["v"][:, :, pos:pos + 1] = values[:, :, row:row + 1].to(dtype=cache["v"].dtype)
            query = q if q.dtype == torch.float32 else q.float()
            attended.append(_attend(query, cache["k"][:, :, :pos + 1], cache["v"][:, :, :pos + 1], scale, pos))
        cache["len"] = w.pos + len(w.tokens)
    out = torch.cat(attended, dim=2).permute(0, 2, 1, 3).reshape(batch, length, -1)
    gated = out * torch.sigmoid(gate.reshape(batch, length, -1).float())
    if gated.dtype != x.dtype:
        gated = gated.to(dtype=x.dtype)
    return _project(gated, layer.o, linear)


def _linear_rows(spec, layer, x: torch.Tensor, windows: list[Window], starts: list[int], index: int, linear):
    batch, length, _ = x.shape
    grouped = _project_group(x, (layer.qkv, layer.z, layer.a, layer.b), linear)
    if grouped is None:
        qkv = _project(x, layer.qkv, linear)
        z = _project(x, layer.z, linear)
        a = _project(x, layer.a, linear)
        b = _project(x, layer.b, linear)
    else:
        qkv, z, a, b = grouped
    z = z.view(batch, length, spec.value_heads, spec.value_dim)
    ys = []
    for w, start in zip(windows, starts):
        cache = w.caches[index]
        conv, state = cache["conv"], cache["state"]
        kept = []
        for i in range(len(w.tokens)):
            # The serial step's conv and fused delta step; each row's states are kept for the commit.
            row = start + i
            mixed, conv = causal_conv(qkv[:, row:row + 1], layer.conv, conv, exact=False)
            q, k, v = mixed.split((spec.key_width, spec.key_width, spec.value_width), dim=-1)
            q = q.reshape(batch, 1, spec.key_heads, spec.key_dim)
            k = k.reshape(batch, 1, spec.key_heads, spec.key_dim)
            v = v.reshape(batch, 1, spec.value_heads, spec.value_dim)
            q, k = normalize_qk(q, k, spec.key_dim, spec.eps)
            y, state = gated_delta(q, k, v, a[:, row:row + 1], b[:, row:row + 1], layer.a_log, layer.dt_bias, state,
                                   fused=True)
            kept.append((conv.clone(), state.clone()))
            ys.append(y)
        cache["conv"], cache["state"] = conv, state
        w.states[index] = kept
    y = torch.cat(ys, dim=1)
    y = rms_norm(y, layer.gnorm, spec.eps) * torch.nn.functional.silu(z).float()
    if y.dtype != x.dtype:
        y = y.to(dtype=x.dtype)
    return _project(y.reshape(batch, length, -1), layer.out, linear)


__all__ = ["Window", "commit", "window_forward"]
