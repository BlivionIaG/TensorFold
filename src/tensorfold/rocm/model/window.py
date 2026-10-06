"""A lane round's forward: every stream's window in one pass, each row with its serial decode step's bits."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tensorfold.rocm.kernels.act import conv_decode, conv_rows, rope_decode
from tensorfold.rocm.kernels.attention import causal_at
from tensorfold.rocm.model.forward import _mlp, _project, _project_group, _project_pair, _residual
from tensorfold.rocm.model.qwen_math import gated_delta, gather_rows, normalize_qk, rms_norm


@dataclass
class Window:
    """One stream's rows this round: its last token and drafts from slot ``pos``, over the stream's caches."""

    tokens: list[int]
    caches: list[dict]
    pos: int
    states: dict = field(default_factory=dict)   # linear layer index -> (conv states, delta states) after each row
    slots: torch.Tensor | None = None            # the rows' positions on the device, int64 (a graph sets them)

    def positions(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.slots is None:
            self.slots = torch.arange(self.pos, self.pos + len(self.tokens), dtype=torch.int64, device=device)
        return self.slots, self.slots.to(torch.int32)


def window_forward(model, windows: list[Window], linear, act_dtype: torch.dtype, *, reduce=None,
                   ids: torch.Tensor | None = None) -> torch.Tensor:
    """The final-normed hidden rows of every window, concatenated in order; ``reduce`` sums tp shares."""

    spec = model.spec
    device = model.embed.words.device if hasattr(model.embed, "words") else model.final_norm.device
    if ids is None:
        ids = torch.tensor([[t for w in windows for t in w.tokens]], dtype=torch.long, device=device)
    x = gather_rows(model.embed, ids, dtype=act_dtype)
    if x.device != device:
        x = x.to(device=device)
    x = x.contiguous()
    starts = [0]
    for w in windows:
        w.states = {}
        starts.append(starts[-1] + len(w.tokens))
    slots = [w.positions(device) for w in windows]
    for index, layer in enumerate(model.layers):
        normed = rms_norm(x, layer.input_norm, spec.eps)
        if spec.full(index):
            y = _attention_rows(spec, layer, normed, windows, starts, slots, index, linear)
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

    every = rows == len(window.tokens)              # the linear states already advanced in place to the last row
    for index, cache in enumerate(window.caches):
        if model.spec.full(index):
            cache["len"] = window.pos + rows
        elif not every:
            # Copied into the stream's own buffers, which a captured window reads and writes in place.
            convs, deltas = window.states[index]
            cache["conv"][0].copy_(convs[rows - 1])
            cache["state"][0].copy_(deltas[0, rows - 1])
    window.states = {}


def _attention_rows(spec, layer, x: torch.Tensor, windows: list[Window], starts: list[int], slots, index: int,
                    linear):
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
    queries = rms_norm(queries, layer.q_norm, spec.eps)
    keys = rms_norm(keys, layer.k_norm, spec.eps)
    attended = []
    for w, start, (at64, at32) in zip(windows, starts, slots):
        # The serial step's calls with a position a row: rope, the cache write, the decode walk.
        cache, rows = w.caches[index], len(w.tokens)
        q = _rope_rows(queries[0, start:start + rows], at32, spec)
        k = _rope_rows(keys[0, start:start + rows], at32, spec)
        cache["k"].index_copy_(2, at64, k.permute(1, 0, 2).unsqueeze(0).to(dtype=cache["k"].dtype))
        cache["v"].index_copy_(2, at64, values[0, start:start + rows].permute(1, 0, 2).unsqueeze(0)
                               .to(dtype=cache["v"].dtype))
        query = q.float().unsqueeze(2).contiguous()                  # (rows, heads, 1, d): a query a row
        attended.append(causal_at(query, cache["k"], cache["v"], spec.head_dim ** -0.5, at32).reshape(rows, -1))
    out = (attended[0] if len(attended) == 1 else torch.cat(attended, dim=0)).unsqueeze(0)
    gated = out * torch.sigmoid(gate.reshape(batch, length, -1).float())
    if gated.dtype != x.dtype:
        gated = gated.to(dtype=x.dtype)
    return _project(gated, layer.o, linear)


def _rope_rows(x: torch.Tensor, at32: torch.Tensor, spec) -> torch.Tensor:
    """(rows, heads, d) rotated row by row at the device positions, in the one-row decode's arithmetic."""

    rows, heads, width = x.shape
    flat = x.reshape(rows * heads, width).float().contiguous()
    y = rope_decode(flat, at32, spec.rotary_dim, spec.rope_theta, per=heads).reshape(rows, heads, width)
    return y if y.dtype == x.dtype else y.to(dtype=x.dtype)


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
    weight = layer.conv.float().contiguous()
    ys = []
    for w, start in zip(windows, starts):
        # The serial step's conv and fused delta step over the window's rows, each row's states kept for the commit.
        cache, rows = w.caches[index], len(w.tokens)
        conv, state = cache["conv"], cache["state"]
        # A row's states are kept only for a commit short of the window's last row.
        convs = torch.empty((rows, *conv.shape[1:]), dtype=torch.float32, device=x.device) if rows > 1 else None
        x_rows = qkv[0, start:start + rows].float().contiguous()
        if rows == 1:                                # the serial step's own conv launch
            mixed = conv_decode(x_rows.view(1, 1, -1), weight, conv).view(1, 1, -1)
        else:
            mixed = conv_rows(x_rows, weight, conv, convs).unsqueeze(0)
        q, k, v = mixed.split((spec.key_width, spec.key_width, spec.value_width), dim=-1)
        q = q.reshape(batch, rows, spec.key_heads, spec.key_dim)
        k = k.reshape(batch, rows, spec.key_heads, spec.key_dim)
        v = v.reshape(batch, rows, spec.value_heads, spec.value_dim)
        q, k = normalize_qk(q, k, spec.key_dim, spec.eps)
        deltas = torch.empty((1, rows, *state.shape[1:]), dtype=torch.float32, device=x.device) if rows > 1 else None
        y, _ = gated_delta(q, k, v, a[:, start:start + rows], b[:, start:start + rows], layer.a_log, layer.dt_bias,
                           state, fused=True, states=deltas)
        w.states[index] = (convs, deltas)
        ys.append(y)
    y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=1)
    y = rms_norm(y, layer.gnorm, spec.eps) * torch.nn.functional.silu(z).float()
    if y.dtype != x.dtype:
        y = y.to(dtype=x.dtype)
    return _project(y.reshape(batch, length, -1), layer.out, linear)


__all__ = ["Window", "commit", "window_forward"]
