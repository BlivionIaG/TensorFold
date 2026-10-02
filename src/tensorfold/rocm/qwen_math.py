"""Qwen3.5 text math for the ROCm forward. The projection is passed in.

Device rows use the HIP norm, rope, conv, gated-delta and attention kernels. Tests pass a scalar
projection (``affine_reference``) to check the same math without the affine extension.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass

import torch

_FMAF = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6").fmaf
_FMAF.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_FMAF.restype = ctypes.c_float

# Tokens resident in one prefill step. A multiple of the 16-wide matmul tile.
SPAN = 2048


@dataclass
class Packed:
    """One affine matrix: packed int32 ``words`` and group ``scale``/``bias`` (fp32, bf16 or fp16).

    ``partial`` marks a tensor-parallel slice along K: its output is one rank's fp32 share of the sum.
    """

    words: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor
    bits: int
    group: int
    partial: bool = False


@dataclass
class Spec:
    hidden: int
    intermediate: int
    n_layers: int
    heads: int
    kv_heads: int
    head_dim: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv: int
    vocab: int
    eps: float
    rope_theta: float
    rotary_dim: int
    full_every: int
    bits: int
    group: int

    def full(self, index: int) -> bool:
        return (index + 1) % self.full_every == 0

    @property
    def key_width(self) -> int:
        return self.key_heads * self.key_dim

    @property
    def value_width(self) -> int:
        return self.value_heads * self.value_dim


def _rms_rows(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    return y if y.dtype == x.dtype else y.to(dtype=x.dtype)


def _rms_torch(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    width = x.shape[-1]
    flat = x.reshape(-1, width)
    if flat.shape[0] <= SPAN:
        return _rms_rows(flat, weight, eps).reshape(x.shape)
    out = torch.empty_like(flat)
    for start in range(0, flat.shape[0], SPAN):
        stop = min(start + SPAN, flat.shape[0])
        out[start:stop] = _rms_rows(flat[start:stop], weight, eps)
    return out.reshape(x.shape)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """fp32 RMSNorm. A short device row uses one HIP launch; a long prefill stays on PyTorch."""

    width = x.shape[-1]
    rows = x.numel() // width
    weight_ok = weight is None or weight.numel() == width
    if x.is_cuda and weight_ok and 1 <= rows <= 256 and 1 <= width <= 8192:
        from tensorfold.rocm.act import rms

        flat = x.reshape(rows, width).float().contiguous()
        scale = None if weight is None else weight.reshape(width).float().contiguous()
        y = rms(flat, scale, eps).reshape(x.shape)
        return y if y.dtype == x.dtype else y.to(dtype=x.dtype)
    return _rms_torch(x, weight, eps)


def normalize_qk(q: torch.Tensor, k: torch.Tensor, head_k: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize q and k and fold ``head_k ** -0.5`` into q, matching the MLX helper."""

    inv = head_k ** -0.5
    rms_eps = eps * inv * inv
    return (inv * inv) * rms_norm(q, None, rms_eps), inv * rms_norm(k, None, rms_eps)


def apply_rope(x: torch.Tensor, pos0: int, theta: float, rotary_dim: int, *, exact: bool = False) -> torch.Tensor:
    """Rotate the first ``rotary_dim`` features. Text positions use one index, so interleaved mrope matches this.

    ``exact`` keeps the prefill formula. The one-row HIP rope is for decode and is not bit-identical.
    """

    width = x.shape[-1]
    rows = x.numel() // width
    short = not exact and x.is_cuda and x.shape[-2] == 1 and rows <= 256 and width <= 8192
    if short and 0 < rotary_dim <= width and rotary_dim % 2 == 0:
        from tensorfold.rocm.act import rope_decode

        flat = x.reshape(rows, width).float().contiguous()
        y = rope_decode(flat, pos0, rotary_dim, theta).reshape(x.shape)
        return y if y.dtype == x.dtype else y.to(dtype=x.dtype)
    half = rotary_dim // 2
    freq = 1.0 / (theta ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    pos = torch.arange(pos0, pos0 + x.shape[2], device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]
    cos = ang.cos()[None, None]
    sin = ang.sin()[None, None]
    x1 = x[..., :half].float()
    x2 = x[..., half:rotary_dim].float()
    rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    if rotary_dim == x.shape[-1]:
        full = rot
    else:
        full = torch.cat((rot, x[..., rotary_dim:].float()), dim=-1)
    return full if full.dtype == x.dtype else full.to(dtype=x.dtype)


def causal_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int) -> torch.Tensor:
    """Causal attention. Query position ``q_pos0 + i`` reads keys ``0 .. q_pos0 + i``. Computed in query chunks."""

    heads = q.shape[1]
    kv_heads = k.shape[1]
    if heads != kv_heads:
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
    length = q.shape[2]
    span = k.shape[2]
    pieces = []
    step = 128
    key_pos = torch.arange(span, device=q.device)
    for start in range(0, length, step):
        stop = min(length, start + step)
        scores = torch.matmul(q[:, :, start:stop].float(), k.float().transpose(-1, -2)) * scale
        pos = torch.arange(start, stop, device=q.device) + q_pos0
        scores = scores.masked_fill(key_pos.view(1, 1, 1, span) > pos.view(1, 1, -1, 1), float("-inf"))
        pieces.append(torch.matmul(torch.softmax(scores, dim=-1), v.float()))
    return torch.cat(pieces, dim=2)


def causal_conv(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor | None, *,
                exact: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution. ``weight`` is (channels, kernel).

    ``exact`` keeps the prefill loop. The one-token HIP kernel uses ``expf`` and does not match it bit for bit.
    """

    batch, length, channels = x.shape
    kernel = weight.shape[1]
    if not exact and x.is_cuda and length == 1 and 1 <= kernel <= 8:
        from tensorfold.rocm.act import conv_decode

        if state is None:
            state = torch.zeros(batch, kernel - 1, channels, device=x.device, dtype=torch.float32)
        else:
            state = state.to(dtype=torch.float32).contiguous()
        sample = x.reshape(batch, 1, channels).float().contiguous()
        y = conv_decode(sample, weight.float().contiguous(), state)
        return y.view(batch, 1, channels), state
    if state is None:
        state = x.new_zeros(batch, kernel - 1, channels)
    window = torch.cat((state.float(), x.float()), dim=1)
    out = torch.zeros(batch, length, channels, device=x.device, dtype=torch.float32)
    taps = weight.float()
    for tap in range(kernel):
        out.add_(window[:, tap:tap + length] * taps[:, tap].view(1, 1, channels))
    return torch.nn.functional.silu(out), window[:, length:].contiguous()


def _gate_beta(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
               ) -> tuple[torch.Tensor, torch.Tensor]:
    beta = torch.sigmoid(b.float())
    gate = torch.exp(-torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float()))
    return gate, beta


def _lanes(x: torch.Tensor, span: int) -> torch.Tensor:
    """Pack the last axis as ``(span, 32)`` with index ``lane + i * 32``. Unused lanes stay zero."""

    width = span * 32
    if x.shape[-1] != width:
        padded = x.new_zeros(*x.shape[:-1], width)
        padded[..., :x.shape[-1]] = x
        x = padded
    return x.reshape(*x.shape[:-1], span, 32)


def _warp0(partial: torch.Tensor) -> torch.Tensor:
    """Lane 0 after a xor-shuffle of 16, 8, 4, 2, 1. Each lane adds ``x[i] + x[i ^ mask]``."""

    x = partial
    index = torch.arange(32, device=partial.device)
    for mask in (16, 8, 4, 2, 1):
        x = x + x[..., index ^ mask]
    return x[..., 0]


def gated_delta_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor,
                          beta: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's reduction, in PyTorch. Decay, then the value residual, then the q readout."""

    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if key_dim not in (16, 128) or value_heads % key_heads != 0:
        raise ValueError("dk is 16 or 128 and value heads are a multiple of key heads")
    span = 1 if key_dim < 32 else key_dim // 32
    repeat = value_heads // key_heads
    q = q.float().contiguous()
    k = k.float().contiguous()
    v = v.float().contiguous()
    gate = gate.float().reshape(batch, length, value_heads)
    beta = beta.float().reshape(batch, length, value_heads)
    state = state.float().contiguous().clone()
    y = torch.empty(batch, length, value_heads, value_dim, device=q.device, dtype=torch.float32)
    heads = torch.arange(value_heads, device=q.device) // repeat
    for t in range(length):
        scaled = state * gate[:, t].view(batch, value_heads, 1, 1)
        kt = k[:, t].index_select(1, heads)
        qt = q[:, t].index_select(1, heads)
        st_l = _lanes(scaled, span)
        k_l = _lanes(kt, span).unsqueeze(2)
        partial = torch.zeros(batch, value_heads, value_dim, 32, device=q.device, dtype=torch.float32)
        for i in range(span):
            partial = partial + st_l[..., i, :] * k_l[..., i, :]
        delta = (v[:, t] - _warp0(partial)) * beta[:, t].view(batch, value_heads, 1)
        for i in range(span):
            st_l[..., i, :] = st_l[..., i, :] + k_l[..., i, :] * delta.unsqueeze(-1)
        state = st_l.reshape(batch, value_heads, value_dim, span * 32)[..., :key_dim].contiguous()
        q_l = _lanes(qt, span).unsqueeze(2)
        acc = torch.zeros_like(partial)
        for i in range(span):
            acc = acc + st_l[..., i, :] * q_l[..., i, :]
        y[:, t] = _warp0(acc)
    return y, state


def gated_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                a_log: torch.Tensor, dt_bias: torch.Tensor, state: torch.Tensor | None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar-gate delta rule. State is fp32 and each request keeps its own.

    Decay, then the value residual, then the q readout. On device the wave kernel runs that
    order; ``gated_delta_reference`` is the same Dk reduction for the test.
    """

    gate, beta = _gate_beta(a, b, a_log, dt_bias)
    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if state is None:
        state = torch.zeros(batch, value_heads, value_dim, key_dim, device=q.device, dtype=torch.float32)
    else:
        state = state.float()
    qf = q.float().contiguous()
    kf = k.float().contiguous()
    vf = v.float().contiguous()
    gate = gate.reshape(batch, length, value_heads).contiguous()
    beta = beta.reshape(batch, length, value_heads).contiguous()
    if qf.is_cuda:
        from tensorfold.rocm.gated_delta import recurrence

        return recurrence(qf, kf, vf, gate, beta, state.contiguous())
    return gated_delta_reference(qf, kf, vf, gate, beta, state)


def _codes(words: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """Unpack codes the way the HIP reader does, including a code that crosses two words."""

    words64 = words.to(torch.int64) & 0xFFFFFFFF
    index = torch.arange(k, device=words.device)
    bit = index * bits
    word = bit // 32
    shift = bit % 32
    shape = (*words.shape[:-1], k)
    low = torch.gather(words64, -1, word.expand(shape))
    nxt = (word + 1).clamp(max=words.shape[-1] - 1)
    high = torch.gather(words64, -1, nxt.expand(shape))
    high = torch.where((shift + bits > 32) & (word + 1 < words.shape[-1]), high, torch.zeros_like(high))
    value = (low >> shift) | (high << ((32 - shift) & 31))
    return value & ((1 << bits) - 1)


def _dequant_rows(packed: Packed, ids: torch.Tensor) -> torch.Tensor:
    """Fp32 rows for a 1-D id list. The packed table is not expanded."""

    words = packed.words[ids]
    groups = packed.scale.shape[1]
    k = groups * packed.group
    if packed.bits == 8:
        # Eight-bit codes are little-endian bytes of the int32 words, one code per byte.
        codes = words.contiguous().view(torch.uint8)[..., :k].to(torch.float32)
    else:
        codes = _codes(words, packed.bits, k).to(torch.float32)
    codes = codes.to(torch.bfloat16).to(torch.float32)
    scale = packed.scale[ids].to(torch.float32).unsqueeze(-1)
    bias = packed.bias[ids].to(torch.float32).unsqueeze(-1)
    return (codes.view(ids.shape[0], groups, packed.group) * scale + bias).reshape(ids.shape[0], k)


def gather_rows(packed: Packed, ids: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dequantize selected embedding rows from packed codes. The table itself stays packed."""

    ids = ids.to(packed.words.device)
    k = packed.scale.shape[1] * packed.group
    flat = ids.reshape(-1)
    out = torch.empty(flat.shape[0], k, dtype=torch.float32 if dtype is None else dtype, device=ids.device)
    for start in range(0, flat.shape[0], SPAN):
        stop = min(start + SPAN, flat.shape[0])
        piece = _dequant_rows(packed, flat[start:stop])
        out[start:stop] = piece if piece.dtype == out.dtype else piece.to(dtype=out.dtype)
    return out.view(*ids.shape, k)


def affine_reference(x: torch.Tensor, packed: Packed) -> torch.Tensor:
    """Scalar affine formula with libm fmaf. Group order matches the GEMV kernel. Does not call the extension."""

    values = x.detach().float().to(torch.bfloat16).float().cpu()
    codes = _codes(packed.words.cpu(), packed.bits, values.shape[-1]).float().to(torch.bfloat16).float()
    scale = packed.scale.float().cpu()
    bias = packed.bias.float().cpu()
    rows, k = values.shape
    cols = codes.shape[0]
    acc = torch.zeros(rows, cols)
    group = packed.group
    for start in range(0, k, group):
        dot = torch.zeros(rows, cols)
        summed = torch.zeros(rows)
        block_x = values[:, start:start + group]
        block_q = codes[:, start:start + group]
        for t in range(group):
            xv = block_x[:, t].contiguous().numpy()
            qv = block_q[:, t].contiguous().numpy()
            # Broadcast the column of x across outputs and call fmaf in C order.
            prod = torch.empty(rows, cols)
            flat_x = torch.from_numpy(xv).unsqueeze(1).expand(rows, cols).reshape(-1).numpy()
            flat_q = torch.from_numpy(qv).unsqueeze(0).expand(rows, cols).reshape(-1).numpy()
            flat_d = dot.reshape(-1).numpy()
            out = prod.reshape(-1).numpy()
            fmaf = _FMAF
            for i in range(out.shape[0]):
                out[i] = fmaf(float(flat_x[i]), float(flat_q[i]), float(flat_d[i]))
            dot = torch.from_numpy(out.copy()).view(rows, cols)
            summed = summed + block_x[:, t]
        g = start // group
        acc = _fma_cols(dot, scale[:, g], acc)
        acc = _fma_rows(summed, bias[:, g], acc)
    return acc.to(torch.bfloat16)


def _fma_cols(dot: torch.Tensor, scale: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    rows, cols = dot.shape
    out = torch.empty(rows, cols)
    flat_a = dot.reshape(-1).numpy()
    flat_b = scale.expand(rows, cols).reshape(-1).numpy()
    flat_c = acc.reshape(-1).numpy()
    dest = out.reshape(-1).numpy()
    fmaf = _FMAF
    for i in range(dest.shape[0]):
        dest[i] = fmaf(float(flat_a[i]), float(flat_b[i]), float(flat_c[i]))
    return torch.from_numpy(dest.copy()).view(rows, cols)


def _fma_rows(summed: torch.Tensor, bias: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
    rows, cols = acc.shape
    out = torch.empty(rows, cols)
    flat_a = summed.unsqueeze(1).expand(rows, cols).reshape(-1).numpy()
    flat_b = bias.expand(rows, cols).reshape(-1).numpy()
    flat_c = acc.reshape(-1).numpy()
    dest = out.reshape(-1).numpy()
    fmaf = _FMAF
    for i in range(dest.shape[0]):
        dest[i] = fmaf(float(flat_a[i]), float(flat_b[i]), float(flat_c[i]))
    return torch.from_numpy(dest.copy()).view(rows, cols)


def _project(x: torch.Tensor, packed: Packed, linear) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    y = linear(flat, packed)
    if y.dtype != x.dtype and not packed.partial:     # a rank's share stays fp32 until the ranks are summed
        y = y.to(dtype=x.dtype)
    return y.reshape(*x.shape[:-1], -1)


def _project_pair(x: torch.Tensor, first: Packed, second: Packed, linear):
    """Two projections of one activation. Falls back to two solo launches."""

    owner = getattr(linear, "__self__", None)
    pair = getattr(owner, "linear_pair", None) if owner is not None else None
    if pair is None:
        return _project(x, first, linear), _project(x, second, linear)
    left, right = pair(x, first, second)
    if left.dtype != x.dtype:
        left = left.to(dtype=x.dtype)
    if right.dtype != x.dtype:
        right = right.to(dtype=x.dtype)
    return left.reshape(*x.shape[:-1], -1), right.reshape(*x.shape[:-1], -1)


def _project_group(x: torch.Tensor, packeds: tuple, linear):
    """One launch for several projections of a short activation. None keeps the solo or pair path."""

    owner = getattr(linear, "__self__", None)
    group = getattr(owner, "linear_group", None) if owner is not None else None
    if group is None:
        return None
    outs = group(x, packeds)
    if outs is None:
        return None
    return tuple(y.reshape(*x.shape[:-1], -1) if y.dtype == x.dtype else y.to(dtype=x.dtype).reshape(*x.shape[:-1], -1)
                 for y in outs)


def forward_hidden(model, tokens: torch.Tensor, caches: list | None, linear, pos0: int,
                   act_dtype: torch.dtype | None = None, *, exact_short: bool = False, reduce=None):
    """One prefill or decode step. ``tokens`` is (batch, length). Returns hidden states and new caches.

    ``reduce`` sums the ranks' shares of each attention and MLP output before the residual add (tensor
    parallel); None is one rank.
    """

    spec = model.spec
    x = gather_rows(model.embed, tokens, dtype=act_dtype)
    if x.device != tokens.device:
        x = x.to(device=tokens.device)
    if not x.is_contiguous():
        x = x.contiguous()
    fresh = caches is None
    new_caches = []
    length = x.shape[1]
    for index, layer in enumerate(model.layers):
        cache = None if fresh else caches[index]
        # One span of the residual. The whole prompt is not a second fp32 copy.
        for start in range(0, length, SPAN):
            stop = min(length, start + SPAN)
            normed = rms_norm(x[:, start:stop], layer.input_norm, spec.eps)
            if spec.full(index):
                y, cache = _attention(spec, layer, normed, cache, linear, pos0 + start, exact_short)
            else:
                y, cache = _linear_attn(spec, layer, normed, cache, linear, exact_short)
            if reduce is not None:
                y = reduce(y)
            x[:, start:stop] = x[:, start:stop] + y
            y = _mlp(spec, layer, rms_norm(x[:, start:stop], layer.post_norm, spec.eps), linear)
            if reduce is not None:
                y = reduce(y)
            x[:, start:stop] = x[:, start:stop] + y
        new_caches.append(cache)
    return rms_norm(x, model.final_norm, spec.eps), new_caches


def _mlp(spec: Spec, layer, x: torch.Tensor, linear) -> torch.Tensor:
    # Gate and up are the wide tensors. A long prefill computes them a chunk at a time.
    batch, length, hidden = x.shape
    flat = x.reshape(-1, hidden)
    if flat.shape[0] <= SPAN:
        grouped = _project_group(x, (layer.gate, layer.up), linear)
        if grouped is None:
            gate, up = _project_pair(x, layer.gate, layer.up, linear)
        else:
            gate, up = grouped
        return _project(torch.nn.functional.silu(gate) * up, layer.down, linear)
    out = torch.empty(flat.shape, dtype=torch.float32 if layer.down.partial else flat.dtype, device=flat.device)
    for start in range(0, flat.shape[0], SPAN):
        stop = min(start + SPAN, flat.shape[0])
        piece = flat[start:stop]
        gate = _project(piece, layer.gate, linear)
        up = _project(piece, layer.up, linear)
        out[start:stop] = _project(torch.nn.functional.silu(gate) * up, layer.down, linear)
    return out.view(batch, length, hidden)


def _linear_span(spec: Spec, layer, x: torch.Tensor, conv_state, rec, linear, exact: bool):
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
    mixed, conv_state = causal_conv(qkv, layer.conv, conv_state, exact=exact)
    q, k, v = mixed.split((spec.key_width, spec.key_width, spec.value_width), dim=-1)
    q = q.view(batch, length, spec.key_heads, spec.key_dim)
    k = k.view(batch, length, spec.key_heads, spec.key_dim)
    v = v.view(batch, length, spec.value_heads, spec.value_dim)
    q, k = normalize_qk(q, k, spec.key_dim, spec.eps)
    y, rec = gated_delta(q, k, v, a, b, layer.a_log, layer.dt_bias, rec)
    y = rms_norm(y, layer.gnorm, spec.eps) * torch.nn.functional.silu(z)
    if y.dtype != x.dtype:
        y = y.to(dtype=x.dtype)
    return _project(y.reshape(batch, length, -1), layer.out, linear), conv_state, rec


def _linear_attn(spec: Spec, layer, x: torch.Tensor, cache, linear, exact: bool):
    batch, length, hidden = x.shape
    conv_state = None if cache is None else cache["conv"]
    rec = None if cache is None else cache["state"]
    if length <= SPAN:
        y, conv_state, rec = _linear_span(spec, layer, x, conv_state, rec, linear, exact)
        return y, {"conv": conv_state, "state": rec}
    y = torch.empty(batch, length, hidden, dtype=torch.float32 if layer.out.partial else x.dtype, device=x.device)
    for start in range(0, length, SPAN):
        stop = min(length, start + SPAN)
        y[:, start:stop], conv_state, rec = _linear_span(
            spec, layer, x[:, start:stop], conv_state, rec, linear, exact)
    return y, {"conv": conv_state, "state": rec}


def _attention(spec: Spec, layer, x: torch.Tensor, cache, linear, pos0: int, exact: bool):
    batch, length, hidden = x.shape
    if length <= SPAN:
        return _attention_span(spec, layer, x, cache, linear, pos0, exact)
    y = torch.empty(batch, length, hidden, dtype=x.dtype, device=x.device)
    for start in range(0, length, SPAN):
        stop = min(length, start + SPAN)
        y[:, start:stop], cache = _attention_span(
            spec, layer, x[:, start:stop], cache, linear, pos0 + start, exact)
    return y, cache


def _attention_span(spec: Spec, layer, x: torch.Tensor, cache, linear, pos0: int, exact: bool):
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
    queries = apply_rope(queries, pos0, spec.rope_theta, spec.rotary_dim, exact=exact)
    keys = apply_rope(keys, pos0, spec.rope_theta, spec.rotary_dim, exact=exact)
    if cache is not None and "len" in cache:
        end = cache["len"] + length
        if end > cache["k"].shape[2]:
            raise RuntimeError("kv cache is shorter than the tokens written into it")
        cache["k"][:, :, cache["len"]:end] = keys.to(dtype=cache["k"].dtype)
        cache["v"][:, :, cache["len"]:end] = values.to(dtype=cache["v"].dtype)
        cache["len"] = end
        kept_k = cache["k"][:, :, :end]
        kept_v = cache["v"][:, :, :end]
        new_cache = cache
    elif cache is None:
        kept_k, kept_v = keys, values
        new_cache = {"k": kept_k, "v": kept_v}
    else:
        kept_k = torch.cat((cache["k"], keys), dim=2)
        kept_v = torch.cat((cache["v"], values), dim=2)
        new_cache = {"k": kept_k, "v": kept_v}
    query = queries if queries.dtype == torch.float32 else queries.float()
    attended = _attend(query, kept_k, kept_v, spec.head_dim ** -0.5, pos0)
    attended = attended.permute(0, 2, 1, 3).reshape(batch, length, -1)
    gated = attended * torch.sigmoid(gate.reshape(batch, length, -1).float())
    if gated.dtype != x.dtype:
        gated = gated.to(dtype=x.dtype)
    return _project(gated, layer.o, linear), new_cache


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, q_pos0: int) -> torch.Tensor:
    """Device attention reads the cache dtype directly. The Python path is the spec for the test."""

    # Even head sizes use the flash tile for prefill and the split key walk for one query.
    # Odd widths stay on the chunked matmul.
    even = q.shape[-1] <= 256 and q.shape[-1] % 2 == 0 and q.shape[1] % k.shape[1] == 0
    if q.is_cuda and k.is_cuda and even:
        from tensorfold.rocm.attention import causal

        return causal(q, k, v, scale, q_pos0)
    return causal_attend(q, k, v, scale, q_pos0)


def _blank_caches(model, batch: int, total: int, device: torch.device, cache_dtype: torch.dtype = torch.float32,
                  ) -> list:
    """One cache per request. Full attention keeps a fixed key/value buffer; linear attention keeps its own state."""

    spec = model.spec
    caches = []
    for index in range(spec.n_layers):
        if spec.full(index):
            shape = (batch, spec.kv_heads, total, spec.head_dim)
            caches.append({
                "k": torch.empty(shape, device=device, dtype=cache_dtype),
                "v": torch.empty(shape, device=device, dtype=cache_dtype),
                "len": 0,
            })
        else:
            caches.append({"conv": None, "state": None})
    return caches


def greedy(model, prompts: list[list[int]], n_new: int, linear, device: torch.device,
           after_token=None, cache_dtype: torch.dtype = torch.float32, *, reduce=None,
           gather=None) -> list[list[int]]:
    """Generate ``n_new`` tokens for every prompt. End-of-sequence does not stop the loop.

    ``after_token(step)`` runs once the work for that token has been queued. Step ``-1`` is the start,
    before the prompt forward. The requests share one batch and one cache each. ``reduce`` and ``gather``
    are the tensor-parallel hooks: the residual sum and the vocabulary slices joined in rank order.
    """

    if n_new < 1:
        raise ValueError("n_new must be positive")
    tokens = torch.tensor(prompts, dtype=torch.long, device=device)
    if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
        raise ValueError("prompts must be a non-empty rectangular batch")
    batch, length = tokens.shape
    caches = _blank_caches(model, batch, length + n_new, device, cache_dtype)
    out = [[] for _ in range(batch)]

    def commit(step: int, nxt: torch.Tensor) -> None:
        # The copy lands the ids on the host. The clock, if any, starts only after that sync.
        ids = [int(token) for token in nxt.tolist()]
        if after_token is not None:
            after_token(step)
        for row, token in enumerate(ids):
            out[row].append(token)

    if after_token is not None:
        after_token(-1)
    def pick(hidden: torch.Tensor) -> torch.Tensor:
        logits = _project(hidden[:, -1], model.output_head(), linear)
        return torch.argmax(logits if gather is None else gather(logits), dim=-1)

    hidden, caches = forward_hidden(model, tokens, caches, linear, 0, cache_dtype, reduce=reduce)
    nxt = pick(hidden)
    commit(0, nxt)
    for step in range(1, n_new):
        hidden, caches = forward_hidden(model, nxt.view(-1, 1), caches, linear, length + step - 1, cache_dtype,
                                        reduce=reduce)
        nxt = pick(hidden)
        commit(step, nxt)
    if any(len(row) != n_new for row in out):
        raise RuntimeError("generation stopped before the requested token count")
    return out
