"""Qwen3.5 text forward on RDNA. Projections are packed affine matmuls.

Serving cells are measured by ``python -m tensorfold.rocm.bench MODEL_DIR``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorfold.rocm import qwen_math
from tensorfold.rocm.qwen_math import GptqPacked, Packed, Spec, greedy

_RDNA2 = {f"gfx103{i}" for i in range(7)}
# AWQ stores literal zeros, GPTQ stores them +1; set by load() and read by _packed_gptq.
_V2 = False


@dataclass
class LinearLayer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: Packed
    z: Packed
    a: Packed
    b: Packed
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    gnorm: torch.Tensor
    out: Packed
    gate: Packed
    up: Packed
    down: Packed
    moe: "Routed | None" = None


def _even(size: int, world: int, name: str) -> int:
    if size % world:
        raise ValueError(f"{name}: {size} does not split into {world} equal parts")
    return size // world


def _rows(packed: Packed, spans: list[tuple[int, int]]) -> Packed:
    """Output rows ``spans`` of ``packed``, in order: one rank's share of a column-split projection."""

    if isinstance(packed, GptqPacked):
        raise ValueError("the RDNA W4A16 path does not slice across ranks yet; run one rank")
    pick = lambda t: torch.cat([t[a:b] for a, b in spans]).contiguous()  # noqa: E731
    return Packed(pick(packed.words), pick(packed.scale), pick(packed.bias), packed.bits, packed.group)


def _row_spans(widths: list[int], rank: int, world: int, name: str) -> list[tuple[int, int]]:
    """This rank's rows of each segment of a concatenated output (q | k | v): each segment split evenly."""

    spans, base = [], 0
    for width in widths:
        part = _even(width, world, name)
        spans.append((base + rank * part, base + (rank + 1) * part))
        base += width
    return spans


def _cols(packed: Packed, rank: int, world: int, name: str) -> Packed:
    """Input columns of one rank, whole groups: a row-split projection whose fp32 outputs the ranks sum."""

    if isinstance(packed, GptqPacked):
        raise ValueError("the RDNA W4A16 path does not slice across ranks yet; run one rank")
    groups = _even(packed.scale.shape[1], world, f"{name} groups")
    words = groups * packed.group * packed.bits // 32
    return Packed(packed.words[:, rank * words:(rank + 1) * words].contiguous(),
                  packed.scale[:, rank * groups:(rank + 1) * groups].contiguous(),
                  packed.bias[:, rank * groups:(rank + 1) * groups].contiguous(), packed.bits, packed.group,
                  partial=True)


def _kv_spans(width: int, kv_heads: int, rank: int, world: int, name: str) -> list[tuple[int, int]]:
    """This rank's rows of k or v. With fewer KV heads than ranks, each head is replicated on ``world // kv_heads``
    ranks in a row, the ones whose query heads it serves."""

    if kv_heads >= world:
        return _row_spans([width], rank, world, name)
    if world % kv_heads:
        raise ValueError(f"{name}: {kv_heads} KV heads do not divide {world} ranks")
    head = rank // (world // kv_heads)
    part = width // kv_heads
    return [(head * part, (head + 1) * part)]


def _heads(tensor: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([tensor[a:b] for a, b in spans]).contiguous()


def slice_for_tp(model: TextModel, rank: int, world: int) -> TextModel:
    """Keep this rank's share of every projection. Mutates ``model`` and ``model.spec`` to per-rank heads.

    Projections that read the replicated hidden state are split by output heads (full-attention q/k/v,
    the linear-attention qkv/z/a/b, MLP gate/up), keeping each segment of a concatenated output whole per
    head. Projections that write the residual (o, out, down) are split by input groups and return fp32
    shares that ``forward_hidden``'s ``reduce`` sums. The output head is split by vocabulary rows; a tied
    checkpoint keeps the full embedding for the lookup and a vocabulary slice of it as the head.
    """

    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} not in [0, {world})")
    spec = model.spec
    for name in ("heads", "key_heads", "value_heads"):
        _even(getattr(spec, name), world, name)
    _even(spec.vocab, world, "vocab")
    key_width, value_width = spec.key_width, spec.value_width
    for index, layer in enumerate(model.layers):
        base = f"layer {index}"
        if isinstance(layer, FullLayer):
            layer.q = _rows(layer.q, _row_spans([spec.heads * spec.head_dim * 2], rank, world, f"{base} q_proj"))
            kv = spec.kv_heads * spec.head_dim
            layer.k = _rows(layer.k, _kv_spans(kv, spec.kv_heads, rank, world, f"{base} k_proj"))
            layer.v = _rows(layer.v, _kv_spans(kv, spec.kv_heads, rank, world, f"{base} v_proj"))
            layer.o = _cols(layer.o, rank, world, f"{base} o_proj")
        else:
            qkv = _row_spans([key_width, key_width, value_width], rank, world, f"{base} in_proj_qkv")
            layer.qkv = _rows(layer.qkv, qkv)
            layer.conv = _heads(layer.conv, qkv)
            layer.z = _rows(layer.z, _row_spans([value_width], rank, world, f"{base} in_proj_z"))
            heads = _row_spans([spec.value_heads], rank, world, f"{base} value heads")
            layer.a = _rows(layer.a, heads)
            layer.b = _rows(layer.b, heads)
            layer.a_log = _heads(layer.a_log, heads)
            layer.dt_bias = _heads(layer.dt_bias, heads)
            layer.out = _cols(layer.out, rank, world, f"{base} out_proj")
        if getattr(layer, "moe", None) is not None:
            layer.moe = _experts_share(layer.moe, rank, world, f"{base} experts")
            continue
        mlp = _row_spans([layer.gate.words.shape[0]], rank, world, f"{base} mlp")
        layer.gate = _rows(layer.gate, mlp)
        layer.up = _rows(layer.up, mlp)
        layer.down = _cols(layer.down, rank, world, f"{base} mlp.down_proj")
    model.head = _rows(model.output_head(), _row_spans([spec.vocab], rank, world, "vocab"))
    if model.mtp is not None:
        _slice_mtp(model.mtp, spec, rank, world)
    spec.heads //= world
    spec.kv_heads = max(1, spec.kv_heads // world)
    spec.key_heads //= world
    spec.value_heads //= world
    return model


def _experts_share(routed: "Routed", rank: int, world: int, name: str) -> "Routed":
    """A rank's routed experts: a contiguous ``E / world`` of them, and the shared expert on rank 0.

    The router stays whole, so every rank picks the same pairs; :func:`tensorfold.rocm.moe.run` runs the
    rank's own and returns their fp32 sum, which the layer's all-reduce adds to the other ranks'.
    """

    from tensorfold.rocm.experts import AffineExperts, GptqExperts
    from tensorfold.rocm.moe import Routed

    total = routed.count                               # routed experts; the shared one is id ``total``
    part = _even(total, world, name)
    ids = list(range(rank * part, (rank + 1) * part)) + ([total] if rank == 0 else [])
    device = routed.router.device
    keep = torch.tensor(ids, dtype=torch.long, device=device)
    remap = torch.full((total + 1,), -1, dtype=torch.int32, device=device)
    remap[keep] = torch.arange(len(ids), dtype=torch.int32, device=device)
    ex = routed.experts
    if isinstance(ex, AffineExperts):
        take = lambda triple: None if triple is None else tuple(t.index_select(0, keep).contiguous()  # noqa: E731
                                                                 for t in triple)
        experts = AffineExperts(take(ex.up), take(ex.down), take(ex.gate), ex.bits, ex.group, ex.limit)
    elif isinstance(ex, GptqExperts):
        take = lambda t: t.index_select(1, keep).contiguous()  # noqa: E731
        experts = GptqExperts(take(ex.up), take(ex.up_z), take(ex.up_s), take(ex.down), take(ex.down_z),
                              take(ex.down_s), ex.group, ex.v2, ex.limit)
    else:
        raise ValueError(f"{name}: expert kind {type(ex).__name__} has no tp split")
    return Routed(routed.router, experts, routed.top_k, remap=remap, partial=True)


def _slice_mtp(head: "MTPHead", spec: Spec, rank: int, world: int) -> None:
    """The MTP head under tp: only its logits projection takes the rank's rows; the rest stays replicated.

    The head is one layer reading replicated inputs (the embedding and the main model's normed state), so
    running it whole on every rank keeps the drafts identical, which the decode loop needs: a rank advances
    its cache by how many drafts matched. Slicing it like a decoder layer would instead put a collective on
    every draft step. A routed MLP stays whole too: every rank holds all of the head's experts.
    """

    if head.head is not None:
        head.head = _rows(head.head, _row_spans([spec.vocab], rank, world, "mtp head"))


@dataclass
class FullLayer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    q: Packed
    k: Packed
    v: Packed
    o: Packed
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    gate: Packed
    up: Packed
    down: Packed
    moe: "Routed | None" = None


@dataclass
class TextModel:
    spec: Spec
    embed: Packed
    layers: list
    final_norm: torch.Tensor
    head: Packed | None = None
    mtp: "MTPHead | None" = None

    def output_head(self) -> Packed:
        return self.embed if self.head is None else self.head


@dataclass
class MTPHead:
    """An MTP head in either shape a checkpoint ships.

    Both read the main model's residual at a position and the next token's embedding, project them and run one
    attention layer. The Flash Next shape stops there: two norms, two fc halves, the attention, a final norm.
    The Qwen3 shape adds an input norm, a post norm and one MLP, dense or routed, and its attention is gated,
    so ``q`` carries a second ``head_dim`` per head. ``head`` is the dedicated ``mtp.head_proj`` projection
    when the checkpoint ships one; None ties with the main model's ``output_head()``.
    """

    fc_e_norm: torch.Tensor
    fc_h_norm: torch.Tensor
    fc_e: Packed
    fc_h: Packed
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    q: Packed
    k: Packed
    v: Packed
    o: Packed
    final_norm: torch.Tensor
    head: Packed | None = None             # None ties with main ``model.output_head()``
    input_norm: torch.Tensor | None = None
    post_norm: torch.Tensor | None = None
    gate: Packed | None = None
    up: Packed | None = None
    down: Packed | None = None
    moe: "Routed | None" = None
    gated: bool = False


def activation_dtype(gfx: str) -> torch.dtype:
    """FP16 on RDNA2, where the affine schedule is the FP16 dot. BF16 on an RDNA3 WMMA part."""

    from tensorfold.rocm.build import WMMA

    if gfx in _RDNA2:
        return torch.float16
    if gfx in WMMA:
        return torch.bfloat16
    raise RuntimeError(f"no activation dtype for {gfx}")


class Engine:
    """The forward's projections. ``schedule`` selects the affine kernel; ``dtype`` is the activation type.

    ``rccl`` is the tensor-parallel ring for a model cut by :func:`slice_for_tp`; None is one GPU.
    """

    def __init__(self, model: TextModel, schedule: str = "auto", dtype: torch.dtype | None = None, rccl=None):
        self.model = model
        self.schedule = schedule
        self.dtype = dtype
        self.rccl = rccl
        self.projections = 0

    def linear(self, flat: torch.Tensor, packed: Packed) -> torch.Tensor:
        from tensorfold.rocm import affine as affine_mod

        if self.dtype is None:
            from tensorfold.rocm.build import gfx_name

            self.dtype = activation_dtype(gfx_name())
        flat = flat.reshape(-1, flat.shape[-1]).contiguous()
        if isinstance(packed, GptqPacked):
            return self._gptq(flat, packed)
        flat = flat.to(dtype=self.dtype)
        self._note(flat, packed)
        words, scale, bias = packed.words, packed.scale, packed.bias
        schedule = self.schedule
        # A short batch on the WMMA part streams each column once. Wide rows stay on WMMA.
        if schedule == "auto" and flat.shape[0] == 1 and packed.bits == 8 and self.dtype == torch.bfloat16:
            schedule = "decode"
        # A tensor-parallel share along K stays fp32 until the ranks are summed.
        kwargs = {"bits": packed.bits, "group": packed.group, "schedule": schedule, "f32": packed.partial}
        span = qwen_math.SPAN
        if flat.shape[0] <= span:
            return affine_mod.matmul(flat, words, scale, bias, **kwargs)
        # One output buffer. Keeping every chunk and then concatenating doubles a long prefill.
        out = torch.empty(flat.shape[0], words.shape[0], dtype=torch.float32 if packed.partial else flat.dtype,
                          device=flat.device)
        for start in range(0, flat.shape[0], span):
            stop = min(start + span, flat.shape[0])
            out[start:stop] = affine_mod.matmul(flat[start:stop], words, scale, bias, **kwargs)
        return out

    def linear_pair(self, x: torch.Tensor, first: Packed, second: Packed):
        """Gate and up, or k and v: one shared activation load when both widths match."""

        if isinstance(first, GptqPacked) or isinstance(second, GptqPacked):
            return self.linear(x, first), self.linear(x, second)
        from tensorfold.rocm import affine as affine_mod
        from tensorfold.rocm.build import WMMA, gfx_name

        if self.dtype is None:
            self.dtype = activation_dtype(gfx_name())
        flat = x.reshape(-1, x.shape[-1]).to(dtype=self.dtype).contiguous()
        same = (first.words.shape[0] == second.words.shape[0] and first.bits == second.bits == 8
                and first.group == second.group and self.schedule != "gemv" and self.dtype == torch.bfloat16
                and gfx_name() in WMMA)
        if not same:
            return self.linear(flat, first), self.linear(flat, second)
        self._note(flat, first)
        self._note(flat, second)
        left, right = affine_mod.matmul_pair(
            flat, first.words, first.scale, first.bias, second.words, second.scale, second.bias,
            bits=first.bits, group=first.group)
        return left, right

    def linear_group(self, x: torch.Tensor, packeds: tuple):
        """Several projections of one short activation. None means the caller uses solo or pair launches."""

        if any(isinstance(p, GptqPacked) for p in packeds):
            return None
        from tensorfold.rocm import affine as affine_mod
        from tensorfold.rocm.build import WMMA, gfx_name

        if self.dtype is None:
            self.dtype = activation_dtype(gfx_name())
        flat = x.reshape(-1, x.shape[-1]).to(dtype=self.dtype).contiguous()
        same = (2 <= len(packeds) <= 4 and flat.shape[0] <= 16 and self.schedule != "gemv"
                and self.dtype == torch.bfloat16 and gfx_name() in WMMA
                and all(p.bits == 8 and p.group == packeds[0].group for p in packeds))
        if not same:
            return None
        for packed in packeds:
            self._note(flat, packed)
        return affine_mod.matmul_group(
            flat, tuple((p.words, p.scale, p.bias) for p in packeds), bits=8, group=packeds[0].group)

    def _note(self, flat: torch.Tensor, packed: Packed) -> None:
        words = packed.words
        k = flat.shape[-1]
        expect = k * packed.bits // 32
        if words.dtype != torch.int32 or words.ndim != 2 or words.shape[1] != expect:
            raise RuntimeError("a projection handed the matmul a weight that is not packed int32 words")
        self.projections += 1

    def _gptq(self, flat: torch.Tensor, packed: GptqPacked) -> torch.Tensor:
        """A W4A16 GPTQ projection: the RDNA2 fp16 dot, tiled by the prefill row counts."""

        from tensorfold.rocm import qgemm

        if packed.g_idx is not None:
            raise ValueError("the RDNA W4A16 path does not carry an act-order permutation yet")
        x = flat.to(dtype=torch.float16).contiguous()
        qweight, qzeros, scales = packed.qweight, packed.qzeros, packed.scales
        span = qwen_math.SPAN
        if x.shape[0] <= span:
            return qgemm.matmul(x, qweight, qzeros, scales, use_v2_format=packed.v2, prefill=x.shape[0] > 16)
        out = torch.empty(x.shape[0], qweight.shape[1], dtype=torch.float16, device=x.device)
        for start in range(0, x.shape[0], span):
            stop = min(start + span, x.shape[0])
            out[start:stop] = qgemm.matmul(x[start:stop], qweight, qzeros, scales, use_v2_format=packed.v2,
                                           prefill=True)
        return out

    def generate(self, prompts: list[list[int]], n_new: int, after_token=None) -> list[list[int]]:
        device = self.model.embed.words.device
        if self.dtype is None:
            from tensorfold.rocm.build import gfx_name

            self.dtype = activation_dtype(gfx_name())
        hooks = {}
        if self.rccl is not None and self.rccl.world > 1:
            from functools import partial

            from tensorfold.rocm.qwen_tp import all_reduce_local, vocab_gather

            hooks = {"reduce": partial(all_reduce_local, self.rccl), "gather": partial(vocab_gather, self.rccl)}
        with torch.inference_mode():
            return greedy(self.model, prompts, n_new, self.linear, device, after_token=after_token,
                          cache_dtype=self.dtype, **hooks)


class _Shards:
    """One logical tensor table over every safetensors file in a checkpoint."""

    def __init__(self, paths: list[Path], strip: str = ""):
        """``strip`` is a leading prefix dropped from the keys that carry it (``language_model.``)."""

        from safetensors import safe_open

        self._open = [safe_open(str(path), framework="pt") for path in paths]
        self._where: dict[str, tuple[int, str]] = {}
        for index, table in enumerate(self._open):
            for key in table.keys():
                short = key[len(strip):] if strip and key.startswith(strip) else key
                self._where[short] = (index, key)

    def get_tensor(self, key: str) -> torch.Tensor:
        try:
            index, stored = self._where[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        return self._open[index].get_tensor(stored)

    def __contains__(self, key: str) -> bool:
        return key in self._where


def _float(table, key: str, device: torch.device) -> torch.Tensor:
    return table.get_tensor(key).to(device=device, dtype=torch.float32).contiguous()


def _packed(table, key: str, bits: int, group: int, device: torch.device) -> Packed:
    try:
        words = table.get_tensor(key + ".weight")
    except KeyError:
        return _packed_gptq(table, key, device)
    if words.dtype == torch.uint32:
        words = words.view(torch.int32)
    if words.dtype != torch.int32 or words.ndim != 2:
        raise ValueError(f"{key} weight is {words.dtype} {tuple(words.shape)}, not packed int32 words")
    scale = table.get_tensor(key + ".scales")
    bias = table.get_tensor(key + ".biases")
    if scale.shape != bias.shape or scale.ndim != 2:
        raise ValueError(f"{key} scale and bias must share shape (N, K / group)")
    k = scale.shape[1] * group
    if k % group != 0 or words.shape[1] != k * bits // 32 or words.shape[0] != scale.shape[0]:
        raise ValueError(f"{key} packed shape {tuple(words.shape)} does not match K={k} bits={bits} group={group}")
    # The kernels read fp32, bf16 or fp16 group tables as stored. Anything else is widened to fp32 once.
    if scale.dtype != bias.dtype or scale.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        scale, bias = scale.to(torch.float32), bias.to(torch.float32)
    return Packed(words.to(device).contiguous(), scale.to(device).contiguous(), bias.to(device).contiguous(), bits,
                  group)


def _packed_gptq(table, key: str, device: torch.device) -> GptqPacked:
    """A GPTQ / AWQ projection: ``.qweight`` / ``.qzeros`` / ``.scales`` in place of ``.weight``."""

    qweight = table.get_tensor(key + ".qweight")
    qzeros = table.get_tensor(key + ".qzeros")
    scales = table.get_tensor(key + ".scales")
    if qweight.dtype == torch.uint32:
        qweight = qweight.view(torch.int32)
    if qzeros.dtype == torch.uint32:
        qzeros = qzeros.view(torch.int32)
    if qweight.dtype != torch.int32 or qweight.ndim != 2 or qzeros.dtype != torch.int32 or qzeros.ndim != 2:
        raise ValueError(f"{key} qweight and qzeros must be packed int32")
    n = qweight.shape[1]
    if n % 8 or qzeros.shape[1] != n // 8 or scales.shape != (qzeros.shape[0], n) or scales.dtype != torch.float16:
        raise ValueError(f"{key} GPTQ shapes do not fit: {tuple(qweight.shape)} {tuple(qzeros.shape)} "
                         f"{tuple(scales.shape)} {scales.dtype}")
    try:
        g_idx = table.get_tensor(key + ".g_idx").to(device=device, dtype=torch.int32).contiguous()
    except KeyError:
        g_idx = None
    return GptqPacked(qweight.to(device).contiguous(), qzeros.to(device).contiguous(),
                      scales.to(device).contiguous(), g_idx, _V2)


def _conv(table, key: str, device: torch.device) -> torch.Tensor:
    weight = table.get_tensor(key)
    if weight.ndim == 3 and weight.shape[-1] == 1:
        weight = weight.squeeze(-1)
    elif weight.ndim == 3 and weight.shape[1] == 1:
        raise ValueError(f"{key} is still in the unsanitized (channels, 1, kernel) layout")
    if weight.ndim != 2:
        raise ValueError(f"{key} must be (channels, kernel), got {tuple(weight.shape)}")
    return weight.to(device=device, dtype=torch.float32).contiguous()


def _affine_quant(quant: dict) -> tuple[int, int]:
    """MLX affine widths. 2, 4 and 8 sit inside a word; 3, 5 and 6 may cross into the next one."""

    bits, group = quant.get("bits"), quant.get("group_size")
    if quant.get("mode") != "affine" or bits not in (2, 3, 4, 5, 6, 8) or group not in (32, 64, 128):
        raise ValueError(f"the RDNA text path loads affine 2/3/4/5/6/8-bit groups 32/64/128, got {quant}")
    return int(bits), int(group)


def _dequant(words: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, group: int) -> torch.Tensor:
    """MLX affine words [..., K * bits / 32] with scale and bias [..., K / group] -> fp32 [..., K] (s * q + b)."""

    k = scale.shape[-1] * group
    bits = 32 * words.shape[-1] // k
    if bits not in (2, 4, 8):
        raise ValueError(f"a {bits}-bit group table is not unpacked here")
    if words.dtype == torch.uint32:
        words = words.view(torch.int32)
    per = 32 // bits
    packed = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(per, device=words.device, dtype=torch.int64) * bits
    codes = ((packed[..., None] >> shifts) & ((1 << bits) - 1)).reshape(*words.shape[:-1], k).to(torch.float32)
    return codes * scale.float().repeat_interleave(group, -1) + bias.float().repeat_interleave(group, -1)


def _router_rows(table, key: str, group: int, device: torch.device) -> torch.Tensor:
    """One router's rows [E, D] fp32: a float tensor as stored, or an MLX affine group table unpacked."""

    weight = table.get_tensor(key + ".weight").to(device)
    if weight.dtype.is_floating_point:
        return weight.float()
    return _dequant(weight, table.get_tensor(key + ".scales").to(device),
                    table.get_tensor(key + ".biases").to(device), group)


def _shared_last(mine: torch.Tensor, one: torch.Tensor, routed: str, shared: str, suffix: str) -> torch.Tensor:
    """The routed stack with the shared expert appended as its last entry; both corners are checked here."""

    if mine.dtype == torch.uint32:
        mine = mine.view(torch.int32)
    if one.dtype == torch.uint32:
        one = one.view(torch.int32)
    if one.ndim == mine.ndim and one.shape[0] == 1:
        one = one[0]
    if mine.ndim != one.ndim + 1 or mine.shape[1:] != one.shape:
        raise ValueError(f"{routed}{suffix} {tuple(mine.shape)} does not stack with {shared}{suffix} "
                         f"{tuple(one.shape)}")
    return torch.cat([mine, one[None]]).contiguous()


def _affine_side(table, routed: str, shared: str, bits: int, group: int, device: torch.device) -> tuple:
    """One projection's ``(E + 1, N, ...)`` affine words, scales and biases, the shared expert last."""

    words = _shared_last(table.get_tensor(routed + ".weight"), table.get_tensor(shared + ".weight"),
                         routed, shared, ".weight").to(device)
    scale = _shared_last(table.get_tensor(routed + ".scales"), table.get_tensor(shared + ".scales"),
                         routed, shared, ".scales").to(device)
    bias = _shared_last(table.get_tensor(routed + ".biases"), table.get_tensor(shared + ".biases"),
                        routed, shared, ".biases").to(device)
    if words.dtype != torch.int32 or scale.shape != bias.shape or scale.ndim != 3:
        raise ValueError(f"{routed} is not an MLX affine expert stack")
    k = scale.shape[2] * group
    if words.shape[2] != k * bits // 32 or words.shape[:2] != scale.shape[:2]:
        raise ValueError(f"{routed} packed stack {tuple(words.shape)} does not fit K={k} bits={bits} group={group}")
    if scale.dtype != bias.dtype or scale.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        scale, bias = scale.to(torch.float32), bias.to(torch.float32)
    return words, scale, bias


def _gptq_side(table, routed: str, shared: str, device: torch.device) -> tuple:
    """One projection's ``(E + 1, ...)`` qweight, qzeros and scales, the shared expert appended last."""

    side = tuple(_shared_last(table.get_tensor(routed + suffix), table.get_tensor(shared + suffix),
                              routed, shared, suffix).to(device) for suffix in (".qweight", ".qzeros", ".scales"))
    qweight, qzeros, scales = side
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32 or scales.ndim != qweight.ndim:
        raise ValueError(f"{routed} is not a GPTQ / AWQ expert stack")
    return side


def _experts(table, prefix: str, spec: Spec, bits: int, group: int, device: torch.device):
    """A layer's ``E + 1`` experts (the shared expert last) in the checkpoint's kind.

    Affine stacks keep their packed words for the affine kernel. A GPTQ / AWQ checkpoint's gate and up
    become the W4A16 kernel's ``mats`` dim, gate first; a checkpoint without a gate is a relu^2 layer.
    """

    from tensorfold.rocm.experts import AffineExperts, GptqExperts

    mine, shared = prefix + "switch_mlp.", prefix + "shared_expert."
    if f"{mine}up_proj.weight" in table:
        gate = (f"{mine}gate_proj.weight" in table)
        return AffineExperts(
            up=_affine_side(table, mine + "up_proj", shared + "up_proj", bits, group, device),
            down=_affine_side(table, mine + "down_proj", shared + "down_proj", bits, group, device),
            gate=_affine_side(table, mine + "gate_proj", shared + "gate_proj", bits, group, device) if gate else None,
            bits=bits, group=group)
    gate = (f"{mine}gate_proj.qweight" in table)
    up = _gptq_side(table, mine + "up_proj", shared + "up_proj", device)
    down = _gptq_side(table, mine + "down_proj", shared + "down_proj", device)
    pair = (_gptq_side(table, mine + "gate_proj", shared + "gate_proj", device), up) if gate else (up,)
    return GptqExperts(
        up=torch.stack([part[0] for part in pair]).contiguous(),
        up_z=torch.stack([part[1] for part in pair]).contiguous(),
        up_s=torch.stack([part[2] for part in pair]).contiguous(),
        down=down[0][None].contiguous(), down_z=down[1][None].contiguous(), down_s=down[2][None].contiguous(),
        group=group, v2=_V2)


def _routed(table, prefix: str, spec: Spec, bits: int, group: int, device: torch.device):
    """A layer's router rows [E + 1, D] bf16 (the shared expert's gate row last) and its E + 1 experts."""

    from tensorfold.rocm.moe import Routed

    router = torch.cat([_router_rows(table, prefix + "gate", group, device),
                        _router_rows(table, prefix + "shared_expert_gate", group, device)])
    if router.shape[0] != spec.experts + 1 or router.shape[1] != spec.hidden:
        raise ValueError(f"{prefix}gate carries {tuple(router.shape)} rows, want {spec.experts + 1} x {spec.hidden}")
    return Routed(router.to(torch.bfloat16).contiguous(), _experts(table, prefix, spec, bits, group, device),
                  spec.top_k)


def _halves(fused: Packed) -> tuple[Packed, Packed]:
    """A fused ``[embedding | hidden]`` projection split into its two halves, words and group tables alike."""

    if fused.bits not in (2, 4, 8):
        raise ValueError(f"a fused fc at {fused.bits} bits does not split on a word boundary")
    words, scale, bias = fused.words, fused.scale, fused.bias
    if words.shape[1] % 2 or scale.shape[1] % 2:
        raise ValueError(f"fused fc {tuple(words.shape)} {tuple(scale.shape)} does not split in half")
    word, table = words.shape[1] // 2, scale.shape[1] // 2
    return (Packed(words[:, :word].contiguous(), scale[:, :table].contiguous(), bias[:, :table].contiguous(),
                   fused.bits, fused.group),
            Packed(words[:, word:].contiguous(), scale[:, table:].contiguous(), bias[:, table:].contiguous(),
                   fused.bits, fused.group))


def _qwen3_head(table, spec: Spec, bits: int, group: int, device: torch.device, head) -> MTPHead:
    """The Qwen3 MTP layer: a fused ``fc`` in halves, a gated attention, and one dense or routed MLP."""

    base, layer = "mtp.", "mtp.layers.0."
    attn, mlp = layer + "self_attn.", layer + "mlp."
    fused_e, fused_h = _halves(_packed(table, base + "fc", bits, group, device))
    routed = (f"{mlp}switch_mlp.up_proj.weight" in table or f"{mlp}switch_mlp.up_proj.qweight" in table)
    if routed:
        dense = (None, None, None)
        moe = _routed(table, mlp, spec, bits, group, device)
    else:
        dense = tuple(_packed(table, mlp + name, bits, group, device)
                      for name in ("gate_proj", "up_proj", "down_proj"))
        moe = None
    return MTPHead(
        _float(table, base + "pre_fc_norm_embedding.weight", device),
        _float(table, base + "pre_fc_norm_hidden.weight", device),
        fused_e, fused_h,
        _float(table, attn + "q_norm.weight", device), _float(table, attn + "k_norm.weight", device),
        _packed(table, attn + "q_proj", bits, group, device), _packed(table, attn + "k_proj", bits, group, device),
        _packed(table, attn + "v_proj", bits, group, device), _packed(table, attn + "o_proj", bits, group, device),
        _float(table, base + "norm.weight", device), head,
        input_norm=_float(table, layer + "input_layernorm.weight", device),
        post_norm=_float(table, layer + "post_attention_layernorm.weight", device),
        gate=dense[0], up=dense[1], down=dense[2], moe=moe, gated=True)


def load_mtp_head(path: str | Path, spec: Spec, bits: int, group: int, device: torch.device,
                  table=None) -> MTPHead | None:
    """Read the MTP layer from ``mtp*.safetensors`` or from the checkpoint's own shards; ``None`` when absent.

    The Qwen3.6 conversions ship the head beside the weights, while the Qwen3.5 and Qwen3.8 ones keep
    ``mtp.*`` among the model's own tensors, so both are read. A checkpoint whose shards carry no head
    loads without one, but a dedicated ``mtp*.safetensors`` that does not form a head is refused by name:
    silently dropping it would cost drafts with nothing said. The Qwen3 layout (``pre_fc_norm_embedding``,
    a fused ``fc``, ``layers.0.*``) goes to :func:`_qwen3_head`; anything else with the prefix is the
    Flash Next head. ``table`` is the checkpoint's own table when the caller already has one open.
    """

    root = Path(path)
    side = sorted(root.glob("mtp*.safetensors"))
    shards = side or sorted(one for one in root.glob("*.safetensors") if "mtp" not in one.name.lower())
    if not shards:
        return None
    if side or table is None:
        # A dedicated MTP file holds the head; the model's shards do not. Some keep the VLM's prefix.
        table = _Shards(shards, strip="language_model.")
    base = "mtp."
    if not side and f"{base}norm_e.weight" not in table and f"{base}pre_fc_norm_embedding.weight" not in table:
        return None
    try:
        head = None
        if f"{base}head_proj.weight" in table:
            head = _packed(table, f"{base}head_proj", bits, group, device)
        if f"{base}pre_fc_norm_embedding.weight" in table:
            return _qwen3_head(table, spec, bits, group, device, head)
        return MTPHead(
            _float(table, f"{base}norm_e.weight", device),
            _float(table, f"{base}norm_h.weight", device),
            _packed(table, f"{base}fc_e", bits, group, device),
            _packed(table, f"{base}fc_h", bits, group, device),
            _float(table, f"{base}q_norm.weight", device),
            _float(table, f"{base}k_norm.weight", device),
            _packed(table, f"{base}q_proj", bits, group, device),
            _packed(table, f"{base}k_proj", bits, group, device),
            _packed(table, f"{base}v_proj", bits, group, device),
            _packed(table, f"{base}o_proj", bits, group, device),
            _float(table, f"{base}final_norm.weight", device),
            head,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{shards[0]} has the MTP prefix but is incomplete or wrong: {exc}") from None


def load(path: str | Path, device: torch.device | None = None) -> TextModel:
    """Load the text tower. Vision weights and any MTP head are left on disk."""

    from safetensors import safe_open

    device = device or torch.device("cuda")
    root = Path(path)
    cfg = json.loads((root / "config.json").read_text())
    if cfg.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError(f"expected model_type qwen3_5 or qwen3_5_moe, got {cfg.get('model_type')}")
    text = cfg.get("text_config") or cfg
    quant = cfg.get("quantization") or text.get("quantization") or {}
    mode = quant.get("mode")
    if mode in ("gptq", "awq"):
        global _V2
        _V2 = mode == "awq"
        bits, group = 4, int(quant.get("group_size", 128))
    else:
        bits, group = _affine_quant(quant)
    experts = int(text.get("num_experts", 0) or 0)
    top_k = int(text.get("num_experts_per_tok", 0) or 0)
    moe_width = int(text.get("moe_intermediate_size", 0) or 0)
    if experts:
        if not 0 < top_k <= experts or moe_width <= 0:
            raise ValueError(f"a {experts}-expert layer needs num_experts_per_tok in 1..{experts} and a "
                             f"moe_intermediate_size, got top_k={top_k} width={moe_width}")
        if not text.get("norm_topk_prob", True):
            raise ValueError("the RDNA MoE path renormalizes the picked weights over the top k, and this "
                             "checkpoint sets norm_topk_prob false")
    tied = bool(cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", True)))
    head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
    rope = text.get("rope_parameters") or {}
    partial = float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25)))
    rotary = int(head_dim * partial)
    spec = Spec(
        hidden=int(text["hidden_size"]), intermediate=int(text.get("intermediate_size", 0) or 0),
        n_layers=int(text["num_hidden_layers"]), heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]), head_dim=head_dim,
        key_heads=int(text["linear_num_key_heads"]), value_heads=int(text["linear_num_value_heads"]),
        key_dim=int(text["linear_key_head_dim"]), value_dim=int(text["linear_value_head_dim"]),
        conv=int(text["linear_conv_kernel_dim"]), vocab=int(text["vocab_size"]),
        eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_theta=float(rope.get("rope_theta", text.get("rope_theta") or 10_000_000)),
        rotary_dim=rotary, full_every=int(text.get("full_attention_interval", 4)),
        bits=bits, group=group, experts=experts, top_k=top_k, moe_width=moe_width,
    )
    if rotary % 2 or not 0 < rotary <= head_dim:
        raise ValueError(f"rotary dim {rotary} does not fit head dim {head_dim}")
    kinds = text.get("layer_types")
    if kinds is not None:
        for index, kind in enumerate(kinds):
            want = "full_attention" if spec.full(index) else "linear_attention"
            if kind != want:
                raise ValueError(f"layer {index} is {kind}, the interval says {want}")
    shards = sorted(path for path in root.glob("*.safetensors") if "mtp" not in path.name.lower())
    if not shards:
        raise ValueError(f"no safetensors weights in {root}")
    table = _Shards(shards)
    prefix = "language_model.model."
    embed = _packed(table, prefix + "embed_tokens", bits, group, device)
    if not isinstance(embed, Packed):
        raise ValueError("the RDNA embedding gather reads MLX affine rows, and this checkpoint's embedding is "
                         "not one")
    if embed.words.shape[0] != spec.vocab:
        raise ValueError("embedding rows are not the vocabulary")
    layers = []
    for index in range(spec.n_layers):
        base = f"{prefix}layers.{index}."
        norms = (_float(table, base + "input_layernorm.weight", device),
                 _float(table, base + "post_attention_layernorm.weight", device))
        if spec.experts:
            mlp = (None, None, None, _routed(table, base + "mlp.", spec, bits, group, device))
        else:
            mlp = (*tuple(_packed(table, base + f"mlp.{name}_proj", bits, group, device)
                          for name in ("gate", "up", "down")), None)
        if spec.full(index):
            attn = base + "self_attn."
            layers.append(FullLayer(
                *norms, _packed(table, attn + "q_proj", bits, group, device),
                _packed(table, attn + "k_proj", bits, group, device),
                _packed(table, attn + "v_proj", bits, group, device),
                _packed(table, attn + "o_proj", bits, group, device),
                _float(table, attn + "q_norm.weight", device), _float(table, attn + "k_norm.weight", device),
                *mlp))
        else:
            lin = base + "linear_attn."
            layers.append(LinearLayer(
                *norms, _packed(table, lin + "in_proj_qkv", bits, group, device),
                _packed(table, lin + "in_proj_z", bits, group, device),
                _packed(table, lin + "in_proj_a", bits, group, device),
                _packed(table, lin + "in_proj_b", bits, group, device),
                _conv(table, lin + "conv1d.weight", device),
                _float(table, lin + "A_log", device), _float(table, lin + "dt_bias", device),
                _float(table, lin + "norm.weight", device),
                _packed(table, lin + "out_proj", bits, group, device), *mlp))
    head = embed if tied else _packed(table, "language_model.lm_head", bits, group, device)
    mtp = load_mtp_head(root, spec, bits, group, device, table)
    return TextModel(spec, embed, layers, _float(table, prefix + "norm.weight", device), head, mtp)
