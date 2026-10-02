"""Qwen3.5 text forward on RDNA. Projections are packed affine matmuls.

Serving cells are measured by ``python -m tensorfold.rocm.bench MODEL_DIR``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorfold.rocm import qwen_math
from tensorfold.rocm.qwen_math import Packed, Spec, greedy

_RDNA2 = {f"gfx103{i}" for i in range(7)}


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


def _even(size: int, world: int, name: str) -> int:
    if size % world:
        raise ValueError(f"{name}: {size} does not split into {world} equal parts")
    return size // world


def _rows(packed: Packed, spans: list[tuple[int, int]]) -> Packed:
    """Output rows ``spans`` of ``packed``, in order: one rank's share of a column-split projection."""

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

    groups = _even(packed.scale.shape[1], world, f"{name} groups")
    words = groups * packed.group * packed.bits // 32
    return Packed(packed.words[:, rank * words:(rank + 1) * words].contiguous(),
                  packed.scale[:, rank * groups:(rank + 1) * groups].contiguous(),
                  packed.bias[:, rank * groups:(rank + 1) * groups].contiguous(), packed.bits, packed.group,
                  partial=True)


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
    for name in ("heads", "kv_heads", "key_heads", "value_heads"):
        _even(getattr(spec, name), world, name)
    _even(spec.vocab, world, "vocab")
    key_width, value_width = spec.key_width, spec.value_width
    for index, layer in enumerate(model.layers):
        base = f"layer {index}"
        if isinstance(layer, FullLayer):
            layer.q = _rows(layer.q, _row_spans([spec.heads * spec.head_dim * 2], rank, world, f"{base} q_proj"))
            layer.k = _rows(layer.k, _row_spans([spec.kv_heads * spec.head_dim], rank, world, f"{base} k_proj"))
            layer.v = _rows(layer.v, _row_spans([spec.kv_heads * spec.head_dim], rank, world, f"{base} v_proj"))
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
        mlp = _row_spans([layer.gate.words.shape[0]], rank, world, f"{base} mlp")
        layer.gate = _rows(layer.gate, mlp)
        layer.up = _rows(layer.up, mlp)
        layer.down = _cols(layer.down, rank, world, f"{base} mlp.down_proj")
    model.head = _rows(model.output_head(), _row_spans([spec.vocab], rank, world, "vocab"))
    spec.heads //= world
    spec.kv_heads //= world
    spec.key_heads //= world
    spec.value_heads //= world
    return model


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
    """One dense attention-only MTP head (the Qwen4_exp / Flash Next shape).

    A draft token at position ``i + 1`` is produced from the main model's residual at position ``i`` and
    the embedding of the token sampled at position ``i``. The head shares the main model's embed and (by
    default) its lm_head; a checkpoint that ships an MTP-specific final projection names it ``mtp.head_proj``.
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
        flat = flat.reshape(-1, flat.shape[-1]).to(dtype=self.dtype).contiguous()
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

    def __init__(self, paths: list[Path]):
        from safetensors import safe_open

        self._open = [safe_open(str(path), framework="pt") for path in paths]
        self._where: dict[str, int] = {}
        for index, table in enumerate(self._open):
            for key in table.keys():
                self._where[key] = index

    def get_tensor(self, key: str) -> torch.Tensor:
        try:
            index = self._where[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        return self._open[index].get_tensor(key)


def _float(table, key: str, device: torch.device) -> torch.Tensor:
    return table.get_tensor(key).to(device=device, dtype=torch.float32).contiguous()


def _packed(table, key: str, bits: int, group: int, device: torch.device) -> Packed:
    words = table.get_tensor(key + ".weight")
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


def load_mtp_head(path: str | Path, spec: Spec, bits: int, group: int, device: torch.device) -> MTPHead | None:
    """Read ``mtp-4bit.safetensors`` (or any ``mtp*.safetensors``) from ``path``; ``None`` when absent.

    Refuses a partial / malformed MTP file by name. The head's ``head`` projection is the dedicated
    ``mtp.head_proj`` tensor when the checkpoint ships one; otherwise the caller ties with the main
    model's ``lm_head`` (or tied embedding).
    """

    root = Path(path)
    shards = sorted(root.glob("mtp*.safetensors"))
    if not shards:
        return None
    table = _Shards(shards)
    base = "mtp."
    try:
        head = None
        if f"{base}head_proj.weight" in table:
            head = _packed(table, f"{base}head_proj", bits, group, device)
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
    if cfg.get("model_type") != "qwen3_5":
        raise ValueError(f"expected model_type qwen3_5, got {cfg.get('model_type')}")
    text = cfg.get("text_config") or cfg
    quant = cfg.get("quantization") or text.get("quantization") or {}
    bits, group = _affine_quant(quant)
    tied = bool(cfg.get("tie_word_embeddings", text.get("tie_word_embeddings", True)))
    head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
    rope = text.get("rope_parameters") or {}
    partial = float(rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25)))
    rotary = int(head_dim * partial)
    spec = Spec(
        hidden=int(text["hidden_size"]), intermediate=int(text["intermediate_size"]),
        n_layers=int(text["num_hidden_layers"]), heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]), head_dim=head_dim,
        key_heads=int(text["linear_num_key_heads"]), value_heads=int(text["linear_num_value_heads"]),
        key_dim=int(text["linear_key_head_dim"]), value_dim=int(text["linear_value_head_dim"]),
        conv=int(text["linear_conv_kernel_dim"]), vocab=int(text["vocab_size"]),
        eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_theta=float(rope.get("rope_theta", text.get("rope_theta") or 10_000_000)),
        rotary_dim=rotary, full_every=int(text.get("full_attention_interval", 4)),
        bits=bits, group=group,
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
    if embed.words.shape[0] != spec.vocab:
        raise ValueError("embedding rows are not the vocabulary")
    layers = []
    for index in range(spec.n_layers):
        base = f"{prefix}layers.{index}."
        norms = (_float(table, base + "input_layernorm.weight", device),
                 _float(table, base + "post_attention_layernorm.weight", device))
        mlp = tuple(_packed(table, base + f"mlp.{name}_proj", bits, group, device)
                    for name in ("gate", "up", "down"))
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
    mtp = load_mtp_head(root, spec, bits, group, device)
    return TextModel(spec, embed, layers, _float(table, prefix + "norm.weight", device), head, mtp)
