"""Qwen3.5 text forward on RDNA. Projections are packed affine matmuls; nothing here densifies a weight.

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

    def output_head(self) -> Packed:
        return self.embed if self.head is None else self.head


def activation_dtype(gfx: str) -> torch.dtype:
    """FP16 on RDNA2, where the affine schedule is the FP16 dot. BF16 on an RDNA3 WMMA part."""

    from tensorfold.rocm.build import WMMA

    if gfx in _RDNA2:
        return torch.float16
    if gfx in WMMA:
        return torch.bfloat16
    raise RuntimeError(f"no activation dtype for {gfx}")


class Engine:
    """Shipped forward. ``schedule`` selects the affine kernel; ``dtype`` is the activation type."""

    def __init__(self, model: TextModel, schedule: str = "auto", dtype: torch.dtype | None = None):
        self.model = model
        self.schedule = schedule
        self.dtype = dtype
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
        kwargs = {"bits": packed.bits, "group": packed.group, "schedule": schedule}
        span = qwen_math.SPAN
        if flat.shape[0] <= span:
            return affine_mod.matmul(flat, words, scale, bias, **kwargs)
        # One activation-dtype buffer. Keeping every chunk and then concatenating doubles a long prefill.
        out = torch.empty(flat.shape[0], words.shape[0], dtype=flat.dtype, device=flat.device)
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
        with torch.inference_mode():
            return greedy(self.model, prompts, n_new, self.linear, device, after_token=after_token,
                          cache_dtype=self.dtype)


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
    return Packed(words.to(device).contiguous(), scale.to(device=device, dtype=torch.float32).contiguous(),
                  bias.to(device=device, dtype=torch.float32).contiguous(), bits, group)


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
    return TextModel(spec, embed, layers, _float(table, prefix + "norm.weight", device), head)
