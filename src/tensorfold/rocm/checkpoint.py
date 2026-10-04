"""Read a Qwen MLX affine (or GPTQ / AWQ) checkpoint into the RDNA model: layers, experts and the MTP head."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tensorfold.rocm.model import FullLayer, LinearLayer, MTPHead, TextModel
from tensorfold.rocm.qwen_math import Dense, GptqPacked, Packed, Spec

# AWQ stores literal zeros, GPTQ stores them +1; set by load() and read by _packed_gptq.
_V2 = False


class _Shards:
    """One logical tensor table over every safetensors file in a checkpoint."""

    def __init__(self, paths: list[Path], strip: str = "", quant: dict | None = None):
        """``strip`` drops a leading key prefix; ``quant`` is the config's quantization, per-tensor widths included."""

        from safetensors import safe_open

        self.strip, self.quant = strip, quant or {}
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

    def width(self, key: str, bits: int, group: int) -> tuple[int, int]:
        """The tensor's own (bits, group) when the config names it (mixed-width conversions), else the default."""

        for name in (key, self.strip + key):
            if isinstance(self.quant.get(name), dict):
                return _affine_quant(self.quant[name])
        return bits, group


def _width(table, key: str, bits: int, group: int) -> tuple[int, int]:
    return table.width(key, bits, group) if hasattr(table, "width") else (bits, group)


def _float(table, key: str, device: torch.device) -> torch.Tensor:
    return table.get_tensor(key).to(device=device, dtype=torch.float32).contiguous()


def _packed(table, key: str, bits: int, group: int, device: torch.device) -> Packed:
    try:
        words = table.get_tensor(key + ".weight")
    except KeyError:
        return _packed_gptq(table, key, device)
    if words.dtype.is_floating_point and words.ndim == 2 and key + ".scales" not in table:
        return Dense(words.to(device=device, dtype=torch.float32).contiguous())
    bits, group = _width(table, key, bits, group)
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
    if quant.get("mode", "affine") != "affine" or bits not in (2, 3, 4, 5, 6, 8) or group not in (32, 64, 128):
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
    group = _width(table, key, 0, group)[1]
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
    """A layer's ``E + 1`` experts, shared last, in the checkpoint's kind (affine or GPTQ / AWQ)."""

    from tensorfold.rocm.experts import AffineExperts, GptqExperts

    mine, shared = prefix + "switch_mlp.", prefix + "shared_expert."
    if f"{mine}up_proj.weight" in table:
        gate = (f"{mine}gate_proj.weight" in table)
        up_bits, up_group = _width(table, mine + "up_proj", bits, group)
        down_bits, down_group = _width(table, mine + "down_proj", bits, group)
        if gate and _width(table, mine + "gate_proj", bits, group) != (up_bits, up_group):
            raise ValueError(f"{mine} gate and up differ in width: the RDNA experts run them as one projection")
        return AffineExperts(
            up=_affine_side(table, mine + "up_proj", shared + "up_proj", up_bits, up_group, device),
            down=_affine_side(table, mine + "down_proj", shared + "down_proj", down_bits, down_group, device),
            gate=(_affine_side(table, mine + "gate_proj", shared + "gate_proj", up_bits, up_group, device)
                  if gate else None),
            bits=up_bits, group=up_group, down_bits=down_bits, down_group=down_group)
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


def _halves(fused: Packed | Dense) -> tuple[Packed | Dense, Packed | Dense]:
    """A fused ``[embedding | hidden]`` projection split into its two halves, words and group tables alike."""

    if isinstance(fused, Dense):
        half = fused.weight.shape[1] // 2
        return Dense(fused.weight[:, :half].contiguous()), Dense(fused.weight[:, half:].contiguous())
    words, scale, bias = fused.words, fused.scale, fused.bias
    half = scale.shape[1] * fused.group // 2
    if scale.shape[1] % 2 or half * fused.bits % 32:
        raise ValueError(f"fused fc {tuple(words.shape)} {tuple(scale.shape)} does not split on a word boundary")
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
    """The MTP layer from ``mtp*.safetensors`` or the model's own shards; None when absent."""

    root = Path(path)
    side = sorted(root.glob("mtp*.safetensors"))
    shards = side or sorted(one for one in root.glob("*.safetensors") if "mtp" not in one.name.lower())
    if not shards:
        return None
    base = "mtp."
    own = table is not None and (f"{base}norm_e.weight" in table or f"{base}pre_fc_norm_embedding.weight" in table)
    if side or not own:
        # A side file holds the head; the model's shards may keep it under the VLM's ``language_model.`` prefix.
        table = _Shards(shards, strip="language_model.", quant=getattr(table, "quant", None) or _quant_of(root))
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


def _quant_of(root: Path) -> dict:
    """The checkpoint config's quantization table (per-tensor widths included), or an empty one."""

    path = root / "config.json"
    if not path.is_file():
        return {}
    cfg = json.loads(path.read_text())
    return cfg.get("quantization") or (cfg.get("text_config") or {}).get("quantization") or {}


def load(path: str | Path, device: torch.device | None = None) -> TextModel:
    """Load the text tower. Vision weights and any MTP head are left on disk."""

    device = device or torch.device("cuda")
    root = Path(path)
    cfg = json.loads((root / "config.json").read_text())
    if cfg.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError(f"expected model_type qwen3_5 or qwen3_5_moe, got {cfg.get('model_type')}")
    text = cfg.get("text_config") or cfg
    quant = cfg.get("quantization") or text.get("quantization") or {}
    if not quant and (cfg.get("quantization_config") or {}).get("quant_method") in ("gptq", "awq"):
        raise ValueError("Hugging Face GPTQ / AWQ exports (quantization_config) are not served on ROCm: its W4A16 "
                         "path reads MLX-layout exports, with stacked switch_mlp experts and an affine embedding")
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
    table = _Shards(shards, quant=quant)
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
