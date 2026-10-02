"""Qwen3.5 text forward on RDNA. Projections are packed affine matmuls; nothing here densifies a weight.

The measurement entry is ``python -m tensorfold.rocm.qwen MODEL_DIR``. It reports prefill tok/s,
decode tok/s, ttft, and itil for prompt/generated loads 1024/512 and 16384/1024 at concurrency 1 and 8.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorfold.rocm import qwen_math
from tensorfold.rocm.qwen_math import Packed, Spec, greedy
_CELLS = ((1024, 512, 1), (1024, 512, 8), (16384, 1024, 1), (16384, 1024, 8))
_RDNA2 = {f"gfx103{i}" for i in range(7)}
_RDNA3 = {f"gfx110{i}" for i in range(4)}


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
        if any(t.dtype.is_floating_point and t.ndim == 2 and t.shape[1] == k for t in (words,)):
            raise RuntimeError("a projection expanded its weight to a dense matrix")
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


def _prompts(prompt_len: int, generated: int, concurrency: int, vocab: int) -> list[list[int]]:
    if prompt_len < 1 or generated < 2 or concurrency < 1:
        raise ValueError("a measured cell needs a prompt, at least two generated tokens, and one request")
    # Distinct requests. Token 0 is avoided so a pad id is not the whole prompt. EOS is not consulted.
    return [[(index + 1 + row * 17) % (vocab - 1) + 1 for index in range(prompt_len)] for row in range(concurrency)]


def measure_cell(engine: Engine, prompt_len: int, generated: int, concurrency: int) -> dict:
    """One shared wall for ``concurrency`` requests. Clocks move only after the device synchronizes."""

    spec = engine.model.spec
    prompts = _prompts(prompt_len, generated, concurrency, spec.vocab)
    stamps: list[float] = []

    def after(step: int) -> None:
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())
        if step > 0 and step % 64 == 0:
            print(f"# c={concurrency} prompt={prompt_len} token {step}/{generated}", file=sys.stderr, flush=True)

    ids = engine.generate(prompts, generated, after_token=after)
    if len(stamps) != generated + 1:
        raise RuntimeError("the clock did not record a start and every generated token")
    counts = [len(row) for row in ids]
    if counts != [generated] * concurrency:
        raise RuntimeError(f"generated {counts}, requested {generated} from each of {concurrency} requests")
    t0, t_first, t_last = stamps[0], stamps[1], stamps[-1]
    prefill_wall = t_first - t0
    decode_wall = t_last - t_first
    if prefill_wall <= 0 or decode_wall <= 0:
        raise RuntimeError("non-positive measured wall")
    ttft = [prefill_wall] * concurrency
    itil = [decode_wall / (generated - 1)] * concurrency
    prefill_tokens = concurrency * prompt_len
    decode_tokens = concurrency * (generated - 1)
    return {
        "prompt": prompt_len, "generated": generated, "concurrency": concurrency,
        "prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens,
        "prefill_tok_s": prefill_tokens / prefill_wall, "decode_tok_s": decode_tokens / decode_wall,
        "ttft_s": statistics.median(ttft), "itil_s": statistics.median(itil),
        "ttft_each": ttft, "itil_each": itil, "generated_each": counts,
    }


def _finite_positive(row: dict) -> None:
    for key in ("prefill_tok_s", "decode_tok_s", "ttft_s", "itil_s"):
        value = row[key]
        if not (value > 0) or value != value or value == float("inf"):
            raise RuntimeError(f"{key} is not a finite positive rate ({value})")


def _format(gfx: str, dtype: torch.dtype, row: dict) -> str:
    activation = "bf16" if dtype == torch.bfloat16 else "fp16"
    each_t = ",".join(f"{v:.9g}" for v in row["ttft_each"])
    each_i = ",".join(f"{v:.9g}" for v in row["itil_each"])
    each_n = ",".join(str(v) for v in row["generated_each"])
    return (f"cell gfx={gfx} activation={activation} prompt={row['prompt']} generated={row['generated']} "
            f"concurrency={row['concurrency']} prefill_tok_s={row['prefill_tok_s']:.9g} "
            f"decode_tok_s={row['decode_tok_s']:.9g} ttft_s={row['ttft_s']:.9g} itil_s={row['itil_s']:.9g} "
            f"prefill_tokens={row['prefill_tokens']} decode_tokens={row['decode_tokens']} "
            f"ttft_each={each_t} itil_each={each_i} generated_each={each_n}")


def measure(path: str | Path, cells=_CELLS) -> list[str]:
    """Load the checkpoint and print one line per cell. Refuses a gfx outside RDNA2 and RDNA3 WMMA."""

    from tensorfold.rocm.build import gfx_name

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RuntimeError("no HIP device is visible")
    gfx = gfx_name()
    if gfx not in _RDNA2 and gfx not in _RDNA3:
        raise RuntimeError(f"{gfx} is not an RDNA2 (gfx1030-gfx1036) or RDNA3 (gfx1100-gfx1103) measurement target")
    dtype = activation_dtype(gfx)
    model = load(path)
    engine = Engine(model, schedule="auto", dtype=dtype)
    probe = model.layers[0]
    packed = probe.qkv if isinstance(probe, LinearLayer) else probe.q
    sample = torch.zeros(1, model.spec.hidden, device=packed.words.device, dtype=dtype)
    got = engine.linear(sample, packed)
    if got.shape != (1, packed.words.shape[0]) or not torch.isfinite(got).all():
        raise RuntimeError("the packed projection did not return a finite row")
    print(f"loaded gfx={gfx} activation={'bf16' if dtype == torch.bfloat16 else 'fp16'} "
          f"layers={model.spec.n_layers} hidden={model.spec.hidden} vocab={model.spec.vocab} "
          f"bits={model.spec.bits} group={model.spec.group} "
          f"embed_words={model.embed.words.shape[0]}x{model.embed.words.shape[1]} dtype=int32", flush=True)
    # The extension is already built. A short generate pays for the first launch before a cell is timed.
    engine.generate(_prompts(8, 2, 1, model.spec.vocab), 2)
    torch.cuda.synchronize()
    print("# warmup done", file=sys.stderr, flush=True)
    lines = []
    for prompt_len, generated, concurrency in cells:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"# start prompt={prompt_len} generated={generated} concurrency={concurrency}", file=sys.stderr, flush=True)
        row = measure_cell(engine, prompt_len, generated, concurrency)
        _finite_positive(row)
        if (row["prompt"], row["generated"], row["concurrency"]) != (prompt_len, generated, concurrency):
            raise RuntimeError("cell lengths do not match the requested load")
        line = _format(gfx, dtype, row)
        print(line, flush=True)
        lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cells = _CELLS
    if len(args) == 4:
        cells = ((int(args[1]), int(args[2]), int(args[3])),)
    elif len(args) != 1:
        print("usage: python -m tensorfold.rocm.qwen MODEL_DIR [PROMPT GENERATED CONCURRENCY]", file=sys.stderr)
        return 2
    measure(args[0], cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
