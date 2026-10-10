#!/usr/bin/env python3
"""MLX affine checkpoints at other widths, quantized with mlx.core.quantize's affine rule.

  mlx_requant.py requant <mlx dir> <bits> <out dir> [--group G]
      every quantized tensor dequantized and quantized again at `bits` (per-module overrides keep their width)
  mlx_requant.py check <raw dir> <mlx dir>
      quantizes the raw tensors the MLX checkpoint holds quantized and reports how many codes agree; published
      checkpoints may come from calibrated conversions, so agreement, not equality, is expected
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def unpack(words: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Values of `bits` bits packed little-endian across uint32 words, (rows, n) int64."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    at = torch.arange(n) * bits
    word, shift = at // 32, at % 32
    low = w[..., word] >> shift
    high = w[..., (word + 1).clamp(max=w.shape[-1] - 1)] << (32 - shift)
    return (low | high) & ((1 << bits) - 1)


def pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """(rows, n) values below 2^bits into uint32 words (rows, n * bits / 32), stored as int32 like MLX's files."""
    rows, n = q.shape
    out = torch.zeros(rows, n * bits // 32, dtype=torch.int64)
    at = torch.arange(n) * bits
    word, shift = at // 32, at % 32
    q = q.to(torch.int64)
    out.index_add_(1, word, (q << shift) & 0xFFFFFFFF)
    spill = shift + bits > 32
    if spill.any():
        out.index_add_(1, word[spill] + 1, q[:, spill] >> (32 - shift[spill]))
    return (out & 0xFFFFFFFF).to(torch.uint32).view(torch.int32)


def dequant(words, scales, biases, bits: int) -> torch.Tensor:
    groups = scales.shape[-1]
    n = words.shape[-1] * 32 // bits
    q = unpack(words, bits, n).to(torch.float32).view(*words.shape[:-1], groups, n // groups)
    return (q * scales.float()[..., None] + biases.float()[..., None]).view(*words.shape[:-1], n)


def quantize(w: torch.Tensor, bits: int, group: int, dtype: torch.dtype):
    """mlx.core.quantize (affine): per group the edge of larger magnitude lands exactly on a code."""
    shape = w.shape
    g = w.float().reshape(-1, shape[-1] // group, group)
    n_bins = float((1 << bits) - 1)
    w_max, w_min = g.amax(-1, keepdim=True), g.amin(-1, keepdim=True)
    mask = w_min.abs() > w_max.abs()
    scales = torch.clamp((w_max - w_min) / n_bins, min=1e-7)
    scales = torch.where(mask, scales, -scales)
    edge = torch.where(mask, w_min, w_max)
    q0 = torch.round(edge / scales)
    scales = torch.where(q0 != 0, edge / q0, scales)
    biases = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    q = torch.clamp(torch.round((g - biases) / scales), 0, n_bins).reshape(-1, shape[-1])
    words = pack(q, bits).reshape(*shape[:-1], shape[-1] * bits // 32)
    return words, scales.squeeze(-1).to(dtype).reshape(*shape[:-1], -1), biases.squeeze(-1).to(dtype).reshape(*shape[:-1], -1)


def tensors(directory: Path) -> dict[str, tuple[Path, str]]:
    found = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                found[k] = (f, k)
    return found


def load(where: tuple[Path, str]) -> torch.Tensor:
    with safe_open(where[0], "pt") as h:
        return h.get_tensor(where[1])


def requant(src: Path, bits: int, out: Path, group: int | None) -> None:
    cfg = json.loads((src / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    if not quant or quant.get("mode", "affine") != "affine":
        raise SystemExit("not an MLX affine checkpoint")
    group = group or quant["group_size"]
    overrides = {k: v for k, v in quant.items() if isinstance(v, dict)}
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.name not in ("config.json", "model.safetensors.index.json") and f.is_file():
            shutil.copy2(f, out / f.name)
    for f in sorted(src.glob("*.safetensors")):
        new = {}
        with safe_open(f, "pt") as h:
            keys = list(h.keys())
            for k in keys:
                if k.endswith((".scales", ".biases")):
                    continue
                t = h.get_tensor(k)
                base = k[: -len(".weight")] if k.endswith(".weight") else None
                if base and base + ".scales" in keys and not any(base.endswith(o) or o.endswith(base) for o in overrides):
                    s, b = h.get_tensor(base + ".scales"), h.get_tensor(base + ".biases")
                    n_in = s.shape[-1] * (quant.get("group_size") or 64)
                    old_bits = t.shape[-1] * 32 // n_in
                    w = dequant(t, s, b, old_bits)
                    words, scales, biases = quantize(w, bits, group, s.dtype)
                    new[k], new[base + ".scales"], new[base + ".biases"] = words, scales, biases
                elif base and base + ".scales" in keys:
                    new[k], new[base + ".scales"], new[base + ".biases"] = t, h.get_tensor(base + ".scales"), h.get_tensor(base + ".biases")
                else:
                    new[k] = t
        save_file(new, str(out / f.name), metadata={"format": "mlx"})
        print(f"{f.name}: {len(new)} tensors")
    quant.update(bits=bits, group_size=group)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    index = src / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        idx["weight_map"] = {k: v for k, v in idx["weight_map"].items()}
        (out / index.name).write_text(json.dumps(idx, indent=2))


def raw_name(mlx_name: str, raw: dict) -> str | None:
    """The raw checkpoint's name of an MLX tensor (MLX's language_model.model. is the raw model.language_model.)."""
    for cand in (mlx_name, mlx_name.replace("language_model.model.", "model.language_model."),
                 mlx_name.replace("language_model.lm_head", "lm_head")):
        if cand in raw:
            return cand
    return None


def check(raw_dir: Path, mlx_dir: Path) -> None:
    cfg = json.loads((mlx_dir / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    bits, group = quant["bits"], quant["group_size"]
    raw, mlx = tensors(raw_dir), tensors(mlx_dir)
    compared = equal = 0
    worst = []
    for k in mlx:
        if not k.endswith(".scales"):
            continue
        base = k[: -len(".scales")]
        name = raw_name(base + ".weight", raw)
        if name is None:
            continue
        w = load(raw[name])
        words, scales, biases = quantize(w, bits, group, load(mlx[k]).dtype)
        want = [load(mlx[base + ".weight"]), load(mlx[k]), load(mlx[base + ".biases"])]
        same = torch.equal(words, want[0].view(torch.int32)) and torch.equal(scales, want[1]) and torch.equal(biases, want[2])
        compared += 1
        equal += int(same)
        if not same:
            n = words.shape[-1] * 32 // bits
            got_q = unpack(words.reshape(-1, words.shape[-1]), bits, n)
            want_q = unpack(want[0].view(torch.int32).reshape(-1, want[0].shape[-1]), bits, n)
            diff = (got_q != want_q).float().mean().item()
            worst.append((diff, base))
    print(f"{equal} of {compared} quantized tensors byte-equal to the MLX checkpoint")
    for diff, base in sorted(worst, reverse=True)[:8]:
        print(f"  {base}: {100 * diff:.4f}% of codes differ")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("requant")
    r.add_argument("src", type=Path), r.add_argument("bits", type=int), r.add_argument("out", type=Path)
    r.add_argument("--group", type=int)
    c = sub.add_parser("check")
    c.add_argument("raw", type=Path), c.add_argument("mlx", type=Path)
    a = ap.parse_args()
    if a.cmd == "requant":
        if a.bits not in (2, 3, 4, 5, 6, 8):
            raise SystemExit("bits is 2, 3, 4, 5, 6 or 8")
        requant(a.src, a.bits, a.out, a.group)
    else:
        check(a.raw, a.mlx)


if __name__ == "__main__":
    main()
