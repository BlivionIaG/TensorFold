"""Oracle dump of the Python ROCm Qwen engine: affine matmul fixtures and per-layer residuals, as .npy files."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)
ROWS = (1, 2, 5, 8, 17, 64, 300)
N_OUT = 96


class Dump:
    """A directory of .npy files and the manifest that describes them."""

    def __init__(self, out: Path, kind: str, meta: dict):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = {"kind": kind, **meta, "files": [], "cases": []}

    def save(self, name: str, tensor: torch.Tensor, meaning: str) -> str:
        """Write one tensor with its raw dtype; bf16 is stored as uint16 bits under a .bf16.npy name."""

        t = tensor.detach().cpu().contiguous()
        dtype = str(t.dtype).replace("torch.", "")
        if t.dtype == torch.bfloat16:
            array, fname = t.view(torch.int16).numpy().view(np.uint16), f"{name}.bf16.npy"
        else:
            array, fname = t.numpy(), f"{name}.npy"
        np.save(self.out / fname, array)
        self.manifest["files"].append({"name": fname, "dtype": dtype, "shape": list(t.shape), "meaning": meaning})
        return fname

    def close(self) -> None:
        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=1) + "\n")


def _engine_env():
    from tensorfold.rocm.kernels.build import gfx_name
    from tensorfold.rocm.model.qwen import activation_dtype

    gfx = gfx_name()
    return gfx, activation_dtype(gfx)


def affine(out: Path) -> None:
    from tensorfold.rocm.kernels import affine as kernel

    gfx, dtype = _engine_env()
    dump = Dump(out, "affine", {"gfx": gfx, "act_dtype": str(dtype).replace("torch.", ""), "schedule": "auto",
                                "n": N_OUT, "seed_rule": "case index, cpu generator"})
    index = 0
    for bits in BITS:
        for group in GROUPS:
            for m in ROWS:
                k = 512 if m >= 64 else 256
                gen = torch.Generator().manual_seed(1000 + index)
                x = torch.randn(m, k, generator=gen).to(dtype)
                words = torch.randint(-2**31, 2**31 - 1, (N_OUT, k * bits // 32), generator=gen, dtype=torch.int32)
                scale = (torch.rand(N_OUT, k // group, generator=gen) * 0.02 + 0.002).to(torch.bfloat16)
                bias = (torch.randn(N_OUT, k // group, generator=gen) * 0.05).to(torch.bfloat16)
                dev = [t.cuda() for t in (x.contiguous(), words, scale, bias)]
                args, kw = dev, {"bits": bits, "group": group, "schedule": "auto"}
                y = kernel.matmul(*args, **kw, f32=False)
                y32 = kernel.matmul(*args, **kw, f32=True)
                torch.cuda.synchronize()
                tag = f"b{bits}_g{group}_m{m}"
                files = {
                    "x": dump.save(f"{tag}.x", x, "activation, (m, k) in the act dtype"),
                    "words": dump.save(f"{tag}.words", words, "packed int32 words, (n, k*bits/32), random bits"),
                    "scale": dump.save(f"{tag}.scale", scale, "bf16 scale, (n, k/group)"),
                    "bias": dump.save(f"{tag}.bias", bias, "bf16 bias, (n, k/group)"),
                    "out": dump.save(f"{tag}.out", y, "matmul(f32=False): fp32 result rounded to the act dtype"),
                    "out_f32": dump.save(f"{tag}.out_f32", y32, "matmul(f32=True): the fp32 result"),
                }
                dump.manifest["cases"].append({"bits": bits, "group": group, "m": m, "n": N_OUT, "k": k,
                                               "files": files})
                index += 1
    dump.close()


class Residuals:
    """Collects what a patched ``_residual`` returns: every layer adds twice (attention, then mlp) per span."""

    def __init__(self):
        self.calls: list[torch.Tensor] = []

    def wrap(self, real):
        def residual(x, y):
            z = real(x, y)
            self.calls.append(z.detach().to(dtype=x.dtype).clone())
            return z
        return residual

    def take(self, n_layers: int) -> torch.Tensor:
        """(layers, rows, hidden): the stream after each layer's second add, spans joined along rows."""

        per = len(self.calls) // n_layers
        assert per * n_layers == len(self.calls) and per % 2 == 0, "unexpected residual call count"
        spans = per // 2
        layers = [torch.cat([self.calls[i * per + 2 * s + 1][0] for s in range(spans)], dim=0)
                  for i in range(n_layers)]
        self.calls = []
        return torch.stack(layers)


def _packed_info(layer) -> dict:
    from tensorfold.rocm.model.qwen_math import Packed

    return {name: {"bits": v.bits, "group": v.group, "shape": list(v.words.shape)}
            for name, v in vars(layer).items() if isinstance(v, Packed)}


def layers(model_dir: str, tokens: list[int], out: Path, decode: int) -> None:
    from tensorfold.rocm.model import forward, window
    from tensorfold.rocm.model.forward import _blank_caches, _project, forward_hidden
    from tensorfold.rocm.serving.engine import QwenEngine, draw

    gfx, dtype = _engine_env()
    engine = QwenEngine.load(model_dir, keep=0, no_drafts=True)
    model, linear, device = engine.model, engine.kernels.linear, engine._device()
    spec = model.spec
    dtype = engine._dtype()
    full = [i for i in range(spec.n_layers) if spec.full(i)]
    meta = {"gfx": gfx, "act_dtype": str(dtype).replace("torch.", ""), "model": Path(model_dir).name,
            "spec": dataclasses.asdict(spec), "full_attention_layers": full, "decode_steps": decode,
            "tokens": tokens, "embed_bits": getattr(model.embed, "bits", None),
            "layer0_projections": _packed_info(model.layers[0]),
            "layer_full_projections": _packed_info(model.layers[full[0]]) if full else {}}
    dump = Dump(out, "layers", meta)
    dump.save("tokens", torch.tensor(tokens, dtype=torch.int64), "prompt token ids")

    records, embeds = Residuals(), []
    real_gather = forward.gather_rows

    def gather(*a, **kw):
        y = real_gather(*a, **kw)
        embeds.append(y.detach().clone())
        return y

    forward._residual = records.wrap(forward._residual)
    window._residual = records.wrap(window._residual)
    forward.gather_rows = gather
    window.gather_rows = gather
    try:
        with torch.inference_mode():
            total = len(tokens) + decode + 1
            caches = _blank_caches(model, 1, total, device, dtype)
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            hidden, caches = forward_hidden(model, ids, caches, linear, 0, dtype, exact_short=True)
            logits = _project(hidden[:, -1], model.output_head(), linear)
            token = draw(logits, None, len(tokens))
            dump.save("prefill.embed", embeds.pop()[0], "embedding rows, (rows, hidden), before layer 0")
            dump.save("prefill.layers", records.take(spec.n_layers), "residual x after each layer, (layers, rows, hidden)")
            dump.save("prefill.hidden", hidden[0], "final-normed hidden, all rows (rows, hidden)")
            dump.save("prefill.logits", logits[0], "last row logits, (vocab,) in the act dtype")
            sampled = [token]
            for step in range(decode):
                pos = len(tokens) + step
                w = window.Window([sampled[-1]], caches, pos)
                h = window.window_forward(model, [w], linear, dtype)
                window.commit(model, w, 1)
                lg = _project(h[0], model.output_head(), linear)
                token = draw(lg, None, pos + 1)
                tag = f"decode{step}"
                dump.save(f"{tag}.embed", embeds.pop()[0], f"embedding row of token {sampled[-1]} at pos {pos}")
                dump.save(f"{tag}.layers", records.take(spec.n_layers), "residual x after each layer, (layers, 1, hidden)")
                dump.save(f"{tag}.hidden", h[0], "final-normed hidden row, (1, hidden)")
                dump.save(f"{tag}.logits", lg[0], "logits row, (vocab,) in the act dtype")
                sampled.append(token)
            torch.cuda.synchronize()
    finally:
        forward._residual, window._residual = _unwrap(forward), _unwrap(window)
    dump.save("sampled", torch.tensor(sampled, dtype=torch.int64),
              "greedy tokens: [0] from the prefill logits, [i] from decode step i-1")
    dump.manifest["sampled"] = sampled
    dump.manifest["decode_note"] = "decode step i feeds sampled[i] at position len(tokens)+i through window_forward"
    # The engine's own generate, same prompt, same GPU, drafts off.
    got: list[int] = []
    engine.generate(list(tokens), decode + 1, None, lambda ts: got.extend(ts) and False, stop_eos=False, draft=False)
    dump.manifest["generate_tokens"] = got
    dump.manifest["matches_generate"] = got == sampled
    dump.close()
    engine.close()
    print("sampled ", sampled)
    print("generate", got)
    print("match   ", got == sampled)
    if got != sampled:
        sys.exit(1)


def _unwrap(module):
    fn = module._residual
    return getattr(fn, "__wrapped__", None) or _ORIGINAL[module.__name__]


def _remember_originals() -> None:
    from tensorfold.rocm.model import forward, window

    _ORIGINAL[forward.__name__] = forward._residual
    _ORIGINAL[window.__name__] = window._residual


_ORIGINAL: dict = {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("affine", help="packed affine matmul fixtures")
    a.add_argument("out")
    d = sub.add_parser("layers", help="per-layer residuals of a prefill and greedy decode steps")
    d.add_argument("model")
    d.add_argument("tokens", help="comma separated token ids")
    d.add_argument("out")
    d.add_argument("--decode", type=int, default=0)
    args = p.parse_args()
    _remember_originals()
    if args.cmd == "affine":
        affine(Path(args.out))
    else:
        layers(args.model, [int(t) for t in args.tokens.split(",")], Path(args.out), args.decode)


if __name__ == "__main__":
    main()
