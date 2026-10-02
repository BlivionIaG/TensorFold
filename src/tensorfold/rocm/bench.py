"""RDNA serving cells: ``python -m tensorfold.rocm.bench MODEL_DIR [PROMPT GENERATED CONCURRENCY] [--runs N]``.

Each cell reports prefill tok/s, decode tok/s, ttft, itil and peak GiB for prompt/generated loads 1024/512 and
16384/1024 at concurrency 1 and 8. A cell runs ``--runs`` times (3 by default); the ``median`` line is the
number the plan compares.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

import torch

from tensorfold.rocm.qwen import Engine, LinearLayer, activation_dtype, load

CELLS = ((1024, 512, 1), (1024, 512, 8), (16384, 1024, 1), (16384, 1024, 8))
RATES = ("prefill_tok_s", "decode_tok_s", "ttft_s", "itil_s", "peak_gib")


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

    torch.cuda.reset_peak_memory_stats()
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
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "ttft_each": ttft, "itil_each": itil, "generated_each": counts,
    }


def median_row(rows: list[dict]) -> dict:
    """Each rate's median over the runs of one cell."""

    out = {key: rows[0][key] for key in ("prompt", "generated", "concurrency", "prefill_tokens", "decode_tokens")}
    for key in RATES:
        out[key] = statistics.median(row[key] for row in rows)
    return out


def _finite_positive(row: dict) -> None:
    for key in RATES:
        value = row[key]
        if not (value > 0) or value != value or value == float("inf"):
            raise RuntimeError(f"{key} is not a finite positive rate ({value})")


def _format(gfx: str, dtype: torch.dtype, row: dict, label: str = "cell") -> str:
    activation = "bf16" if dtype == torch.bfloat16 else "fp16"
    line = (f"{label} gfx={gfx} activation={activation} prompt={row['prompt']} generated={row['generated']} "
            f"concurrency={row['concurrency']} prefill_tok_s={row['prefill_tok_s']:.9g} "
            f"decode_tok_s={row['decode_tok_s']:.9g} ttft_s={row['ttft_s']:.9g} itil_s={row['itil_s']:.9g} "
            f"peak_gib={row['peak_gib']:.4g} prefill_tokens={row['prefill_tokens']} "
            f"decode_tokens={row['decode_tokens']}")
    if "ttft_each" in row:
        line += (f" ttft_each={','.join(f'{v:.9g}' for v in row['ttft_each'])}"
                 f" itil_each={','.join(f'{v:.9g}' for v in row['itil_each'])}"
                 f" generated_each={','.join(str(v) for v in row['generated_each'])}")
    return line


def measure(path: str | Path, cells=CELLS, runs: int = 3) -> list[str]:
    """Load the checkpoint and print each run and one median line per cell."""

    from tensorfold.rocm.build import gfx_name

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RuntimeError("no HIP device is visible")
    if runs < 1:
        raise ValueError("runs must be 1 or more")
    gfx = gfx_name()
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
        rows = []
        for run in range(runs):
            gc.collect()
            torch.cuda.empty_cache()
            print(f"# start prompt={prompt_len} generated={generated} concurrency={concurrency} run={run + 1}/{runs}",
                  file=sys.stderr, flush=True)
            row = measure_cell(engine, prompt_len, generated, concurrency)
            _finite_positive(row)
            if (row["prompt"], row["generated"], row["concurrency"]) != (prompt_len, generated, concurrency):
                raise RuntimeError("cell lengths do not match the requested load")
            print(_format(gfx, dtype, row), flush=True)
            rows.append(row)
        line = _format(gfx, dtype, median_row(rows), "median")
        print(line, flush=True)
        lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    runs = 3
    if "--runs" in args:
        at = args.index("--runs")
        runs = int(args[at + 1])
        del args[at:at + 2]
    cells = CELLS
    if len(args) == 4:
        cells = ((int(args[1]), int(args[2]), int(args[3])),)
    elif len(args) != 1:
        print("usage: python -m tensorfold.rocm.bench MODEL_DIR [PROMPT GENERATED CONCURRENCY] [--runs N]",
              file=sys.stderr)
        return 2
    measure(args[0], cells, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
