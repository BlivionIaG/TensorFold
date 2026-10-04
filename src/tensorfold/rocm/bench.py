"""RDNA serving cells: ``python -m tensorfold.rocm.bench MODEL_DIR [PROMPT GENERATED CONCURRENCY]``."""

from __future__ import annotations

import gc
import math
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
        if not (value > 0 and math.isfinite(value)):
            raise RuntimeError(f"{key} is not a finite positive rate ({value})")


def _format(gfx: str, dtype: torch.dtype, row: dict, label: str = "cell", tp: int = 1) -> str:
    activation = "bf16" if dtype == torch.bfloat16 else "fp16"
    line = (f"{label} gfx={gfx} activation={activation} tp={tp} prompt={row['prompt']} generated={row['generated']} "
            f"concurrency={row['concurrency']} prefill_tok_s={row['prefill_tok_s']:.9g} "
            f"decode_tok_s={row['decode_tok_s']:.9g} ttft_s={row['ttft_s']:.9g} itil_s={row['itil_s']:.9g} "
            f"peak_gib={row['peak_gib']:.4g} prefill_tokens={row['prefill_tokens']} "
            f"decode_tokens={row['decode_tokens']}")
    if "ttft_each" in row:
        line += (f" ttft_each={','.join(f'{v:.9g}' for v in row['ttft_each'])}"
                 f" itil_each={','.join(f'{v:.9g}' for v in row['itil_each'])}"
                 f" generated_each={','.join(str(v) for v in row['generated_each'])}")
    return line


def measure(path: str | Path, cells=CELLS, runs: int = 3, *, tp: int = 1, rank: int = 0, master: str = "",
            master_port: int = 29551) -> list[str]:
    """Load the checkpoint and print each run and one median line per cell (rank 0 prints)."""

    from tensorfold.rocm.build import gfx_name

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RuntimeError("no HIP device is visible")
    if runs < 1:
        raise ValueError("runs must be 1 or more")

    def say(text: str, **kwargs) -> None:
        if rank == 0:
            print(text, flush=True, **kwargs)

    rccl = None
    if tp > 1:
        from tensorfold.rocm.comm import RCCL

        torch.cuda.set_device(rank % torch.cuda.device_count())
        rccl = RCCL(rank, tp, master, master_port)
        rccl.ready("startup")
    gfx = gfx_name()
    dtype = activation_dtype(gfx)
    model = load(path)
    if rccl is not None:
        from tensorfold.rocm.qwen import slice_for_tp

        slice_for_tp(model, rank, tp)
    engine = Engine(model, schedule="auto", dtype=dtype, rccl=rccl)
    probe = model.layers[0]
    packed = probe.qkv if isinstance(probe, LinearLayer) else probe.q
    sample = torch.zeros(1, model.spec.hidden, device=packed.words.device, dtype=dtype)
    got = engine.linear(sample, packed)
    if got.shape != (1, packed.words.shape[0]) or not torch.isfinite(got).all():
        raise RuntimeError("the packed projection did not return a finite row")
    say(f"loaded gfx={gfx} activation={'bf16' if dtype == torch.bfloat16 else 'fp16'} tp={tp} "
        f"layers={model.spec.n_layers} hidden={model.spec.hidden} vocab={model.spec.vocab} "
        f"bits={model.spec.bits} group={model.spec.group} "
        f"embed_words={model.embed.words.shape[0]}x{model.embed.words.shape[1]} dtype=int32 "
        f"rank_weights_gib={torch.cuda.memory_allocated() / 1024**3:.2f}")
    # One generate per prompt length compiles every kernel a cell launches before any cell is timed.
    from tensorfold.rocm.qwen_math import SPAN

    for length in sorted({8, *(min(prompt_len, SPAN) for prompt_len, _, _ in cells)}):
        engine.generate(_prompts(length, 2, 1, model.spec.vocab), 2)
    torch.cuda.synchronize()
    say("# warmup done", file=sys.stderr)
    lines = []
    for prompt_len, generated, concurrency in cells:
        rows = []
        for run in range(runs):
            gc.collect()
            torch.cuda.empty_cache()
            say(f"# start prompt={prompt_len} generated={generated} concurrency={concurrency} run={run + 1}/{runs}",
                file=sys.stderr)
            row = measure_cell(engine, prompt_len, generated, concurrency)
            _finite_positive(row)
            if (row["prompt"], row["generated"], row["concurrency"]) != (prompt_len, generated, concurrency):
                raise RuntimeError("cell lengths do not match the requested load")
            say(_format(gfx, dtype, row, tp=tp))
            rows.append(row)
        line = _format(gfx, dtype, median_row(rows), "median", tp=tp)
        say(line)
        lines.append(line)
    return lines


def measure_served(path: str | Path, prompt_len: int, generated: int, runs: int = 3, *, mtp: int = 0, tp: int = 1,
                   rank: int = 0, master: str = "", master_port: int = 29551) -> str | None:
    """One request at a time through the engine ``tensorfold serve`` runs; rank 0 prints, the others follow."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.rocm.build import gfx_name
    from tensorfold.rocm.engine import QwenEngine

    engine = QwenEngine.load(path, keep=0, tp=tp, rank=rank, master=master or "127.0.0.1",
                             master_port=master_port, mtp_depth=max(1, mtp), no_drafts=mtp == 0)
    if rank != 0:
        engine.follow()
        return None
    if mtp and engine.mtp is None:
        engine.close()
        raise ValueError(f"--mtp {mtp}: this checkpoint has no MTP head")
    gfx, dtype = gfx_name(), engine._dtype()
    prompt = _prompts(prompt_len, generated, 1, engine.model.spec.vocab)[0]
    greedy = Sampling(seed=1, temperature=0.0)
    # The full prompt once: every prefill tile (the Triton attention tile included) compiles before a timed run.
    engine.generate(prompt, 4, greedy, lambda tokens: None, stop_eos=False, draft=False)
    rows = []
    for run in range(runs):
        stamps = []

        def seen(tokens, stamps=stamps):
            torch.cuda.synchronize()
            stamps.extend([time.perf_counter()] * len(tokens))

        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.generate(prompt, generated, greedy, seen, stop_eos=False, draft=False)
        if len(stamps) != generated:
            raise RuntimeError(f"generated {len(stamps)} tokens, requested {generated}")
        rows.append({"prefill_tok_s": prompt_len / (stamps[0] - start),
                     "decode_tok_s": (generated - 1) / (stamps[-1] - stamps[0])})
        print(f"served gfx={gfx} tp={tp} mtp={mtp} prompt={prompt_len} generated={generated} "
              f"prefill_tok_s={rows[-1]['prefill_tok_s']:.6g} decode_tok_s={rows[-1]['decode_tok_s']:.6g}",
              flush=True)
    engine.close()
    middle = {key: statistics.median(row[key] for row in rows) for key in rows[0]}
    line = (f"median-served gfx={gfx} activation={'bf16' if dtype == torch.bfloat16 else 'fp16'} tp={tp} mtp={mtp} "
            f"prompt={prompt_len} generated={generated} prefill_tok_s={middle['prefill_tok_s']:.6g} "
            f"decode_tok_s={middle['decode_tok_s']:.6g} peak_gib={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(line, flush=True)
    return line


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    options = {"--runs": 3, "--tp": 1, "--rank": 0, "--master": "", "--master-port": 29551, "--mtp": -1}
    for flag, default in options.items():
        if flag in args:
            at = args.index(flag)
            options[flag] = type(default)(args[at + 1])
            del args[at:at + 2]
    served = "--served" in args or options["--mtp"] >= 0
    if "--served" in args:
        args.remove("--served")
    cells = CELLS
    if len(args) == 4:
        cells = ((int(args[1]), int(args[2]), int(args[3])),)
    elif len(args) != 1:
        print("usage: python -m tensorfold.rocm.bench MODEL_DIR [PROMPT GENERATED CONCURRENCY] [--runs N] "
              "[--tp N --rank R --master ADDR [--master-port P]] [--served [--mtp N]]", file=sys.stderr)
        return 2
    if served:
        prompt_len, generated, concurrency = cells[0] if len(args) == 4 else (1024, 512, 1)
        if concurrency != 1:
            print("--served measures one request at a time, as the server runs them", file=sys.stderr)
            return 2
        measure_served(args[0], prompt_len, generated, options["--runs"], mtp=max(0, options["--mtp"]),
                       tp=options["--tp"], rank=options["--rank"], master=options["--master"],
                       master_port=options["--master-port"])
        return 0
    measure(args[0], cells, options["--runs"], tp=options["--tp"], rank=options["--rank"],
            master=options["--master"], master_port=options["--master-port"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
