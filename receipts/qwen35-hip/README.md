# Qwen3.5 / 3.6 HIP engine: serving receipts (#631, #655)

Qwen3.6-35B-A3B MLX 4-bit with its MTP head (`Qwen3.6-35B-A3B-MLX-4bit-MTP`, Hugging Face revision `f84b054c5677b6e59bdb969ae61915b0983d68bb`), served by `tensorfold-native` built from:

- tp 1: `qwen35-rccl` at `10926cf3` (#655's head, which contains #631's head `79ce7747`);
- tp 2 and tp 4: `66d8a19b`, the same plus the window fix (every rank sizes its streams for rank 0's window).

Environment: Fedora Linux 7.1.7, HIP 7.14.60850, RCCL from the same ROCm, Zig 0.17.0, `-Doptimize=ReleaseFast`, an EPYC 7K62 host. Radeon PRO W7800 (gfx1100, 48 GB) and Radeon PRO V620 (gfx1030, 32 GB). One set of our model processes ran at a time; the host's other GPUs ran an unrelated workload, and every memory figure below counts our server processes only.

## How it was measured

- **Cold prefill:** `tools/bench_cache.py --prompt-file prompts/prefill_N.json --expected-prompt-tokens N --require-cold --tokens 16 --repeats 1 --streams 1`, server `--prompt-cache-gib 0 --parallel 1 --context 0` (the model's 262,144-token window). Each size runs three times cold (first, repeated, concurrent); every row reports 0 cached tokens and exactly N prompt tokens (`qualified: true`).
- **Token arrays:** `prompts/prefill_N.json` are exact token ids made by `mkprompts.py` from the repo's public docs (`docs/**/*.md`, README, CONTRIBUTING, LICENSE) with the checkpoint's own tokenizer; 97,971 tokens a pass, cycled for 131,072 and 261,888. `prompts/prompts_source.json` lists the files. The request bodies are `bench_cache.py`'s: a streamed `/v1/completions` with the array as `prompt`, `max_tokens` 16, temperature 0, seed 1234.
- **Decode and concurrency:** CONTRIBUTING's commands, `bench_openai.py --tokens 256 --reps 3 --temperatures 0` and `bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`, server `--prompt-cache-gib 0 --parallel 4` (32,768-token window).
- **Memory:** `memwatch.sh` samples every second, for exactly the server's PIDs (rank 0 and its followers): device memory from `rocm-smi --showpids`, host RSS from `/proc/PID/status`, and the kernel's high-water mark (`VmHWM`), which also catches the load.
- Scripts: `run.sh` / `run_fix.sh` (one configuration), `all.sh` / `fix_all.sh` (the sequence), `memwatch.sh`, `summarize.py` (these tables). Raw JSON is in each configuration's directory; `*.log` are the server logs.

Cold prefill, time to first token (median of 3 cold runs, 0 cached tokens each) and prefill tok/s:

| Prompt tokens | W7800 tp1 | 2 × W7800 tp2 | V620 tp1 | 2 × V620 tp2 | 4 × V620 tp4 |
| --- | --- | --- | --- | --- | --- |
| 2,048 | 0.64 s, 3,190 tok/s | 0.44 s, 4,663 tok/s | 0.91 s, 2,258 tok/s | 0.58 s, 3,542 tok/s | 0.50 s, 4,075 tok/s |
| 8,192 | 2.81 s, 2,914 tok/s | 1.88 s, 4,350 tok/s | 3.92 s, 2,091 tok/s | 2.46 s, 3,323 tok/s | 2.08 s, 3,940 tok/s |
| 32,768 | 16.07 s, 2,039 tok/s | 10.09 s, 3,247 tok/s | 20.98 s, 1,562 tok/s | 12.50 s, 2,621 tok/s | 9.60 s, 3,415 tok/s |
| 65,536 | 45.04 s, 1,455 tok/s | 26.85 s, 2,441 tok/s | 56.03 s, 1,170 tok/s | 32.14 s, 2,039 tok/s | 22.67 s, 2,891 tok/s |
| 131,072 | 141.86 s, 924 tok/s | 79.73 s, 1,644 tok/s | 166.89 s, 785 tok/s | 91.79 s, 1,428 tok/s | 59.04 s, 2,220 tok/s |
| 261,888 | 489.29 s, 535 tok/s | 266.79 s, 982 tok/s | 551.12 s, 475 tok/s | 292.93 s, 894 tok/s | 172.75 s, 1,516 tok/s |

Decode (`bench_openai.py --tokens 256 --reps 3 --temperatures 0`), median tok/s and time to first token:

| Prompt | W7800 tp1 | 2 × W7800 tp2 | V620 tp1 | 2 × V620 tp2 | 4 × V620 tp4 |
| --- | --- | --- | --- | --- | --- |
| fibonacci-raw | 176.5 tok/s, 36 ms | 181.8 tok/s, 30 ms | 157.3 tok/s, 41 ms | 149.1 tok/s, 35 ms | 153.8 tok/s, 32 ms |
| gpu-chat-no-think | 140.8 tok/s, 55 ms | 143.6 tok/s, 41 ms | 127.9 tok/s, 63 ms | 127.0 tok/s, 47 ms | 129.5 tok/s, 42 ms |

Concurrency (`bench_concurrent.py --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial`), total tok/s; equal to solo / to `"draft": false`:

| Prompt, streams | W7800 tp1 | 2 × W7800 tp2 | V620 tp1 | 2 × V620 tp2 | 4 × V620 tp4 |
| --- | --- | --- | --- | --- | --- |
| code, 1 | 174.5 tok/s; 3/3, 1/1 | 179.2 tok/s; 3/3, 1/1 | 156.9 tok/s; 3/3, 1/1 | 149.3 tok/s; 3/3, 1/1 | 153.6 tok/s; 3/3, 1/1 |
| code, 4 | 283.7 tok/s; 12/12, 1/1 | 372.6 tok/s; 12/12, 1/1 | 257.1 tok/s; 12/12, 1/1 | 341.3 tok/s; 12/12, 1/1 | 406.8 tok/s; 12/12, 1/1 |
| chat, 1 | 138.0 tok/s; 3/3, 1/1 | 142.3 tok/s; 3/3, 1/1 | 127.6 tok/s; 3/3, 1/1 | 127.0 tok/s; 3/3, 1/1 | 129.4 tok/s; 3/3, 1/1 |
| chat, 4 | 252.0 tok/s; 12/12, 1/1 | 310.5 tok/s; 12/12, 1/1 | 239.9 tok/s; 12/12, 1/1 | 285.3 tok/s; 12/12, 1/1 | 333.4 tok/s; 12/12, 1/1 |

Memory of the server processes only (every rank; sampled each second plus the kernel's high-water mark):

| | W7800 tp1 | 2 × W7800 tp2 | V620 tp1 | 2 × V620 tp2 | 4 × V620 tp4 |
| --- | --- | --- | --- | --- | --- |
| Device memory, peak (native window, 1 stream) | 26.2 GiB | 27.9 GiB | 26.1 GiB | 27.7 GiB | 35.6 GiB |
| Host RSS while serving, peak (native window, 1 stream) | 0.50 GiB | 1.99 GiB | 0.47 GiB | 1.83 GiB | 3.38 GiB |
| Host RSS high-water mark incl. load (native window, 1 stream) | 19.5 GiB | 39.0 GiB | 19.5 GiB | 38.8 GiB | 76.2 GiB |
| Device memory, peak (32K window, 4 streams) | 23.8 GiB | 25.7 GiB | 23.8 GiB | 25.6 GiB | 31.3 GiB |
| Host RSS while serving, peak (32K window, 4 streams) | 0.48 GiB | 1.99 GiB | 0.45 GiB | 1.81 GiB | 3.38 GiB |
| Host RSS high-water mark incl. load (32K window, 4 streams) | 19.5 GiB | 39.0 GiB | 19.5 GiB | 38.8 GiB | 76.2 GiB |

- **Host RSS high-water mark** is the mapped checkpoint during the load: every rank maps all of it while it reads its share, so it grows with the ranks (about 19.5 GiB each). Those are clean file pages the kernel can reclaim; while serving, a rank holds about 0.5 GiB alone and about 1 GiB under tensor parallelism (RCCL and the link).
- **The window fix:** before it, a follower started without `--context` sized its streams for the 32,768-token default. The first prompt past that (65,536) stopped rank 1 with `Invalid` on both GPU types; `w7800-tp2`, `v620-tp2` and `v620-tp4` hold those runs (64K and above not qualified), and `rerun-fast` the repeat. The `*-fix` directories are the same configurations on `66d8a19b`.
