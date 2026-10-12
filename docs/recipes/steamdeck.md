# Qwen3.5 on a Steam Deck (gfx1033), fork branch `steamdeck`

The HIP engine serves Qwen3.5-9B MLX 4-bit with its MTP head on an LCD Steam Deck (Van Gogh APU, gfx1033, RDNA2).

## Build (on any Linux host with HIP)

```sh
zig build native-hip tf-qwen35-test -Dhip-arch=gfx1033 -Dhipcc=<rocm>/bin/hipcc -Doptimize=ReleaseFast --prefix out1033
zig build grammar -Dxgrammar=<xgrammar v0.2.8 checkout> --prefix out1033   # structured output
```

Copy `out1033` and the HIP runtime libraries (`libamdhip64`, `libhsa-runtime64`, `libamd_comgr`, `librocprofiler-register`, `librocm_kpack`, `rocm_sysdeps`, `llvm/lib/libLLVM`, `llvm/lib/libclang-cpp`) into the Deck's home directory; nothing goes to the read-only root.

## GPU memory

HIP on the Deck allocates from the GTT, not the BIOS carve-out (raising the carve-out lowers what the engine counts).
Raise the GTT from 8 to 12 GiB with `ttm.pages_limit=3145728 ttm.page_pool_size=3145728 amdgpu.gttsize=12288` in `GRUB_CMDLINE_LINUX_DEFAULT` (`/etc/default/grub`), then `sudo update-grub` and a reboot.

## Run

```sh
export LD_LIBRARY_PATH=$HOME/tensorfold/rocm/lib:$HOME/tensorfold/rocm/lib/llvm/lib:$HOME/tensorfold/out1033/lib
tensorfold-native serve <model> --parallel 1 --context 8192 --prompt-cache-gib 0.5
TENSORFOLD_MEMORY_RESERVE_GIB=2.5 tf-qwen35-test check <model>   # 14 streams need the smaller reserve
```

Graphs stay on, the MTP head's too. On an integrated GPU the engine sets `HSA_ENABLE_SDMA=0` itself before HIP
loads (SDMA host-to-device copies stall on gfx1033: the first or second never retires); an exported value wins.

## What `steamdeck-opt` tunes for the Deck

- The stream tile (decode products) wants 96 waves a compute unit, not 3000 a GPU: on 8 CUs it carries 4 or 8 columns
  a lane, and its 4-row verify products run 4 columns where they ran 2.
- One-row launches (and two-row launches at 8 columns on a small GPU) stage the block's rows of x in LDS once instead
  of every lane loading and summing them again: products stream at 74-78 GB/s instead of 66-72.
- An integrated GPU drafts one MTP token by default (`--mtp-drafts` still sets 1 to 3): a 4-row verify costs about
  1.7 plain rounds on the Deck, a 2-row one about 1.1.
- Prompts of 25 to 64 rows run the GEMM tile on 32 or 64 rows a block.

Every change keeps every output bit: the kernel tests compare the staged and unstaged stream tile and pin its hash
to the previous tile's, the rows test compares every prompt tile byte for byte, and decode and prefill logits of the
9B (and of the 35B-A3B on a W7800) are byte-identical to the `steamdeck` branch's.

## What `steamdeck-opt2` adds

- Prompts of 9 to 48 and 65 to 88 rows on a small RDNA2 GPU (16 compute units or fewer) run the column tile
  (`tiles/gemm_cols.hpp`): a lane a column through all of K with the GEMM tile's arithmetic, its words loaded on long
  runs and turned in LDS. Every other prompt tile on the Deck loads 16 bytes a column a lane, which streams at 1-6 GB/s.
- The MTP head copies its scalars inside its graph, so it replays graphs on integrated GPUs too (a copy queued before
  the capture began was not done by the first launch on gfx1033). The head's time on the Deck is its projections, not
  its launches: a draft costs about 6.6 ms eager or replayed.
- An integrated GPU copies by blit kernels without `HSA_ENABLE_SDMA=0`.

Same bits: the rows test compares the column tile with every prompt tile; prefill logits at 9 to 256 tokens and
decode logits are byte-identical to `steamdeck-opt`'s, and served greedy replies are the same text.

## Measured on `steamdeck-opt2` against `steamdeck-opt` (Qwen3.5-9B MLX 4-bit MTP, 12 GiB GTT, 15 W)

Served, greedy, each request started at or below 70 C, `steamdeck-opt2` with `HSA_ENABLE_SDMA` unset.

| prompt | drafts | TTFT before | TTFT after | decode before | decode after |
| --- | --- | --- | --- | --- | --- |
| 47 tokens | off | 0.87 s | 0.71 s | | |
| 66 tokens | off | 1.20 s | 1.04 s | | |
| 91 tokens | off | 1.44 s | 1.29 s | | |
| 246 tokens | off | 2.03 s | 2.02 s | | |
| 266 tokens | default (1) | 2.82 s | 2.67 s | 18.5 / 19.3 tok/s | 18.5 / 19.3 tok/s |
| 266 tokens | off | 2.84 s | 2.67 s | 14.7 tok/s | 14.7 tok/s |
| 2,046 tokens | default (1) | 19.2 s | 19.2 s | 17.8 / 14.6 tok/s | 17.8 / 15.3 tok/s |
| 2,046 tokens | off | 19.2 s | 19.2 s | 13.7 tok/s | 13.7 tok/s |

`tf-qwen35-test prefill` (engine alone): 15 tokens 0.383 -> 0.231 s, 30 tokens 0.521 -> 0.367 s, 47 tokens
0.621 -> 0.527 s; 64 tokens and more unchanged. Two streams: 21.4 -> 21.3 tok/s in all (four streams: 25.5 before,
not measured after: the Deck was needed elsewhere). The loader drops each part's host pages once uploaded, so an
APU holds the weights once (the 27B oQ2e loads whole on the Deck).

## Measured on `steamdeck-opt` against `steamdeck` (Qwen3.5-9B MLX 4-bit MTP, 12 GiB GTT, 15 W)

Greedy, 256 tokens, a word-list story prompt and short code and explain prompts, the two builds alternating; each request starts with the APU at or below 70 C, because minutes of
load reach about 90 C and the APU then drops to 200 MHz until it cools. `steamdeck` drafts 3, `steamdeck-opt` 1.

| prompt | drafts | TTFT | prefill | decode before | decode after | ITL after |
| --- | --- | --- | --- | --- | --- | --- |
| 266 tokens (story) | default | 2.8 s | 94 tok/s | 15.5 tok/s | 19.4 tok/s | 52 ms |
| 266 tokens (story) | off | 2.8 s | 94 tok/s | 13.4 tok/s | 14.7 tok/s | 68 ms |
| 2,046 tokens (story) | default | 19.3 s | 106 tok/s | 13.6 tok/s | 16.3 tok/s | 61 ms |
| 2,046 tokens (story) | off | 19.3 s | 106 tok/s | 12.5 tok/s | 13.7 tok/s | 73 ms |
| 53 tokens (code) | default | 0.95 s | 56 tok/s | 16.3 tok/s | 20.3 tok/s | 49 ms |
| 52 tokens (explain) | default | 0.95 s | 55 tok/s | 15.7 tok/s | 20.1 tok/s | 50 ms |

`tf-qwen35-test check` on `steamdeck-opt`: 46 passed, 0 failed, 1 skipped (radix memory: too few ids).
Short prompts (drafts off): 47 tokens 0.91 s -> 0.87 s TTFT, 58 tokens 0.95 s -> 0.89 s, 91 tokens 1.72 s -> 1.44 s.
Four streams at once (`--parallel 4`, 256 tokens each, wall time from the first request): 23.0 -> 25.4 tok/s
in all without drafts, 22.4 -> 25.3 with the default drafts; two streams 18.3 -> 21.5 tok/s.

## Measured on `steamdeck` (Qwen3.5-9B MLX 4-bit MTP, 12 GiB GTT, 15 W)

| prompt | drafts | TTFT | prefill | decode | ITL |
| --- | --- | --- | --- | --- | --- |
| 270 tokens | on | 2.7 s | 99 tok/s | 14.3 tok/s | 70 ms |
| 270 tokens | off | 2.8 s | 96 tok/s | 13.3 tok/s | 75 ms |
| 1,883 tokens | on | 18.2 s | 104 tok/s | 14.8 tok/s | 68 ms |
| 1,883 tokens | off | 18.1 s | 104 tok/s | 12.8 tok/s | 78 ms |

`tf-qwen35-test check`: 46 passed, 0 failed (drafted == serial included). Structured output: JSON schema, JSON object, regex and choice valid, drafted == serial.
