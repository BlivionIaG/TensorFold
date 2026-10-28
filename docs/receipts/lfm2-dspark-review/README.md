# DSpark review fixes on current main

Source `14565fa73`, rebased onto main `d0c9dc9fb`, tested on 2026-10-28.
Apple M4 Pro, 20 GPU cores, 48 GiB RAM, macOS 26.5.1, Zig 0.17.0,
Apple Metal compiler 32023.883. Builds target `apple_m1`.

The shared and lone guards from the rejected PRs are absent. The lane core retains
confidence/grant pricing, hold-cost pricing and `copy_first`. Hold ordering reuses
one bounded allocation instead of allocating on every round. The initialization
cleanup releases Metal buffers before their owning list; injected scratch and warm
allocation failures exercise that path. Config admission refuses rank above 256 and
nonfinite f32 RoPE theta. The quantizer test widens its size calculation before
multiplication, and `--drafter` refuses a target restricted to one-row verification.

## Checks

- `zig build test test-kernel-contract test-golden fuzz-cortheon native dev-tools -Dcpu=apple_m1 --summary all`:
  749/749 steps succeeded; 8,414 tests passed, 46 skipped, 0 failed.
- `zig build -Dcpu=apple_m1 --summary all`: 239/239 steps succeeded.
- `zig build test-lfm2 test-lfm2-dspark -Dcpu=apple_m1 --summary all`:
  111/111 tests passed. The final scratch/warm failure test also passed in a subsequent
  focused run: 51/51 DSpark tests.
- `zig build test-lfm2-dspark -Dcpu=apple_m1 -Doptimize=debug --summary all`:
  51/51 passed, including both injected allocation failures.
- Debug `zig test` on `dspark_config.zig`: 4/4 passed; on `dspark_quant.zig`: 2/2 passed.
  The original quantizer test from `2e5966f68` aborted with integer overflow in Debug.
- `zig fmt --check zig` and `git diff --check`: passed.
- Lean: 428 findings on main, 428 on this source, 0 new.
- `tf-kernel-contract --only lfm2`: 11 kernels passed gates 1-4, 0 failures,
  0 known findings; all 17 planted controls red, 0 missed; 0 limit problems.
- `tf-kernel-contract --only tile_qmm`: 6 kernels passed gates 1-4, 0 failures,
  0 known findings; 0 limit problems.

## Served equality

Target `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` at
`667408d494bde191c2cb9dd4f4549827816b9275`, with the local released DSpark weights.
The drafter safetensors SHA-256 is recorded in the accompanying JSON.

For both default bf16 and explicit `--drafter-bits 8`, serve with `--parallel 16
--context 4096 --prompt-cache-gib 0 --keep-warm 0`. Run:

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:18194 review \
  --tokens 64 --reps 1 --temperatures 0 --levels 1,4,8,16 --alone --serial
python3 tools/bench_concurrent.py http://127.0.0.1:18194 review \
  --tokens 64 --reps 1 --temperatures 0 --levels 1,4,8,16 --alone --serial \
  --prompts tools/bench_prompts.json
```

Each precision mode passed all 12 cells: eight same-prompt cells (code and chat)
and four distinct-prompt cells. Concurrent replies equal solo replies and drafted
solo replies equal `"draft": false` token hashes. Zero unequal replies and zero
failed requests. [Equality evidence](equality-and-load.json) retains per-request
concurrent, solo and plain token hashes.

These are short correctness checks, with one repetition, while builds were also
running. Their timings do not establish the decode performance gate.

## Peak during load

One model process at a time under a model lock. Both servers used the same target
and bf16 drafter, `--parallel 1 --context 4096 --prompt-cache-gib 0 --keep-warm 0`.
Read `/memory` after readiness and before sending any request. Its lifetime peak
agreed with `/usr/bin/time -l` at process exit.

- Exact #1392 head `1bb6e5db2`: 16,036,097,352 bytes, 14.935 GiB.
- Corrected stack source `14565fa73`: 15,527,930,776 bytes, 14.462 GiB.

The measured reduction during load is 508,166,576 bytes (484.6 MiB). This compares
the original pre-allocation head with the corrected stack rebased onto current
main; it is not an isolated comparison of #1393's commit on the same base.

## Remaining qualification

No new long-context, cancellation/replacement stress, resumed-serving, sampled
serving or alternating speed qualification is claimed here. The maintainer's
requested current-main performance cells and other-family A/B remain to be run.
M1, M2, M3, M5, CUDA and HIP were not run for this revision. The focused Metal
contract checks above are not the full cross-family `kernel-contract` gate.
