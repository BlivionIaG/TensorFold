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
- `tf-lfm2-dspark MODEL DRAFTER 256`: 9/9 drafts equal the independent fp32
  reference's picks; maximum confidence difference 0.00106; 20 steady drafter
  rounds had zero host allocations and resizes; exit 0.
- Completed `tf-lfm2-exact`: 72/72 checks on both M4 Pro and M5 Ultra, zero
  failures or skipped checks. This supersedes the earlier interrupted run.

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

## Extended qualification on the published stack

Source `2dbbc0e8f`, main baseline `d0c9dc9fb`. Additional device: Apple M5 Ultra,
256 GiB RAM, macOS 27.0, Metal compiler 32023.921.6, Zig 0.17.0.
[Extended evidence](extended-qualification.json) retains request hashes, counters,
per-round throughput and latency. Generated text and token arrays are omitted.
[Reproduction harness](qualification.py) supports `functional`, `long`,
`performance`, `performance qwen`, and `cold`; set `TF_QUAL_MAIN`,
`TF_QUAL_CANDIDATE`, `TF_QUAL_MODEL`, `TF_QUAL_DRAFTER`, and `TF_QUAL_OUT`
for the local installations. Run one model process at a time.

Both devices passed real two-turn resumed-versus-fresh chat equality (3,517 cached
prompt tokens), 29 seeded sampled equality cases across 1/4/8/16 streams, and
cancellation/replacement stress with Metal API validation enabled. M4 completed
143 requests with 84 cancellations at six slots and 252 with 134 at sixteen;
M5 completed 453 with 247 and 852 with 459 respectively. Zero wrong replies or
request errors. Server shutdowns were graceful.

M5 plain/drafted serving hashes agree at 4,096, 16,384, 65,536 and 130,816 prompt
tokens. Independent fp32 drafter checks agree on all nine picks at histories
65,535 and 130,815; maximum confidence differences 0.00532 and 0.00351.
Twenty steady rounds had zero host allocations/resizes. M5 also passes the same
11 LFM2 and six tile_qmm contracts, with all 17 LFM2 planted controls red.

### Performance remains an open issue

Three alternating rounds per arm, 256 generated tokens, 1/4/8/16 streams,
32-token warmups and uncached prompts. Aggregate decode rate uses the span from
the earliest first token to the latest last token. The `chat` fixture label in
the JSON means prose submitted to raw completions, not a templated chat request.
All cross-arm output hashes agree. These results do **not** clear a no-loss gate.

On M5, drafted versus main-plain median decode changes for code at 1/4/8/16
streams are +19.44%, -0.96%, -2.20%, -4.23%; for prose they are +35.89%,
-0.72%, -1.25%, -1.17%. Candidate-plain closely follows main-plain.
On M4, changes are approximately -6.0%, -13.9%, -5.6%, +0.7% for code and
-6.4%, -13.4%, -7.8%, -8.0% for prose. M4 also shows variation between plain
arms, so these measurements do not isolate every source of the loss.

M5 cold prefill medians at 2k/8k/32k/64k tokens are 0.1527/0.6027/2.9244/7.3276
seconds on main-plain versus 0.1634/0.6327/3.0466/7.5652 with the drafter.
Prompt counts were checked and cached-token counts were zero.

Other-family Qwen3.8-27B plus DFlash2 was tested at one and four streams with
three alternating rounds on both devices. M4 results vary with each executable's
independently measured persistent cost record: fresh records give code changes
of -7.13%/+3.71% and prose -6.47%/+0.83%. The original records and fresh-record
runs are both retained. No timings were copied between executables. M5 changes are +0.20%/-0.47% for code and +0.05%/+0.70% for prose.
No blanket other-family performance pass is claimed.

Two isolated, uncommitted experiments (a larger absorb staging buffer and
removing forced per-stream copy pricing) did not consistently remove the loss.
Neither experiment is part of this PR. Further pricing work requires its own
validation; the rejected shared/lone guards have not been restored.

M1, M2, M3, CUDA and HIP were not rerun for this revision. Focused Metal checks
are not a full cross-family kernel-contract receipt. All nine GitHub checks passed on `2dbbc0e8f` before this receipt-only update.
