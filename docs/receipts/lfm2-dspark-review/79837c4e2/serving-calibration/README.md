# Calibration context corrected; captured copy-work divergence removed

**Runtime implementation source `79837c4e205a18b5eb7236ea393e51dca1264fe2`**, published in documentation/receipt-only descendant `afe8bda62562519b4fd84ee9a956392a21f26a60`. Diagnostics add the identical private instrumentation patch `41fd643ed0ba78fddf752257ec8b35f4323cd145bb9a5c7dbc49df19c930dc40` used for the [preceding causal report](../../2acdb56a2/copy-gpu-attribution/README.md). Production arithmetic and BF16 activations are unchanged. This packet contains **no full performance qualification verdict**; that is a separate unchanged matrix at the final pushed head.

## Problem and decision

Target startup prices previously preceded serving residency and loaded-head buffer enrollment. Temporary calibration caches did not follow the normal serving owner. The actual allocator consequently made different cheap-copy decisions on identical offers, despite all four neural contexts being declined.

The retained change separates `Metal.init` from `measure`, measures after admission/residency, and commits pending head/cache buffers through the existing idle-flush contract before timing. Temporary caches unregister before the owner leaves. Tap destinations are prepared afterward: the first candidate `5d9469fe5` passed original kernel/reference guards but actual serving failed with `BadLfmTapDestination`, retained in [the original startup log](rejected-5d9469fe5-startup.log). It was never a PR tip. No exception is suppressed; corrected source798 passes the actual original serving workflow.

No target math, dtype, calibration fixture, warmup, timing-trial count, admission ceiling, driver, numerical oracle or performance tolerance changes. Normal serving residency can still refuse buffers under its existing bounds. Plain and loaded modes use the same corrected calibration procedure; no baseline-only slowdown, cutoff or benchmark branch is introduced.

## Decisive same-input allocator check

Using the previously captured four identical offers, fifteen exact probability bits and a common measured fixed overhead of 0.24972280811416847 ms, the real allocator now selects three drafts per stream under every corrected M4 curve:

| M4 price curve | Before: source2ac | After: source798 |
|---|---|---|
| Plain | 3, 3, 3, 3 | 3, 3, 3, 3 |
| BF16 | 6, 6, 6, 6 | 3, 3, 3, 3 |
| 8-bit | 3, 3, 3, 3 | 3, 3, 3, 3 |

[Executed M4 replay](m4-allocator-replay.zig); [observed replay outputs](allocator-replays-observed.json). Only the captured price arrays change. In the new capture, M4's 16-row startup prices are 43.085 / 43.211 / 43.197 ms for plain/BF16/8-bit; the old BF16 quote was 46.542 ms. Exact lossless bit arrays are in the source-bound summaries.

The M5 curves are close, but not identical: their maximum observed across-arm spread is 0.9355787222667655% at one row. On these *same saved M4 offers*, the M5 counterfactual replay still chooses `[7,6,6,6]` versus BF16 `[7,7,6,6]`. These are not new M5 request observations. Small quote differences can still change an integer near-tie; this is not a guarantee of identical work or strict performance clearance.

## Actual serving work, not only a replay

M4's measured code/T0/four-stream phase now has the same logical work and physical target shape counts in all three modes, with all neural contexts declined:

| Source | Arm | Logical rounds | Proposed | Accepted | Target calls | Target GPU ms |
|---|---|---:|---:|---:|---:|---:|
| 2ac | plain | 1284 | 836 | 760 | 321 | 7029.534412 |
| 2ac | bf16 | 1176 | 1020 | 868 | 294 | 7470.156667 |
| 2ac | 8-bit | 1284 | 836 | 760 | 321 | 7007.462378 |
| 798 | plain | 1284 | 836 | 760 | 321 | 6808.844548 |
| 798 | bf16 | 1284 | 836 | 760 | 321 | 6842.530581 |
| 798 | bits8 | 1284 | 836 | 760 | 321 | 6795.946624 |

The correction removes the observed wider-batch BF16 work divergence. GPU elapsed values are retained as diagnostic observations, not a before/after performance gate: different runs can have different clocks or other timing state. The [M4](m4-copy-summary.json) and [M5](m5-copy-summary.json) summaries include all five measured phases per mode, startup prices, exact physical row histograms and source/native/driver/model bindings.

All three diagnostic servers on each device return zero cleanly. Each mode retains 46 original phases. [M4](m4-target-token-comparison.json) and [M5](m5-target-token-comparison.json) compare source2ac to798: **138 phase pairs per device, zero generated-token hash/count/seed mismatches**. The same-input allocator does not replace these actual HTTP observations.

## Original native correctness, both devices

The unmodified original six-command scope and cache/fresh-cache/sample/cancellation workflows pass in both weight modes. These are source798 artifacts, not relabelled as final-head AFE results.

| Device | Original build steps | Fresh test summary | Stress completed | Cancellations | Wrong/errors | Server exits |
|---|---:|---|---:|---:|---:|---|
| M4 Pro | 82/82 | 355/362; 7 skipped; shared-allocation test cached | 818 | 468 | 0 | 4 clean, non-forced, 0 |
| M5 Ultra | 82/82 | 767/771; 4 skipped | 2788 | 1514 | 0 | 4 clean, non-forced, 0 |

Both head-weight modes retain BF16 activations. FP32 reference consumes the same BF16 target taps. M4 matches 9/9 reference picks in both modes; M5 matches BF16 9/9 and 8-bit 8/9, with 0.075 largest margin for the different 8-bit pick. Confidence maxima remain M4 0.00085/0.00152 and M5 0.00080/0.00225. The 8-bit result is not universal equality or an FP32 target-server comparison.

[Combined source/binary-bound correctness](original-contracts.json) contains exact commands, sample hashes, cached-prompt counts and shutdowns. [M4 raw build log](original-contracts/m4/build.log) and [M5 raw build log](original-contracts/m5/build.log), plus original operator/native/functional JSON files, are preserved byte-for-byte. No failing check is retried for luck or narrowed.

## Remaining timing uncertainty

Work equivalence is not a strict performance pass. In the corrected M4 sampled-solo phase, all modes report 504 rounds, 66 proposed copies, seven accepted and no neural context. Their physical row histograms agree, but q8 target GPU elapsed is still 4271.962293 ms versus plain 4071.150951 ms. This packet does not attribute that residual to clocks, occupancy, bandwidth or hidden neural work. Existing partial clock evidence did not cover q8. The full balanced matrix remains required; no loss is waived.

The preceding operator-class capture identifies expert projections as the largest wider-batch work, but serial encoder fences are intrusive and command-buffer stopwatches invalid under that instrumentation are excluded. No matched MLX-LM comparison or SIMD/bandwidth saturation claim is established.

## Reproduce the CPU consumer check

Use source798 (or its code-identical AFE descendant), point `-Mroot` at the downloaded M4/M5 replay, and run:

```sh
zig run -lc --dep allocation \
  -Mroot=/path/to/m4-allocator-replay.zig \
  -Mallocation=/path/to/TensorFold/zig/src/core/lanes/allocate.zig
```

The [runtime-only source delta](runtime-from-2ac.patch), [capture provenance](capture-bindings.json), exact input bits and [SHA256 manifest](SHA256.json) bind this packet. The [diagnostic-only instrumentation](../../2acdb56a2/copy-gpu-attribution/diagnostic-only.patch) remains outside production. Neither this report nor its replay is a new oracle or re-review request.
