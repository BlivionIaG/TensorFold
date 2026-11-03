# DSpark current-main re-review evidence

## Status

Code PR: [ashhart/TensorFold#1394](https://github.com/ashhart/TensorFold/pull/1394). This is the sole landing PR; #1391–#1393 are closed as superseded, not merged. Branch/feature history is preserved.

- **Measured code revision:** `750bca95c457dc2119505568dff593e151cbeed7`.
- **Integrated main snapshot:** `8e35d523b1d30c5188c830d3d6175c61afb35bcb` by normal merge.
- **CI:** all 11 checks passed on that exact PR head; see [the saved check records](ci.json).
- **M5 Ultra:** complete original qualification; exactness passed, strict performance failed.
- **M4 Pro:** complete same-head qualification; exactness passed, strict performance failed. Both device workflows are finished.

This report/receipt branch does not change the measured PR code head. No new kernel, pricing change, force push, merge or auto-merge is hidden in this update.

## Two distinct performance decisions

The [maintainer checklist](https://github.com/ashhart/TensorFold/pull/1394#issuecomment-6589493432) requires every drafter median to be at least the same-head plain median on both devices. The task owner subsequently requested that a regression **strictly less than 1% count as passing for review**. That is a supplementary review assessment, not a claim that the maintainer accepted a different merge requirement. Gains count as passing; decisions use unrounded values.

| Device / arm | Within owner's <1% regression limit | Outside that limit | Worst delta | Passes original >=plain rule |
|---|---:|---:|---:|---:|
| M4 Pro / BF16 weights | 9/16 | 7 | -4.949% | 7/16 |
| M4 Pro / 8-bit weights | 10/16 | 6 | -7.180% | 7/16 |
| M5 Ultra / BF16 weights | 16/16 | 0 | -0.652% | 2/16 |
| M5 Ultra / 8-bit weights | 12/16 | 4 | -2.398% | 6/16 |

**17 of 64 comparisons remain outside even the owner's 1% review limit:** 13 on M4 Pro and four on M5. They are visible in the complete tables; none is waived. This is published review evidence, **not an all-performance-passing or merge-qualified claim**. Requesting review with these known exceptions is a separate decision from satisfying the stated performance bar.

## Complete M4 Pro matrix

Binary SHA256: `1193de98ce1ec00e19a0bf4a5a11cd0a2059a17086efc5f6219dbaeecacf10e5`. Every arm used this same binary. All 144 launch cells completed; all 306 target-token oracle records passed. There were nine clean, non-forced shutdowns, with zero exactness/scenario failures. The unchanged full driver's exit code was **3: performance failure**, with 18/32 negative comparisons.

| Prompt | T | Streams | Plain TPS | BF16 TPS | BF16 Δ | 8-bit TPS | 8-bit Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0 | 1 | 162.516 | 154.473 | -4.949% | 150.847 | -7.180% |
| code | 0 | 4 | 279.177 | 268.020 | -3.996% | 269.646 | -3.414% |
| code | 0 | 8 | 300.448 | 297.307 | -1.045% | 298.830 | -0.539% |
| code | 0 | 16 | 333.154 | 330.538 | -0.785% | 328.296 | -1.458% |
| code | 1 | 1 | 105.376 | 108.827 | +3.275% | 108.623 | +3.081% |
| code | 1 | 4 | 180.422 | 185.270 | +2.687% | 186.809 | +3.540% |
| code | 1 | 8 | 225.277 | 222.961 | -1.028% | 223.857 | -0.631% |
| code | 1 | 16 | 262.802 | 269.091 | +2.393% | 271.281 | +3.226% |
| chat | 0 | 1 | 110.629 | 111.715 | +0.982% | 109.480 | -1.039% |
| chat | 0 | 4 | 228.389 | 229.407 | +0.446% | 232.288 | +1.707% |
| chat | 0 | 8 | 267.688 | 259.263 | -3.147% | 266.354 | -0.498% |
| chat | 0 | 16 | 330.061 | 314.971 | -4.572% | 321.618 | -2.558% |
| chat | 1 | 1 | 116.012 | 110.418 | -4.822% | 111.723 | -3.697% |
| chat | 1 | 4 | 196.973 | 197.975 | +0.508% | 200.620 | +1.851% |
| chat | 1 | 8 | 229.894 | 236.717 | +2.968% | 235.163 | +2.292% |
| chat | 1 | 16 | 284.554 | 284.325 | -0.080% | 298.456 | +4.886% |

All 32 M4 comparisons appear above. Per-rotation samples and paired deltas remain in [the original summary](m4-pro/summary.json); differing signs between rotations are not used to waive a losing median.

## Complete M5 matrix

Binary SHA256: `77515b6f0a1f69cb9d05ee633b90f55ecf0d87d3ab8449ee8072b850ba274580`. Every arm used this same binary. All 144 launch cells completed; all 306 target-token oracle records passed. There were nine clean, non-forced shutdowns, with zero exactness/scenario failures. The unchanged full driver's exit code was **3: performance failure**, with 24/32 negative comparisons.

| Prompt | T | Streams | Plain TPS | BF16 TPS | BF16 Δ | 8-bit TPS | 8-bit Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0 | 1 | 285.797 | 288.641 | +0.995% | 289.924 | +1.444% |
| code | 0 | 4 | 646.239 | 644.969 | -0.196% | 645.046 | -0.185% |
| code | 0 | 8 | 854.567 | 852.742 | -0.214% | 852.872 | -0.198% |
| code | 0 | 16 | 872.800 | 871.507 | -0.148% | 936.915 | +7.346% |
| code | 1 | 1 | 272.056 | 273.299 | +0.457% | 274.282 | +0.818% |
| code | 1 | 4 | 553.080 | 549.649 | -0.620% | 546.747 | -1.145% |
| code | 1 | 8 | 714.770 | 710.112 | -0.652% | 697.632 | -2.398% |
| code | 1 | 16 | 879.180 | 876.904 | -0.259% | 862.495 | -1.898% |
| chat | 0 | 1 | 265.238 | 264.454 | -0.296% | 266.562 | +0.499% |
| chat | 0 | 4 | 625.157 | 624.431 | -0.116% | 623.736 | -0.227% |
| chat | 0 | 8 | 822.106 | 821.660 | -0.054% | 820.601 | -0.183% |
| chat | 0 | 16 | 809.476 | 808.176 | -0.161% | 895.161 | +10.585% |
| chat | 1 | 1 | 259.816 | 258.341 | -0.568% | 260.530 | +0.275% |
| chat | 1 | 4 | 562.697 | 560.279 | -0.430% | 554.882 | -1.389% |
| chat | 1 | 8 | 721.262 | 720.111 | -0.160% | 720.346 | -0.127% |
| chat | 1 | 16 | 885.913 | 884.149 | -0.199% | 880.106 | -0.655% |

All 32 comparisons appear above: each row has BF16-versus-plain and 8-bit-versus-plain. TPS is the median over three balanced rotations; delta is relative to that row's plain median. Raw per-rotation samples and paired deltas remain in [summary.json](m5/summary.json).

## Original full-run configuration and provenance

- Code/chat; T=0 and T=1; widths 1/4/8/16; 512-token replies; three balanced arm rotations.
- Capacity 16, context 4096, prompt cache disabled, no snapshots; original memory-pressure/admission checks retained.
- Same-head plain retains cheap-copy speculation. Drafter flags alone select released BF16 or explicit 8-bit weights; **both use BF16 activations**.
- Target revision `667408d494bde191c2cb9dd4f4549827816b9275`; drafter revision `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`.
- The frozen seven-file driver and model-file hashes are identical across devices. Compiler, binary and original receipt hashes are in [M4 bindings](m4-pro/bindings.json) and [M5 bindings](m5/bindings.json); M5 also records the source-archive hash. No numerical bound or original verdict code was changed.
- Both [M4 Pro](m4-pro/) and [M5](m5/) preserve metadata, cells, events, summary and losslessly compressed original requests. Bindings include original uncompressed and compressed-file SHA256 hashes. Private server/controller logs are not published.
- Smoke proof: the unchanged frozen verdict module reproduced each complete original summary from its published metadata/cells/events, with JSON replica-key normalization only. Decompressing each requests archive reproduced its original SHA256.

## Exercised correctness and the FP32 comparison limit

| Device | Canonical scoped build | BF16 weights / FP32 picks | 8-bit weights / FP32 picks |
|---|---|---|---|
| M4 Pro | 79/79 steps; 655 passed, 6 skipped | 9/9; worst confidence difference 0.00085 | 9/9; worst confidence difference 0.00152 |
| M5 Ultra | 79/79 steps; 657 passed, 4 skipped | 9/9; worst confidence difference 0.00080 | 8/9; worst confidence difference 0.00225 |

The actual native fixture used 63 prior-context tokens and retained nine proposals. M5's 8-bit differing pick has FP32 margin 0.075. The diagnostic's [final FP32 comparison](https://github.com/BlivionIaG/TensorFold/blob/750bca95c457dc2119505568dff593e151cbeed7/zig/tests/lfm2_dspark.zig#L548-L586) prints results but does **not** enforce final pick/confidence agreement; exit 0 is not substituted for the printed figures.

Original attention, BF16 and norm/tap kernel contracts passed on both devices. Native controls exercised retained-prefix 1/3/9 byte/pick/confidence invariance, 8/16 logical cohorts, deferred/rewound context, ownership/release and planted invalid/nonfinite failures. These checks are distinct from both approximate drafter/reference agreement and full served target-token exactness.

## What the code and current receipts establish

1. **The plain baseline already makes speculative progress.** In M5 rotation 0, greedy code at width 4, all three arms have exactly the same mixed-source counts: 1,660 per-request rounds, 786 proposals, 384 accepted tokens. That does not prove zero neural holds, but it does show no additional net progress in those counters. Disabling cheap copies would compare against the wrong baseline.
2. **Loaded-drafter overhead remains when no hold pays.** Target tap capture stays enabled; five BF16 taps at width 2048 store 20,480 bytes per captured row. Reservation, copy queries and hold economics also run before final funding. Staging normally owns taps in place and defers projection; simply discarding zero-grant context would break later recovery. See [attachment](https://github.com/BlivionIaG/TensorFold/blob/750bca95c457dc2119505568dff593e151cbeed7/zig/src/native/lfm2_moe.zig#L115-L131), [preparation](https://github.com/BlivionIaG/TensorFold/blob/750bca95c457dc2119505568dff593e151cbeed7/zig/src/core/lanes/drafted.zig#L297-L356) and [owned staging](https://github.com/BlivionIaG/TensorFold/blob/750bca95c457dc2119505568dff593e151cbeed7/zig/src/families/lfm2_moe/dspark_stage.zig#L38-L106).
3. **Physical batching is narrower than request concurrency.** Capacity 32 fits only three full nine-row query blocks. Larger actually held cohorts split, and all their passes are synchronous; one physical pass is asynchronous. A larger capacity might help a funded cohort, but does not explain the two single-stream BF16 chat losses, or prove that a width-16 request cohort actually held 16 neural blocks. See [holdFn](https://github.com/BlivionIaG/TensorFold/blob/750bca95c457dc2119505568dff593e151cbeed7/zig/src/families/lfm2_moe/dspark.zig#L362-L397).
4. **BF16/8-bit differ in body/context projection and Markov W2, not activation dtype or tied vocabulary weights.** No extra BF16-only host wait was found. BF16 narrow rows, tensor thresholds and weight representation are different execution designs; stored-byte ratios do not establish bandwidth saturation. See [BF16 dispatch](https://github.com/BlivionIaG/TensorFold/blob/750bca95c457dc2119505568dff593e151cbeed7/zig/src/families/lfm2_moe/dspark_bf16.zig#L10-L35) and [body/head execution](https://github.com/BlivionIaG/TensorFold/blob/750bca95c457dc2119505568dff593e151cbeed7/zig/src/families/lfm2_moe/dspark_block.zig#L428-L459).

### Unproven causes, not claimed fixes

- Reservation and the subsequent window both query cheap-copy proposals against unchanged history; the proposer decrements silence per ask. This is a concrete source-level redundancy candidate, not a measured explanation of a given loss. Borrowed proposal slices and backoff semantics prevent blindly caching a pointer.
- Rejected/recovery decisions repeatedly scan calibrated clean/context price tables and can repeat shape searches. Per-decision quote reuse is a bounded candidate; changing recovery cadence or caching mutable costs across rounds is not justified by the current receipts.
- The BF16 row kernel declares 16 accumulator slots for nine live rows; compiler dead-code elimination, register pressure and physical occupancy have not been measured. A nine-row specialization is not yet a demonstrated speedup.
- Pointwise work after the last bidirectional attention still covers unretained rows. Removing it without preserving physical routing/reduction regimes can break retained-prefix invariance. The straightforward tail-shortening and rejected target-NAX shortcuts are not shipped.

The remaining causal discriminator is executed hold/context-projection exposure versus copy/quote CPU work and tap-enabled target GPU work on the actual losing cells. These source findings do **not** establish the cause of an MLX-LM gap; a matched MLX-LM execution breakdown is still absent. The [older diagnosis](../diagnosis.md) retains its historical phase timings and rejected experiments rather than relabelling them as this source.
