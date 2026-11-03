# What the remaining slowdown is doing

**Diagnosis, not a performance pass.** Production source `2acdb56a26d71361412136f5f36bf3bfdd12d563` plus diagnostic-only patch `41fd643ed0ba78fddf752257ec8b35f4323cd145bb9a5c7dbc49df19c930dc40`. The complete unchanged CF matrices still fail 12/32 M4 and 16/32 M5 strict comparisons; their [original full packet](../../cf2376236/qualification/README.md) is not relabelled as newer-source evidence.

## Proven copy-pricing cause, with no neural contexts active

In the actual M4 code/T0/four-stream phase, every neural context is declined. At physical allocation 31, position 45, all four streams offer the same 15 token IDs and the same joint-probability bits in all three arms. Startup copy settings are also identical. The plain target-price curve selects three drafts per stream; the BF16 curve selects six. This is cheap-copy work, not hidden neural work.

The [captured lossless inputs](copy-price-divergence.json) and [executed real-allocator replay](allocator-replay.zig) isolate the cause:

| Exact replay variant | Drafts per stream |
|---|---|
| Plain prices, original overhead | 3, 3, 3, 3 |
| BF16 prices, original overhead | 6, 6, 6, 6 |
| Plain prices, BF16 fixed overhead | 3, 3, 3, 3 |
| BF16 prices, zero fixed overhead | 6, 6, 6, 6 |
| 8-bit prices, BF16 fixed overhead | 3, 3, 3, 3 |

**The curve alone changes the allocator decision.** BF16's measured fixed overhead is 0.24972280811416847 ms; the head residual is zero. The relevant four-stream overhead has already been measured, so an unknown-cost fallback is not involved. This does not prove the older M5 all-declined observation has the same cause.

The actual counter-off M4 phase shows the consequence, not just a replay:

| Arm | Logical rounds | Proposed drafts | Accepted drafts | Physical target calls | Target GPU ms |
|---|---:|---:|---:|---:|---:|
| Plain | 1284 | 836 | 760 | 321 | 7029.534412 |
| BF16 | 1176 | 1020 | 868 | 294 | 7470.156667 |
| 8-bit | 1284 | 836 | 760 | 321 | 7007.462378 |

BF16 buys more accepted drafts and fewer target calls, but its wider physical batches cost more GPU time in total. The raw bit-preserving startup prices and measured row-shape totals are in [M4 copy summary](m4-copy-summary.json).

The [M5 capture](m5-copy-summary.json) has a different limit: the first four-stream request starts before peers arrive and remains neural-enrolled; subsequent peers decline. It does **not** reproduce the older all-four-declined M5 case. The queued-cohort fix cannot predict future requests.

## Actual target GPU operator classes

Each sample uses GPU counter timestamps calibrated with actual CPU/GPU clock pairs. Startup calibration is excluded. The per-forward command-buffer stopwatch becomes absolute/invalid when encoders are instrumented; it is explicitly excluded from these calculations. All selected counter samples satisfy the checked calibration envelope.

The table is the plain arm's measured code/T0 phases; physical row shapes with fewer than ten calls are omitted here but retained in [all 124 groups](gpu-class-composition.json). Percentages share the sum of measured operator-class GPU durations, **not wall time or occupancy**.

| Device | Streams | Physical rows | Calls | Expert gate/up | Expert down | Conv projection | Vocab head |
|---|---:|---:|---:|---:|---:|---:|---:|
| M4 | 1 | 1 | 251 | 31.29% | 16.80% | 19.13% | 9.97% |
| M4 | 1 | 16 | 20 | 44.67% | 24.89% | 12.54% | 5.59% |
| M4 | 4 | 4 | 251 | 35.81% | 23.00% | 16.79% | 7.95% |
| M4 | 4 | 16 | 66 | 41.18% | 26.28% | 13.58% | 6.06% |
| M5 | 1 | 1 | 389 | 24.15% | 14.74% | 18.17% | 6.12% |
| M5 | 1 | 16 | 21 | 40.50% | 24.20% | 13.22% | 4.37% |
| M5 | 4 | 4 | 377 | 29.69% | 18.32% | 17.42% | 4.63% |
| M5 | 4 | 30 | 37 | 38.75% | 24.58% | 14.82% | 4.46% |

MoE expert gate/up and down projections dominate the larger physical target batches. This identifies the largest measured operator work; it does not establish whether those kernels are compute-bound or bandwidth-bound, their SIMD occupancy, or a matched MLX speed difference. No such claim is supported by this capture.

Instrumentation rotates and fences serial encoders and materially perturbs scheduling, cost feedback and throughput. It is diagnostic only. GPU counters are never presented as a new qualification result. Counter-off/on outputs match exactly in [138 original phases per device](paired-target-tokens.json): seed, generated-token count and SHA256 all agree. All twelve servers return zero; no qualification is rerun for luck.

## Source correction being tested separately

The existing LFM2 startup prices were measured before the serving residency set and neural head buffers were enrolled. Calibration caches also did not follow the admitted serving residency. A separate local change moves measurement after normal serving residency and commits the temporary calibration buffers before timing, using the existing two-stage `init`/`measure` convention. The first candidate `5d9469fe5` passed original kernel/reference guards on both devices but failed actual native startup with `BadLfmTapDestination`; it was not pushed. Corrected candidate `79837c4e2` prepares tap destinations after calibration and uses the existing idle-flush contract. At this packet's freeze, its original serving workflows are still running. Neither candidate is claimed to fix all regressions or pass performance. Target math, dtype, admission limits, timing trial count, drivers and oracles are unchanged.

## Reproduce the causal CPU replay

Check out source `2acdb56a26d71361412136f5f36bf3bfdd12d563`. Point `-Mroot` at the downloaded replay and `-Mallocation` at that checkout's real module:

```sh
zig run -lc --dep allocation \
  -Mroot=/path/to/downloaded/allocator-replay.zig \
  -Mallocation=/path/to/TensorFold-at-2acdb56a2/zig/src/core/lanes/allocate.zig
```

The [observed five replay results](allocator-replay-observed.json) are saved separately. The included [diagnostic-only patch](diagnostic-only.patch), [original-history controller](diagnostic-controller.py) and [offline parser](analyze-diagnostic.py) bind the private measurements. They are not production instrumentation or a new performance oracle.

Activations remain BF16 in both head-weight modes. The FP32 comparison is the same-checkpoint head reference on the same BF16 target taps, not an FP32 target server. See [exact-source M5 correctness](../m5-original-contracts/README.md) for the unchanged original contracts and the honestly retained 8-bit 8/9 pick result.
