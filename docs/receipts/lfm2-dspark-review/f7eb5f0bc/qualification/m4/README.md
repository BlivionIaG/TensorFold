# Complete original M4 Pro qualification — runtime `f7eb5f0bc`

**Performance FAIL under both the unchanged original zero-tolerance rule and the owner’s observed<1% loss budget.** Only **2/32** observed comparisons fit the owner budget; **30/32** lose at least1%. Worst observed loss is **−8.714872783%**,8-bit code/T0/1 stream. Every original cell is retained. This is not a performance pass and is not labelled measurement noise.

The original matrix stage finished before the separately prespecified M4 A/A study. Publishing this complete stage does not inspect partial controlled timings,change that collection,select a favorable run,or replace either frozen verdict.

## Exact source and complete scope

Runtime source `f7eb5f0bcf7c373a50d95a55acdee4e75cd75fce`, native SHA256 `a7be9659f48c7cdb6a19a63a013bc850c679343a9e18e5dfa81ef220ff702fc0`. Same production native executable in plain,BF16 and8-bit arms; no private instrumentation. Archive SHA256 `3ca340ec39493b7d32ba23902e2087111e54def1184c1665331b4ada1cbe7b12`. Docs-only published descendant `51dce745143d72dd52231c44a31989632107af4a` has identical native/build source objects,not a relabelled qualification.

**144/144 cells;306/306 exact target-token oracles;0 exactness/scenario failures;9 clean non-forced exits.** Original full scope:code/chat,T0/T1,1/4/8/16 streams,512 tokens,three balanced rotations,parallel16,context4096,cache0,plain cheap copies enabled. Original seven-file driver unchanged; no scope-reducing option was supplied.

## Every original performance comparison

Each arm’s three launch samples are summarized by the original median TPS and compared with same-head plain. Unrounded values decide the original rule. Every one of32 original medians is below plain; original **FAIL/exit3** is retained.

| Cell | Plain TPS | BF16 TPS | BF16 Δ% | 8-bit TPS | 8-bit Δ% |
|---|---:|---:|---:|---:|---:|
| code T0 w1 | 169.961074 | 161.852827 | -4.770649 | 155.149182 | -8.714873 |
| code T0 w4 | 291.580707 | 276.868927 | -5.045526 | 271.420913 | -6.913967 |
| code T0 w8 | 314.887024 | 308.001278 | -2.186736 | 300.032562 | -4.717394 |
| code T0 w16 | 357.390907 | 343.363754 | -3.924877 | 333.688222 | -6.632145 |
| code T1 w1 | 113.245500 | 112.730987 | -0.454334 | 107.237099 | -5.305642 |
| code T1 w4 | 203.164117 | 192.274769 | -5.359878 | 189.249228 | -6.849088 |
| code T1 w8 | 244.125666 | 233.276217 | -4.444207 | 227.830478 | -6.674918 |
| code T1 w16 | 299.888138 | 283.488076 | -5.468726 | 278.142757 | -7.251164 |
| chat T0 w1 | 119.602463 | 116.485991 | -2.605692 | 110.917098 | -7.261862 |
| chat T0 w4 | 247.654180 | 241.827045 | -2.352932 | 230.582485 | -6.893361 |
| chat T0 w8 | 284.469291 | 273.305870 | -3.924298 | 265.123940 | -6.800506 |
| chat T0 w16 | 337.797012 | 328.929023 | -2.625242 | 320.887780 | -5.005738 |
| chat T1 w1 | 112.177864 | 110.959096 | -1.086460 | 110.016886 | -1.926385 |
| chat T1 w4 | 202.835955 | 201.305830 | -0.754366 | 200.598806 | -1.102935 |
| chat T1 w8 | 244.378893 | 241.572649 | -1.148317 | 236.695378 | -3.144099 |
| chat T1 w16 | 297.841304 | 294.727700 | -1.045390 | 292.288126 | -1.864476 |

Largest observed plain three-sample range `(max/min−1)*100`:**10.412325067%**,chat/T1/4 streams. **20/32** per-repetition comparisons change sign. These observations quantify spread in this dataset,not a general measurement-error floor or an exemption for the8.7% loss. The prespecified [controlled method](../../controlled-study/README.md) has its own mandatory A/A validity and uncertainty checks; no completed controlled verdict is asserted here.

## Actual work counters and limits

All three arms’ aggregate round/draft/accept counters match in26/48 paired rep/cells. In22/48 they differ. For code/T0/1 stream in rep0:

| Arm | Rounds | Drafted | Accepted |
|---|---:|---:|---:|
| plain |272|310|239|
| BF16 |272|310|239|
| 8-bit |260|327|251|

The8-bit arm accepts more tokens and takes fewer target rounds there,yet the complete comparison is slower. Aggregate activity includes **all proposal sources** and is not proof that the neural head proposed those tokens or that padded GPU work was identical. There is no current-run operator timing or occupancy attribution in this uninstrumented qualification. `work-counter-analysis.json` retains all three worst-cell repetitions,totals and these limitations. Startup head weights are625.1MiB BF16 /361.4MiB8-bit; plain loads no DSpark head.

## Safety and precision

Original native/GPU/model guards:82/82 steps;767 tests passed,8 skipped. Kernel contract16/16 kernels and24/24 planted controls,zero missed controls/limit problems. Both head-weight modes retain **BF16 activations** and match9/9 FP32-reference picks; worst confidence differences0.00085 BF16 /0.00152 8-bit,largest differing-pick margin0. This is the head FP32 reference on BF16 target features,not an FP32 target server. Serving tokens independently match every original target oracle.

## Original data and replay

`metadata.json`,`cells.jsonl`,`events.jsonl` and`summary.json` are byte-for-byte original. `requests.jsonl.gz` is deterministic lossless compression of the entire original raw request file;`inputs.json` records original decoded size/SHA and compressed SHA. Private native logs remain retained locally,not published with personal paths.

The original unchanged CLI `summarize` ran only on copied saved inputs—**no model launch**—and returned3. The replayed summary is byte-identical to the original SHA256 `37cdd6ac3e439b5f63504e6884ca3244ce7eb7d76c23c776e5dade8dc2157d3b`. `verdict-replay-proof.json` records the proof. No original qualification file or verdict was modified.

The M5 result is separate:[all32 observed comparisons inside the owner budget](../m5/README.md),with its own raw zero-tolerance FAIL preserved. A pass on M5 does not clear this M4 result. Current integration/CI status is [source-labelled here](../../review-readiness/README.md).

## Subsequently completed controlled study

The separate [prespecified M4 A/A study](../../controlled-results/m4/README.md) subsequently completed in full and returned **INCONCLUSIVE** under its published criteria. It does not replace this original matrix, change its medians or clear its performance failure. The completed report retains every candidate bound, A/A observation, warmup request and parsed clock record.
