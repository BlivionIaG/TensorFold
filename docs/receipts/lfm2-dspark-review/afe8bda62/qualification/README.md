# Complete qualification — source `afe8bda62562519b4fd84ee9a956392a21f26a60`

## Both devices complete; M4 timing inconclusive

The unchanged maintainer script fails both datasets. Negative point estimates are not counts of statistically confirmed regressions. Correctness retains its strict zero-failure requirement; the owner's 1% performance budget, measurement uncertainty and literal zero-tolerance verdict remain separate.

| Device | Exact cells / oracles | Clean nonforced exits | Negative median flags | Worst observed delta | Largest plain three-run range | Owner less-than-1% interpretation |
|---|---:|---:|---:|---:|---:|---|
| M4 Pro | 144/144 / 306/306 | 9 | 16/32 | -3.269046% | 32.256410% | Inconclusive: strongly nonstationary timing |
| M5 Ultra | 144/144 / 306/306 | 9 | 12/32 | -0.268625% | 1.388918% | All 32 observed medians inside budget; uncertainty not established |

Both devices finish the original matrix on the actual final pushed source: three balanced rotations; plain/BF16/8-bit; code/chat; temperatures0/1; widths1/4/8/16;512-token replies; parallel16; context4096; prompt cache0. Plain cheap-copy speculation stays enabled. Activations remain BF16 in both head-weight modes. No rerun, pooling, reduced scope or relabelled historical results.

The plain range is `(maximum/minimum-1)*100` within the three unchanged-baseline samples, not a confidence interval or measurement-error floor. There is no A/A control. No combined formal performance clearance is claimed.

## Source and unchanged contracts

- Source: `afe8bda62562519b4fd84ee9a956392a21f26a60`; pinned base `6a5048697e0770a464a4093ff6373f6a73f998b6`.
- M4 native binary: `c03ab9600d36c32dd1d80b7d94bf9dae15d7256fa8f41df7f23b488c19b43a2c` in every arm.
- M5 native binary: `21a7ddd36deae47cbc484ecb2018f010907fb4be16803cacab2318aa91f31c22` in every arm.
- Source archive: `e76f35347e9d57d27b04e7aa8665cb326bccf4b817e7f93767bd9f980427fb0e`.
- Target revision: `667408d494bde191c2cb9dd4f4549827816b9275`; drafter: `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`.
- All seven [frozen driver files](frozen-driver/) retain their original hashes. The unchanged verdict's `summarize(metadata, cells, events)` reproduces the saved summary **byte-for-byte**, including exit3 and every loss.
- M5 own-head original82-step kernel/native/reference scope and both complete cache/fresh-cache/sample/cancellation workflows finish before qualification; see [M5 original contracts](m5/original-contracts/observed.json). M4 completes its own-head82-step scope and full matrix; see [M4 original contracts](m4/original-contracts/observed.json). M4's complete cache/sample/cancellation proof remains explicitly bound to runtime798, whose only subsequent production changes are documentation; it is not relabelled as a fresh AFE functional run.

## All 32 M4 comparisons and timing diagnosis

[Complete M4 table, raw evidence and per-rotation baseline work](m4/README.md).
Fifteen of sixteen negative flags change sign across paired rotations. Unchanged plain sampled-chat/16-stream throughput falls from315.822245 to290.219123 to238.795417 TPS with nearly identical logical work. This dataset cannot reliably resolve the owner's1% margin; neither sixteen confirmed code regressions nor equivalence is established. The cause of the elapsed-time drift is not yet proven.

## All 32 M5 comparisons

Metric: median of three aggregate decode TPS samples. Delta is `(arm median / same-cell plain median - 1) *100`. Every unrounded negative fails; displayed precision is not a tolerance.

| Prompt | T | Streams | Plain TPS | BF16 TPS | BF16 Δ | 8-bit TPS | 8-bit Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 0 | 1 | 285.530245 | 288.099753 | +0.899908% | 289.731205 | +1.471284% |
| code | 0 | 4 | 637.145166 | 646.310512 | +1.438502% | 645.743087 | +1.349445% |
| code | 0 | 8 | 853.696218 | 854.012183 | +0.037011% | 854.177927 | +0.056426% |
| code | 0 | 16 | 872.721058 | 873.277697 | +0.063782% | 885.199160 | +1.429793% |
| code | 1 | 1 | 272.019362 | 271.850656 | -0.062020% | 274.034568 | +0.740832% |
| code | 1 | 4 | 552.790402 | 552.894747 | +0.018876% | 552.254107 | -0.097016% |
| code | 1 | 8 | 714.635116 | 713.670782 | -0.134941% | 714.131982 | -0.070404% |
| code | 1 | 16 | 878.199714 | 878.968193 | +0.087506% | 878.839574 | +0.072860% |
| chat | 0 | 1 | 265.705810 | 265.211010 | -0.186221% | 265.015326 | -0.259868% |
| chat | 0 | 4 | 625.121656 | 624.885141 | -0.037835% | 625.060667 | -0.009756% |
| chat | 0 | 8 | 821.886239 | 822.344013 | +0.055698% | 822.255556 | +0.044935% |
| chat | 0 | 16 | 810.655325 | 810.574682 | -0.009948% | 811.166279 | +0.063030% |
| chat | 1 | 1 | 259.584750 | 258.887440 | -0.268625% | 259.947567 | +0.139768% |
| chat | 1 | 4 | 562.096112 | 562.758022 | +0.117757% | 561.275467 | -0.145997% |
| chat | 1 | 8 | 720.892939 | 721.405246 | +0.071066% | 721.260549 | +0.050994% |
| chat | 1 | 16 | 885.174845 | 884.729867 | -0.050270% | 885.705362 | +0.059934% |

## Evidence and reproduction

[M5 metadata](m5/metadata.json), [all cells and oracles](m5/cells.jsonl), [events](m5/events.jsonl), [unchanged summary](m5/summary.json), [original controller output](m5/controller.log) and [source/binary-bound wrapper](m5/controller-observed.json) are retained. [Raw requests](m5/requests.jsonl.gz) are deterministic gzip with mtime0; [binding](m5/raw-request-binding.json) records original SHA256 and byte count, and decompression is checked byte-identical. `private-logs/` are explicitly excluded by the original driver's publication contract.

To replay the exact verdict without serving a model, put `frozen-driver` on `PYTHONPATH`, load `m5/metadata.json`, each JSON line of `m5/cells.jsonl` and `m5/events.jsonl`, call `bench_lfm2_dspark_verdict.summarize`, and serialize with `json.dumps(result, indent=1)+'\n'`. The [replay identity record](verdict-replay-check.json) binds the original and recomputed bytes. [SHA256 manifest](SHA256.json) covers every published file.

## M5 measurement variability is separate from a literal timing flag

The [existing paired observations](measurement-variation.json) show that six of the twelve negative
median comparisons change sign across the three rotations; six are negative in all three pairings.
For the worst median, BF16 sampled-chat/one-stream, the paired differences are -0.203%, -0.372%
and -0.132%. Its plain samples span 0.074% across these three runs. A small systematic cost is
plausible there; it is not valid to dismiss every negative as noise.

Conversely, a negative point estimate is not proof of a real regression. Three rotations establish
neither a reliable measurement-error floor nor a robust uncertainty bound, and this matrix has no
unchanged-binary plain-vs-plain A/A control. Sign changes are not proof of equivalence either.
The owner's 1% budget is an engineering acceptance margin, not an empirically measured noise floor.

Correctness keeps its strict zero-failure requirement. Performance reporting separates the observed
differences, uncertainty, owner budget and literal maintainer-policy result. No qualification is
rerun for a better sign, and tiny timing flags are not treated as confirmed bugs without causal or
control evidence.

## What changed, and what remains

The retained calibration fix removes the captured M4 cheap-copy work divergence; its [causal report and original proof](../../79837c4e2/serving-calibration/README.md) remain bound to runtime798 and its documentation-only AFE descendant. That diagnosis is not substituted for this complete matrix. Historical CF M5 had16/32 negative median comparisons and worst−1.4341478593879908%; this run has12 and worst−0.2686253027662344%, without pooling the two datasets or treating either count as a statistical regression count.

The separate [complete clock/environment diagnostic](../../79837c4e2/clock-environment/README.md), explicitly runtime798 plus private instrumentation, now covers q8 through clean exit and shows reference/history/time sensitivity: the same plain sampled-code request changes8.822183% in GPU elapsed between original occurrences with identical activity and physical-row histograms, reversing the apparent cross-mode sign. It establishes neither a universal noise floor nor a hardware cause. This qualification packet and its raw verdicts are unchanged; no later diagnostic is substituted for them. GitHub also still reports DIRTY/no checks at the pushed head; no re-review or merge readiness is claimed.
