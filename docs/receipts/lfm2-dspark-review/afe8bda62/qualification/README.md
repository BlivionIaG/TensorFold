# Complete qualification — source `afe8bda62562519b4fd84ee9a956392a21f26a60`

## M5 Ultra complete; M4 same-head result not included yet

**M5 strict maintainer verdict: FAIL.** 12/32 comparisons are below same-head plain (BF16 seven, 8-bit five), worst **-0.268625302766234%** in BF16 sampled-chat/one-stream. All 32 meet the task owner's less-than-1% regression criterion; that does **not** override the maintainer's explicit zero-tolerance rule. No rerun, pooling, reduced scope or loss waiver.

M5 completed **144/144 launch cells and306/306 token oracles**, with zero exactness/scenario failures and all nine servers exiting0 without forced kills. Three balanced rotations; plain/BF16/8-bit; code/chat; temperatures0/1; widths1/4/8/16;512-token replies; parallel16; context4096; prompt cache0. Plain cheap-copy speculation stays enabled. Activations remain BF16 in both head-weight modes.

This is the actual final pushed source, not relabelled implementation-source798 or historicalCF data. M4's separate same-head workflow is still running at this packet's first publication; no combined two-device clearance is claimed here.

## Source and unchanged contracts

- Source: `afe8bda62562519b4fd84ee9a956392a21f26a60`; pinned base `6a5048697e0770a464a4093ff6373f6a73f998b6`.
- M5 native binary: `21a7ddd36deae47cbc484ecb2018f010907fb4be16803cacab2318aa91f31c22` in every arm.
- Source archive: `e76f35347e9d57d27b04e7aa8665cb326bccf4b817e7f93767bd9f980427fb0e`.
- Target revision: `667408d494bde191c2cb9dd4f4549827816b9275`; drafter: `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`.
- All seven [frozen driver files](frozen-driver/) retain their original hashes. The unchanged verdict's `summarize(metadata, cells, events)` reproduces the saved summary **byte-for-byte**, including exit3 and every loss.
- Own-head original 82-step kernel/native/reference scope and both complete cache/fresh-cache/sample/cancellation workflows finish before qualification, on this source/archive/binary. Their original files are retained under [M5 original contracts](m5/original-contracts/observed.json).

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

## What changed, and what remains

The retained calibration fix removes the captured M4 cheap-copy work divergence; its [causal report and original proof](../../79837c4e2/serving-calibration/README.md) remain bound to runtime798 and its documentation-only AFE descendant. That diagnosis is not substituted for this complete matrix. Historical CF M5 had16/32 strict losses and worst−1.4341478593879908%; this run has12 and worst−0.2686253027662344%, without pooling the two datasets.

This packet does not attribute the remaining small losses to timing noise, wrapper overhead or GPU clocks without causal evidence. The separately observed same-work M4 q8 sampled-solo GPU difference remains unexplained. GitHub also still reports DIRTY/no checks at the pushed head; no re-review or merge readiness is claimed.
