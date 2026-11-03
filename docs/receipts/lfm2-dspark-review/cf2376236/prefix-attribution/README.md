# Remaining losses: target GPU elapsed and admission-cohort mismatch

**Diagnostic evidence, not performance qualification or a pass.** The completed production qualification remains [12/32 M4 and 16/32 M5 strict losses](../qualification/README.md). These probes do not replace either verdict.

## Source and execution

Production source under diagnosis is `cf23762368785b470b3bf7485026e2ea4837ce82`. Instrumented source is `5bed4bd92c7c257f963046b945d83fc931dcbf29`; CF differs only by deleting an obsolete test, not by runtime policy or target kernels. The complete diagnostic-only [patch](instrumentation.patch) is SHA256 `a6ec7fc13dfca077ba2ccf74e79cc7b7bda08fd56d6f9fd6aee51b4917a74151`; [controller](run-prefix-probe.py) SHA256 is `0c3a4e17b2ddc30e4198c9fda3d680e947cd210bf1987100bb527b910321c5d7`.

| Device | Instrumented native binary SHA256 | Completed phases | Clean server exits |
| --- | --- | ---: | ---: |
| M4 Pro | `aad6f3383981dd43edff2e6ea923c47c3e66f5eb00435d9f0b36ecbef57c0798` | 138 | 3/3, exit 0, non-forced |
| M5 Ultra | `2aebc26f18ba592576a026e57f19b8f77ca6b71815d99a6686837d6a74307814` | 138 | 3/3, exit 0, non-forced |

Each device ran plain, BF16 head weights, then 8-bit head weights: the original code/T=0 scenario at widths 1/4/8/16, followed by the original code/T=1 warmup and all 16 seed oracles, then width-1 warmup and measurement. Each arm has 46 synchronous phases. Context 4096, parallel 16, 512-token requests, original frozen driver/model hashes, memory-pressure guard and persistent model lock remain in force. Both head modes use BF16 activations. Original token-oracle comparisons and cross-arm measured token hashes pass.

The instrumentation reads existing command-buffer GPU durations after existing waits; it adds no GPU synchronization or GPU timestamp query. It snapshots cumulative counters only after all active requests finish. Admission price is additionally evaluated on the CPU for diagnostic display. The patch is not production code.

## M4: neural computation does not explain these losses

All 30 measured requests in **each** loaded-head mode decline context. In the following cases, round/draft/accept counts and physical target row-count histograms also equal plain, with zero reservation-span calls. The entire BF16 measured set has matching row histograms; 8-bit's wider cases have copy-work differences preserved in the raw record.

| Diagnostic case | Plain target GPU ms | BF16 target GPU ms | 8-bit target GPU ms | Round / draft / accept counts in all three arms |
| --- | ---: | ---: | ---: | --- |
| code, T=0, width 1 | 3011.355123 | 3124.243414 | 3144.874038 | 272 / 310 / 239 |
| code, T=1, width 1 | 4312.433500 | 4484.559128 | 4566.335293 | 504 / 66 / 7 |

These are **whole-RPC target GPU elapsed sums**, including prompt work, not decode-only throughput, hardware occupancy, or memory bandwidth. Raw `decode`/`prefill` counter labels mean recorded/unrecorded forwards; unrecorded can also include queued decode. They are summed here. Equal histograms are not a claim of an identical ordered command trace.

For sampled width 1, all planning spans together are only 0.300/0.350/0.431 ms in plain/BF16/8-bit; the target GPU elapsed differences are much larger. A host-planner or neural-head explanation alone is insufficient. **The cause of the GPU elapsed difference is not established.** Inactive head resources remain loaded, device clocks and other GPU activity were not recorded in these captures, and this diagnostic has one fixed arm order. Do not call it thermal noise, occupancy loss, or a proven residency problem.

## M5: a solo admission can survive into a shared cohort

In the measured greedy width-4 case, both head modes have context ownership `[true, false, false, false]`. The first request was admitted with `streams=1, heads=1` and a positive price; the other three declined. This is not an unprofitable-price recovery admission. For the complete 8-bit first-request quote: plain cost 3.759 ms; quoted cost 8.139735 ms; quoted progress 2.619512 tokens; selected depth 4. Its admission does not know the eventual four-request decode cohort.

| Diagnostic width-4 arm | Target GPU ms | Round / draft / accept counts | Context-tracked requests |
| --- | ---: | --- | --- |
| Plain | 2984.685003 | 1660 / 786 / 384 | none |
| BF16 | 3030.518293 | 1662 / 775 / 382 | first of four |
| 8-bit | 3031.388414 | 1662 / 775 / 382 | first of four |

Both copy work and context ownership differ, so the extra GPU time cannot be assigned solely to tap capture. The actionable design limitation is the **temporal mismatch between request-stable admission and the eventual decode cohort**, not a width-4 hard cutoff. A fix must preserve cache/adopt/cancel lifecycle and cheap-copy progress; it must not disable user speculation to remove neural context.

Source: [registration and admission before prefill](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/core/lanes/drafted.zig#L170-L230), [positive-price versus bounded recovery transition](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/families/lfm2_moe/dspark_admission.zig#L141-L164).

## Actual GPU routes and specific optimisation candidates

- **The exactness-gated twin is enabled on both devices in all three arms.** Model loading actually assigns the result of its per-shape parity probes; the kernels' initial default is not the final route. Source: [model load and gate](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/families/lfm2_moe/model.zig#L27-L84). Thus unconditional 32-row fallback padding is not the explanation for these captures.
- Recorded target verification is bounded to 32 rows and selects non-prompt kernels. Prompt/NAX eligibility is a different path; do not enable an unsupported arithmetic route to chase a score. Source: [verification](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/families/lfm2_moe/backend.zig#L338-L371), [prompt and capture selection](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/families/lfm2_moe/forward.zig#L190-L217).
- The active twin branch bypasses `projectAll`'s compatible-input precomputed-sum reuse. **Candidate:** share exact ordered input sums across compatible Q/K/V and gate/up projections, without changing reduction order or BF16 boundaries. The existing non-twin sum route is not automatically interchangeable. Source: [projection dispatch](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/families/lfm2_moe/forward.zig#L85-L116).
- Routed experts use indexed slot calls below the current 16-row grouping crossover. **Candidate:** measure by-expert grouping at smaller physical row counts and actual expert-overlap patterns, including grouping overhead and exact output equality, before changing that crossover. More reuse is possible when rows pick the same expert; its timing benefit is not yet measured here. Gate/up SwiGLU is already fused, so “add missing expert GLU fusion” is not a valid diagnosis. Source: [expert dispatch](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/families/lfm2_moe/forward.zig#L421-L428), [grouping and indexed calls](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/core/tile_qmm.zig#L125-L146), [existing kernels](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/zig/src/core/tile_qmm.zig#L204-L267).

These are source-backed candidates, not measured speedups or claims of saturated bandwidth. No matched MLX-LM run was performed; these receipts cannot quantify an MLX gap or prove its cause.

## Raw evidence and limits

[m4-observed.json](m4-observed.json) and [m5-observed.json](m5-observed.json) are byte-exact parsed-controller receipts, not rewritten qualification data. [derived.json](derived.json) includes all 15 measured cases per device, actual admissions, row/activity equality checks, timings, and incomplete-price records. The original private server logs are not published.

One diagnostic price record per device has missing or tainted fields from log interleaving. Those records are preserved and explicitly listed; missing fields are not repaired, inferred, or used as numeric price evidence. This does not change the controller's complete per-phase GPU counters, actual enrollment decisions, token comparisons, or clean-exit observations. The M5 8-bit price cited above is complete.

The unchanged full qualification has balanced rotations and no diagnostic instrumentation. Only [that packet](../qualification/README.md) determines the current performance verdict, which remains **FAIL on both devices**.
