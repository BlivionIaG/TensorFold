# DSpark request enrollment and copy ownership — source `cf2376236`

**Both full device matrices are complete; both strict speed gates failed. Not review-ready.** Original native/GPU checks and correctly source-bound M5 functional serving workflows pass. The maintainer requires every cell >= same-head plain, with no tolerance. All historical failures remain preserved.

| Device | Complete cells / token oracles | Strict speed losses | Worst delta |
| --- | --- | ---: | ---: |
| M4 Pro | 144/144 / 306/306; zero exactness or scenario failures | 12/32 | -5.322375682894121% |
| M5 Ultra | 144/144 / 306/306; zero exactness or scenario failures | 16/32 | -1.4341478593879908% |

**[All 64 current-source comparisons, unchanged verdicts and raw receipts](qualification/README.md).** Both original controllers exited 3; all nine servers per device exited 0 without forced kills. Complete and exact is not speed-qualified.

This is an **evidence-only branch**, not the qualified code branch. The pushed production source is [`cf23762368785b470b3bf7485026e2ea4837ce82`](https://github.com/BlivionIaG/TensorFold/commit/cf23762368785b470b3bf7485026e2ea4837ce82), with pinned main `6a5048697e0770a464a4093ff6373f6a73f998b6` integrated by normal merge `0b2e9336e2b77dad06652048a25f283e16100974`. Documentation commits here must never be substituted for the production source revision.

## What was wrong and what changed

1. **Cheap-copy lookup was not a harmless peek.** Shared reservation and next-window planning could both ask the suffix proposer before a new committed window. Its silence/backoff advances per query; asking twice could reawaken a copy and replace an already funded neural proposal. Reservation now owns one decision, including an empty answer and its trust/recall metadata. Buffer lifetime, history/cache position, room, proposer identity and lifecycle invalidations are preserved. The failing-before/passing-after consumer proof is linked below.
2. **Rejecting a neural hold did not remove its context cost.** Loaded DSpark requests continued owning/capturing target taps even when no neural work was funded. Request-stable enrollment now prices complete context debt and legal physical head/cohort shapes before prefill. Declined requests keep cheap copies and take the no-neural-tap target route. Already tracked requests with a temporary zero grant retain their context and bounded recovery; that contract is not silently removed.
3. **Native target calibration is intentionally sparse.** Missing candidate widths were being treated as an unknown whole quote, making the lone request fall back to tracking. The policy now skips only unpriced candidates, exactly as the existing lone depth rule does. Entirely unpriced calibration stays unknown. The real `Config.init` regression failed before and passes after.
4. **Other feature providers require their original prefill capability.** Enrollment is provider opt-in. Providers without the hook keep their original feature-prefill lifecycle, even when `drafts:false` disables token proposals. A real feature-only backend failed before with `FeatureStreamRequired` and passes after. Its regression verifies complete greedy/sampled target output, no drafted tokens and released resources. The obsolete test pinning zero context callbacks was deleted, not re-pinned.

There is one admission registry, bounded provider/precision-local conditional history, and separate greedy/sampled acceptance evidence. No hard-coded benchmark cell, stream-count cutoff, target arithmetic change, activation shortcut, or frozen-verdict change was added.

[Source-bound before/after receipts](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/docs/receipts/lfm2-dspark-review/enrollment-diagnosis.json) and [written implementation diagnosis](https://github.com/BlivionIaG/TensorFold/blob/cf23762368785b470b3bf7485026e2ea4837ce82/docs/receipts/lfm2-dspark-review/diagnosis.md).

## Actual remaining bottleneck: target GPU work, not reservation churn

The **M4 Pro** diagnostic below uses source `5bed4bd92c7c257f963046b945d83fc931dcbf29` with a private instrumentation patch, before the later test-only deletion. Its binary is `4ece120959114a1fddc1006fe9e0823daf78d5473d2962c437a6c1dd241c8799`. Instrumentation changes no math, scheduling policy, kernels or synchronization; it reuses existing completed command-buffer timing and records host elapsed spans. It is not a production throughput qualification. No equivalent M5 GPU-time breakdown was measured.

For code,T=1,width8, shares of active decode-step wall time were:

| Non-exclusive span | Plain | BF16 weights | 8-bit weights |
|---|---:|---:|---:|
| Target command-buffer elapsed | 96.392619% | 96.289897% | 96.304168% |
| Host sampling span | 2.022240% | 2.078071% | 2.040523% |
| Host encode span | 0.344797% | 0.356507% | 0.351919% |
| Reservation span | 0.000000% | 0.000412% | 0.000326% |

Most active-step wall time is inside target command buffers. This identifies the dominant wait at this granularity, **not saturated ALUs, achieved memory bandwidth, or hardware occupancy**. It does not separate kernel math, memory stalls, or gaps inside GPU work. Counting thousands of price quotes alone does not establish a CPU bottleneck; their counters are not direct quote-duration measurements.

The measured sampled requests all decline neural context and have the same target-round/copy activity as plain. Aggregate timers/capture counts include warm-up; actual measured admissions are separated by `max_new=512` in the raw enrollment log. The one enrolled warm-up is not misrepresented as measured context capture.

## Work actually executed — all six diagnostic arms

Rounds/drafted/accepted are **mixed-source measured serving counters**, not neural acceptance alone. Physical holds/retained prefixes include warm-up and measured requests.

| Case | Arm | Measured requests enrolled | Target rounds | Drafted | Accepted | Physical holds | Retained head positions |
|---|---|---:|---:|---:|---:|---:|---:|
| code,T=0,width1 | plain | 0/1 | 272 | 310 | 239 | 0 | 0 |
| code,T=0,width1 | bf16 | 0/1 | 272 | 310 | 239 | 1 | 3 |
| code,T=0,width1 | bits8 | 1/1 | 279 | 289 | 232 | 8 | 12 |
| code,T=1,width8 | plain | 0/8 | 3739 | 532 | 349 | 0 | 0 |
| code,T=1,width8 | bf16 | 0/8 | 3739 | 532 | 349 | 0 | 0 |
| code,T=1,width8 | bits8 | 0/8 | 3739 | 532 | 349 | 0 | 0 |

All six diagnostic arms returned exactly the same served target tokens as their plain controls and exited cleanly under the original pressure/competitor guard. No diagnostic arm was silently dropped.

The greedy width1 8-bit arm remains a **negative diagnostic case**: it enrolls after the short warm-up, pays eight full nine-query blocks for twelve retained positions, and needs more measured target rounds than plain. Those72 physical query rows are required by the released bidirectional block; truncating the block to the retained prefix would change the model. This is poor paid-work/progress payoff, not evidence that a smaller activation dtype or faster CPU reservation loop solves it. Priced enrollment is not, by itself, proof of positive end-to-end performance. The full unchanged matrix decides readiness.

## BF16 activation versus FP32 reference

Default released head weights are BF16. Explicit8-bit quantizes weights only; **both modes use BF16 activations**. The FP32 native host reference uses the same head checkpoint and BF16 target taps; this is not an FP32 target-server comparison.

| Current-source device | Head weights | Equal FP32-reference picks | Maximum confidence difference | FP32 margin for differing pick |
|---|---|---:|---:|---:|
| M4 Pro | BF16 |9/9|0.00085|0|
| M4 Pro |8-bit|9/9|0.00152|0|
| M5 Ultra |BF16|9/9|0.00080|0|
| M5 Ultra |8-bit|**8/9**|0.00225|0.075|

The final native comparison is reporting-only. Exit0 is never used to assert nine matching picks or an unprinted confidence bound. Served target-token exactness and numerical kernel contracts are separate checks.

## Current production checks and provenance

- M4 original scope:82/82 steps;355 executed tests passed,7 skipped. The core consumer suite408 passed,1 skipped was separately exercised after the stale test deletion and reused from its valid same-source cache. Do not count cache hits as newly executed tests.
- M5 original scope:82/82 steps;767 tests passed,4 skipped.
- Original attention/BF16/norm-tap bounds, released9-row query and requested-prefix invariance, logical cohorts8/16, context lifecycle and planted failure controls remain unchanged.
- M4 production binary SHA256: `20d09fbea8be085a9b9b55d2843517779ac794e914b1bb1cb5d61481cba774ac`.
- M5 production binary SHA256: `7a1210390424df60a2610454499f53018fa99bd466cd5a24757ecd7680a541ab`.
- M5 source archive SHA256: `5c43f984522404e10242ca25f4afc998f8bf6c4a9e505ed7a15ffbb78e564253`.
- Correctly source-bound M5 cache-resume and sampled solo/draft-off/concurrent equality pass at widths1/4/8/16 in both weight modes. All four functional servers exited0 without a forced kill. Earlier mislabeled raw receipts remain unchanged and are not used as qualification.

### Actual M5 cancellation/concurrency outcomes at this source

| Weights | Concurrent slots | Duration seconds | Completed | Cancelled | Wrong replies | Errors |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 6 | 60 | 459 | 251 | 0 | 0 |
| BF16 | 16 | 90 | 936 | 508 | 0 | 0 |
| 8-bit | 6 | 60 | 457 | 248 | 0 | 0 |
| 8-bit | 16 | 90 | 934 | 506 | 0 | 0 |

These are functional stress outcomes, not pooled performance qualification. `serving-functionals.json` preserves the full results, source-labelled proofs and actual shutdown records.

Full qualification completed using the original plain/BF16/8-bit, code/chat, T=0/1, widths 1/4/8/16, 512-token, three-balanced-rotation workflow on both devices. The [complete packets](qualification/README.md) preserve both failures, every comparison, source/binary/model/driver bindings, all original publishable records and unchanged verdict replay. All 31 publication-manifest entries were independently SHA256-checked. No selected winning subset, pooled score, or <1% waiver replaces all 32 comparisons per device.

## GitHub blocker, separate from performance

Current PR metadata says CONFLICTING/DIRTY and has no current-head CI rollup, while [GitHub compare](https://github.com/BlivionIaG/TensorFold/compare/6a5048697e0770a464a4093ff6373f6a73f998b6...cf23762368785b470b3bf7485026e2ea4837ce82) and actual Git ancestry agree that main is already the merge base with behind_by=0. A normal guarded update-branch request returned422 “merge conflict between base and head”. The [maintainer notification](https://github.com/ashhart/TensorFold/pull/1394#issuecomment-6602970480) requests a metadata/check-start refresh, not re-review. No force push, empty commit, assistant PR merge or another moving-main chase is used to hide this.

## Files and historical evidence

- `native-guards.json`: exact source, production/diagnostic binary identities, original contracts and actual FP32 results.
- `private-observed.json`: byte-exact original records for all six diagnostic arms.
- `private-causal-diagnostic.json`: diagnostic-only bindings, actual enrollment/capture/hold attribution and timer definitions.
- `private-instrumentation.patch` and `run-private-probe.py`: exact diagnostic-only reproduction inputs; **not part of production PR code**.
- `github-readiness.json`: contradictory ancestry/mergeability evidence and rejected guarded sync request.
- `source-label-correction.json`: metadata-only correction provenance; raw records are not rewritten.
- `serving-functionals.json`: actual current-source cache/sample/cancellation results and clean shutdowns in both weight modes.
- [Current-source full qualification](qualification/README.md): all 64 comparisons, both complete failed speed verdicts, raw records, frozen driver, source-bound controllers and assembly proof.
- [Remaining-loss GPU/cohort diagnosis](prefix-attribution/README.md): completed exact-history probes on both devices; actual kernel gate, enrollment decisions, physical target row histograms and GPU elapsed, with raw receipts and explicit limits.
- [Complete historical750bca95c report](https://github.com/BlivionIaG/TensorFold/blob/0cece53cc7be9fa67c9c792945ae3b2f17fbc6bf/docs/receipts/lfm2-dspark-review/750bca95c/README.md): all64 original comparisons and raw receipts. Both original speed verdicts failed. They are not renamed as current-source results.

No current MLX comparison was run in this diagnostic. These findings identify TensorFold's own paid work and dominant waits; they do not invent an MLX speedup, achieved-bandwidth measurement, or GPU-occupancy claim.
