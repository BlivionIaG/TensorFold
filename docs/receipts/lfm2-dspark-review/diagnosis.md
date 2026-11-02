# Diagnosis: unproductive speculation and missed M5 dispatch opportunities

**A real model-input bug is corrected; M5 target acceleration has a separate arithmetic obstacle.**
Requested proposal depth changed the drafter's bidirectional input. That correction and its maximum-context
capacity fix passed actual M4/M5 checks. Simply enabling the existing target tensor route is unsafe:
the private dispatch-only experiment failed 50 of 72 unchanged target forward/cache checks.
Neither the corrected model input nor a component timing establishes the full MLX-LM gap.

Working report. This is a causal diagnosis, not a claim of release or performance qualification.

## What is actually wrong

We have been optimizing drafter kernels before establishing how much **additional useful progress** the neural drafter buys over the existing cheap speculation. Numerical correctness and faster component kernels do not establish that adding the drafter reduces work per emitted token.

The current evidence proves a regression against TensorFold's own no-drafter arm. It does **not** establish the cause of a gap against MLX-LM: a matched MLX-LM execution breakdown has not been supplied by this investigation. Those are distinct questions.

## 1. The baseline already gets speculative tokens

The plain arm is not a naive one-token-per-round baseline. Existing copy speculation is enabled. Its accepted/drafted counters must not be mistaken for DSpark neural acceptance.

One concrete example from frozen source `7eb317eda495891a5b6d1525e566d746bd249937`: rotation 0, code, greedy, four concurrent requests, each emitting 512 tokens:

| Arm | Emitted tokens | Sum of per-request verification rounds | Accepted speculative tokens | Proposed speculative tokens |
|---|---:|---:|---:|---:|
| No neural drafter | 2048 | 1660 | 384 | 786 |
| BF16 neural drafter | 2048 | 1654 | 390 | 795 |
| 8-bit-weight neural drafter | 2048 | 1656 | 388 | 806 |

BF16 buys just six additional accepted tokens in this example; 8-bit buys four. Both still pay neural-model overhead. These counters combine proposal sources; they do not isolate neural acceptance. Summed per-request rounds are not a count of physical batched GPU command buffers. This example diagnoses incremental usefulness, not every workload or the full speed gate.

The immediate question is why the neural path contributes so little beyond the cheap path: poor proposal quality, incorrect model inputs, conservative or inaccurate pricing, or work spent on recovery trials. The available aggregate counters do not yet distinguish these causes. Disabling copies to make neural acceptance look better would answer the wrong question.

## 2. A hold is substantial GPU work, not a cheap token lookup

In [dspark_block.zig](../../../zig/src/families/lfm2_moe/dspark_block.zig), `hold` first projects staged context, runs the drafter transformer, projects the hidden rows into the full 128,000-token vocabulary (`hold`, approximately lines 410–460), and applies the Markov chain and confidence work. This must save enough target execution to repay its cost.

Owned context cannot simply be skipped on zero-grant rounds: it is required by later proposals. The opportunity is to eliminate genuinely redundant work or improve useful proposals, not corrupt context retention.

The existing M5 serving traces show additional compute-encoder-active time in drafter arms, not a growing gap between encoders. That is evidence of added GPU-side work. It is **not** proof of full ALU occupancy, tensor-unit utilization or memory-bandwidth saturation; per-phase attribution remains missing.

## 3. Target M5 routing crosses an arithmetic-compatibility boundary

In [forward.zig](../../../zig/src/families/lfm2_moe/forward.zig):

- `tile_rows` is 33.
- `segments` selects prompt/tiled mode with `(!record and head == .last) or total >= tile_rows`.
- `Encoder.nax` returns null outside that mode.
- `Encoder.projectAll` takes `encodeTwin` early when outside that mode.

The target admits at most 32 verification rows (`state.zig`), so the 33-row threshold makes this M5 NAX route unreachable for every admitted recorded verification call. `encodeTwin` can itself select an older SIMD-group MMA window kernel; bypassing NAX does **not** mean every such operation is pure scalar arithmetic.

The drafter uses NAX from five rows for its vocabulary head and nine for other wide products. Copying
those thresholds into the target is **not a safe correction**: production source
`174a74681920cbaefc3d3e65a9cd18c5e0f15a5a` passed the unchanged actual M5 target oracle **72/72**;
private dense-only candidate `b99a794df` passed its build/family checks but only **22/72** target
forward/cache checks. Expert and prompt routing were retained. The candidate is excluded from publication.

The installed comparison is MLX-LM 0.32.2 with MLX 0.32.4. Version-matched [MLX dense dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L89-L125) selects by shape and architecture, rather than a verification flag. Static inspection also found native target serialization of independent projections, eager state/logits commits, attention dispatch tied to prompt mode and a redundant CPU argmax in the one-token queue path. Their runtime contributions are not measured; rollback state cannot be deleted, and the redundant argmax is not on the greedy verification path.

The first newly eligible seven-row window changes only the six-bit tied head. A real-input localization
replayed its exact final-normalization BF16 rows and released head weights through one-row NAX,
seven-row NAX and the established one-row route, with a separately unpacked FP64 reference:

| Head computation on identical captured inputs | Differences from canonical BF16 outputs | Printed worst normalized FP64 error |
|---|---:|---:|
| Canonical one-row route | 0 / 896,000 | 1.151e-3 |
| NAX, original epilogue, one/seven rows | 57 / 896,000 | 1.151e-3 |
| NAX, canonical nested-FMA epilogue, one/seven rows | 4 / 896,000 | 1.151e-3 |

Within each NAX variant, the two widths match byte-for-byte: this exposes a **cross-route arithmetic
mismatch**, not a seven-row packing failure. The error is normalized by
`sum_groups(abs(s * D) + abs(b * T))`; the printed summaries have only three significant figures.
They do not make the BF16 outputs identical.

A second private experiment changed only the group64 epilogue to the canonical nested FMA order.
Its reduced mismatch count isolates accumulation order as a major cause, but not the entire cause.
The remaining first difference is immediately across a BF16 rounding boundary: FP64
`-2.976563190`, canonical `-2.984375`, NAX `-2.96875`. This neither establishes a target token change
nor licenses waiving the original target oracle.

The source has different group evaluation graphs
([canonical twin](../../../zig/kernels/metal/core/tile_qmv.metal),
[tensor product](../../../zig/kernels/metal/core/affine_mm.metal)):

```cpp
// Established decode group:
acc = fma(b, sum, fma(s, dot, acc));
// Current tensor group:
acc += s * dot + b * sum;
```

The matrix primitives also differ. Six-bit codes have no four-bit `+128` offset, so that offset cannot
explain this first head-only case. Passing ordinary FP64-error bounds is not permission to replace the
target's canonical per-row arithmetic. MiniMax explicitly maintains the same target/drafter separation;
Gemma's fixed tensor schedule is a useful design precedent, not a drop-in equivalent to LFM2 decode.
The remaining discriminator is the group dot/sum primitive after epilogue alignment, not another lower
dispatch threshold. The shared shader experiment is diagnostic only and is not published.

## 4. Fusion can make utilization worse

The drafter already encodes independent gate/up projections in a concurrent encoder, then places barriers at dependencies. Eliminating one dispatch is not automatically an improvement: a fused kernel can increase live registers, reduce available parallel work or serialize previously overlapping products.

Two new fusion candidates passed actual M5 numerical suites and completed six alternating real-weight pairs each. Their current general policies are rejected:

| Private fusion policy | Measured eligible shapes | Paired median GPU change |
|---|---|---:|
| 8-bit low-row fusion | All nine shapes slower in all six pairs | +10.5 to +65 microseconds |
| BF16 partial-tile fusion | Rows 8/9/12/24 slower in all six pairs | +69.5 to +93.5 microseconds |
| BF16 full-tile cases | Some 32-row shapes improve; not a general-policy win | −24 to −31 microseconds |

These are component timings, not a serving or MLX-LM comparison. They establish that reducing dispatch count made important paths slower. Register pressure, padded work and lost projection overlap remain possible mechanisms, not measured occupancy findings. Neither experiment is in the published fixes.

## 5. Confirmed released-model input discrepancy: the query block is shortened

The released [checkpoint configuration](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/blob/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55/config.json) has **nine query rows**. Its five layers use bidirectional/full attention. The pinned official [proposer](https://github.com/sgl-project/sglang/blob/488869c2d/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L354-L411) constructs the configured query block independently of a scheduler's eventual retained prefix.

Before correction `e06ce301`, native `hold` set its query width to the largest requested proposal depth in the current physical pass. A one-proposal request got only the pending row; a three-proposal request got pending plus two masks. The same short request beside a nine-proposal request got all nine query rows. Earlier queries attend these later mask rows, so shortening the block changed the attention computation and could make a proposal depend on its batch companions.

This is a **reproduced model-input discrepancy**, not a demonstrated cause of the acceptance regression. On actual M4 released weights, private before-source `da8029b5b09317c5b668a3b7e33cbf41c92c95c9` failed the full first-row base-logit prefix invariant (`DraftPrefixDependsOnGrant`). Official SGLang permits an explicitly configured alternative block size; what is unsupported is treating a dynamically shortened noncausal block as prefix-preserving pruning of the configured nine-row computation.

Correction `e06ce30116c0a77017136ebacc4e8f15ce671def` separates configured query width from returned proposals and updates physical packing, clean/context funding and the reference together, without increasing grants. The funding regression rejected the fourth short head when four fixed blocks require a second physical pass. The core suite changed from one new failing regression to 391 passing tests and one platform skip. On M4, requested prefixes 1/3/9 in both precisions preserve first-row base-logit bytes, first proposal and confidence, and all FP32 picks matched in these cases. Real 8/16-logical-cohort controls passed. M5 passed the original numerical/resource contracts and both native precision scenarios, including the new invariants; its BF16 reference matched 9/9 picks, while 8-bit matched 8/9. These are source-bound correctness checks, not serving or MLX-LM speed evidence.

BF16 activations are compared with a separately implemented FP32 host reference; explicit 8-bit mode changes weights, not activation dtype. Previously, that reference used native captures and reproduced the shortened query block. It now runs the full configured query block and compares the requested proposal prefix. It still does not independently establish every target-feature value against an official target implementation.

The official audit found no indexing, raw-versus-normalized capture, prompt/verified temporal shift, RoPE-layout, matrix-orientation or Markov-input mismatch in the inspected paths. It did find a drafter RMSNorm rounding-order difference and a different sampled-proposal policy; their effect remains unproven. Those are separate from the query-block correction, and do not justify silently changing target mathematics.

**Maximum-context capacity defect reproduced and corrected:** the full query block can extend beyond
the target's 131,072-key span. Partials and merge-local arrays previously held only 32 chunks of 4,096
keys. On private before-source `a6164114c3e5cc57cbfa83fa7a072eb9d295732b`, an actual GPU fixture at
131,077 visible keys overwrote the next allocated scratch row (`0x7fc12345` became zero). Correction
`174a74681920cbaefc3d3e65a9cd18c5e0f15a5a` reserves a drafter-only 33rd chunk; target attention and
the advertised context window remain unchanged. M4 and M5 passed the guard-row and analytic tail-value
regression, original attention/BF16 numerical and resource bounds, and actual released-model scenarios
in both precisions. This boundary fixture is not a new full-model 131k-context qualification.

The corrected-capacity source also passed all six M4 requested-prefix scenarios (1/3/9, BF16/8-bit):
every requested pick matched its FP32 reference in these cases. M5 BF16 matched 9/9; 8-bit matched 8/9,
with largest differing-pick reference margin 0.075 and worst confidence difference 0.00225. These are
drafter-reference results, not permission to tolerate different final target tokens.

## Work order from here

1. Compare independently captured official/native target features and drafter intermediates. The FP32
   arithmetic reference alone cannot establish that every model input is the intended one.
2. Develop a target tensor replacement compatible with the established one-row computation; preserve
   M4 behavior and the original byte/token/cache checks. The simple threshold-copying experiment is rejected.
3. Attribute drafter spending to context projection, transformer, vocabulary head and Markov work, and
   isolate neural incremental acceptance from cheap copies. Per-phase critical-path and hardware occupancy
   attribution remain unmeasured; encoder-active time is not an ALU-utilization counter.
4. Retain only changes that remove identified waste. Resume full qualification after a causal fix,
   rather than using repeated full matrices as diagnosis. Publication does not mean merge-ready.

## Source and publication provenance

- Last complete serving evidence: `7eb317eda495891a5b6d1525e566d746bd249937`; all token checks passed, but 21 of 32 speed comparisons failed. This report does not relabel that evidence to newer heads.
- Initial execution-path inspection: `8fb1c35c417ffc1011ad62220ad6b23a4141accb`. Fixed-query implementation and M4/M5 checks belong to `e06ce30116c0a77017136ebacc4e8f15ce671def`; subsequent documentation or comment-only changes do not inherit a different source label.
- Maximum-context before/after: `a6164114c3e5cc57cbfa83fa7a072eb9d295732b` → `174a74681920cbaefc3d3e65a9cd18c5e0f15a5a`. The before M4 log SHA-256 is `27d50388cf2fe6fa8b7f16e39015758879acace88c55df8a6a08119357dc02c6`; the after M4 build/units log is `54418ea32f23fd47d695095bac21fac8bf406476ad495780667d2cee228f0bf8`, and the M5 build/units log is `6f6079c30e6f82c7f7f67d3a3a99017f9b0b0394533450d72d624b931c01a5ec`. Original numerical bounds and controls were not relaxed.
- #1391: `caf9282b8c1f85d4a8dd7f66c09feaf89053d977`; #1392: `5d81437d0f16c9c52ed60f095e4c26a16c498592`; #1393: `5e7ae48da075f4e318c4d137aa8d3f53cd80f707`. All three were fast-forward pushed after their own 76/76 scoped build/check steps (251 passed, five platform skips), original attention/BF16 contracts and actual M4 BF16/8-bit native scenarios. Each mode matched 9/9 FP32 picks and passed prefix/split/lifecycle controls; the depth-only stage does not claim confidence coverage. Their code-parent CPU suites passed 341/366/365 respectively, each with one skip.
- #1394 correctness/report publication: `5215a8b3588e9ad8992846111250e505081a9935`. Its own native/CPU build passed 60/60 steps, 391 consumers and one platform skip, plus actual CLI help. Production GPU receipts retain `174a7468`; that source also passed the unchanged M5 target oracle 72/72. All four retained 424 inherited lean findings and zero new.
- Exact-head CI on those four publications was observed cancelled, not green: #1391 has one successful formatting check and ten cancellations; the other three have eleven cancellations each. Workflow concurrency is scoped by PR/ref, so the configuration does not explain cross-PR cancellation. Cause is unproven. No gate, workflow, test skip or permission retry was introduced.
- Target arithmetic localization: private probe `751656d464635bd38127b39bfc1e660b034be033`, log SHA-256 `4969548bc69dc94b7e8812101b55f8c3d1ecb9e991d468837ebcd52c46724b76`; packet `/Users/kletorch/tf-lfm2-test/r18-head-probe-751656d46` on the M5. Input: the first seven tokens of the existing exact fixture, zero prior context, canonical target forward, its final-normalization BF16 rows; head 128000×2048, six-bit/group64. The original independent FP64 unpack/reference was reused. This is localization, not a replacement for the failed full oracle.
- Epilogue-only localization: private `5b90d6232f3ef1b6c640b80419ad23dba185e8e5`, log SHA-256 `944eda8b134563196c9c3e66798a2254e8f2d1a5d7163e2e55d38664f5a6ad68`, binary SHA-256 `0187669099bb50aa72b3143cc598441681b724f398e21f01c3ad1384fc8c9a8a`; M5 packet `/Users/kletorch/tf-lfm2-test/r18-head-fma-5b90d6232`. Same captured inputs, every vocabulary output checked, no source-threshold or oracle change.
- No new full serving matrix is being used as a substitute for the missing causal explanation.
