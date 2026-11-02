# Diagnosis: unproductive speculation and missed M5 dispatch opportunities

**Two confirmed implementation problems:** requested proposal depth changed the drafter's bidirectional
input, and target verification could not reach the M5 dense/head NAX route. The first correction passed
actual M4 and M5 model checks on `e06ce30116c0a77017136ebacc4e8f15ce671def`; the dispatch correction
remains private and unqualified. Neither result establishes an acceptance gain or the full MLX-LM gap.

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

## 3. Target dense projections have a concrete M5 routing gap to investigate

In [forward.zig](../../../zig/src/families/lfm2_moe/forward.zig):

- `tile_rows` is 33.
- `segments` selects prompt/tiled mode with `(!record and head == .last) or total >= tile_rows`.
- `Encoder.nax` returns null outside that mode.
- `Encoder.projectAll` takes `encodeTwin` early when outside that mode.

The target admits at most 32 verification rows (`state.zig`), so the 33-row threshold makes this M5 NAX route unreachable for every admitted recorded verification call. `encodeTwin` can itself select an older SIMD-group MMA window kernel; bypassing NAX does **not** mean every such operation is pure scalar arithmetic.

The drafter already uses a more selective policy: quantized vocabulary-head projections can use NAX from five rows, other wide projections from nine, and narrow projections retain the scalar route longer. That inconsistency is a specific candidate for correcting target dispatch. It does not prove that copying those thresholds to the target will be faster or numerically interchangeable. M4 fallback and the original numerical/token contracts must remain intact.

The installed comparison is MLX-LM 0.32.2 with MLX 0.32.4. Version-matched [MLX dense dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L89-L125) selects by shape and architecture, rather than a verification flag. Static inspection also found native target serialization of independent projections, eager state/logits commits, attention dispatch tied to prompt mode and a redundant CPU argmax in the one-token queue path. Their runtime contributions are not measured; rollback state cannot be deleted, and the redundant argmax is not on the greedy verification path.

## 4. Fusion can make utilization worse

The drafter already encodes independent gate/up projections in a concurrent encoder, then places barriers at dependencies. Eliminating one dispatch is not automatically an improvement: a fused kernel can increase live registers, reduce available parallel work or serialize previously overlapping products.

Two new fusion candidates passed actual M5 numerical suites and completed six alternating real-weight pairs each. Their current general policies are rejected:

- The 8-bit fusion regresses all nine eligible low-row shapes in every pair, with paired median GPU additions of 10.5–65 microseconds. Its inactive higher-row controls do not show the same systematic loss.
- The BF16 tensor fusion regresses the tested partial-tile shapes (8, 9, 12 and 24 physical rows) in every pair, adding 69.5–93.5 microseconds by paired median. Some full 32-row shapes improve by roughly 24–31 microseconds, but that does not justify the broad policy.

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

1. Publish the stage-preserving query/capacity corrections. Port CPU suites passed 341/366/365 tests for #1391/#1392/#1393 respectively, each with one platform skip; all retain zero new lean findings. Their own-head native checks and publication are separate from the cumulative M4/M5 receipts.
2. Compare independent official/native intermediate captures, then correct the target M5 dispatch gap without changing M4 behavior or weakening numerical checks. The initial dispatch-only prototype remains private and unqualified.
3. Attribute drafter cost to context projection, transformer, vocabulary head and Markov work; relate spending to incremental accepted tokens over copies.
4. Retain only changes that remove identified waste. Resume full qualification after a causal fix, rather than using repeated full matrices as diagnosis.
5. Continue publishing independently verified correctness fixes. Publication does not mean merge-ready.

## Source and publication provenance

- Last complete serving evidence: `7eb317eda495891a5b6d1525e566d746bd249937`; all token checks passed, but 21 of 32 speed comparisons failed. This report does not relabel that evidence to newer heads.
- Initial execution-path inspection: `8fb1c35c417ffc1011ad62220ad6b23a4141accb`. Fixed-query implementation and M4/M5 checks belong to `e06ce30116c0a77017136ebacc4e8f15ce671def`; subsequent documentation or comment-only changes do not inherit a different source label.
- Maximum-context before/after: `a6164114c3e5cc57cbfa83fa7a072eb9d295732b` → `174a74681920cbaefc3d3e65a9cd18c5e0f15a5a`. The before M4 log SHA-256 is `27d50388cf2fe6fa8b7f16e39015758879acace88c55df8a6a08119357dc02c6`; the after M4 build/units log is `54418ea32f23fd47d695095bac21fac8bf406476ad495780667d2cee228f0bf8`, and the M5 build/units log is `6f6079c30e6f82c7f7f67d3a3a99017f9b0b0394533450d72d624b931c01a5ec`. Original numerical bounds and controls were not relaxed.
- #1391 was fast-forward pushed to `08196a4b48badceef244a96353b1ad5932168de5`. Its PR description distinguishes production checks on parent `ed731bfa` from the formatting successor's own CPU checks and pending exact-head CI.
- #1392 was fast-forward pushed to `554e83b3c502c8f5a87c5f7327db8d0212dec52f`. Production-code checks belong to parent `74438fcc`; the formatting successor's own CPU suite passed. Its PR description preserves that distinction.
- #1393 was fast-forward pushed to `55f3095ed172cd59872df450ae63cd70cefe43e4` after its own complete build/check invocation and actual M4 native scenarios. The new query-contract correction is being ported to all three published stages; those old-head receipts do not validate the ports.
- No new full serving matrix is being used as a substitute for the missing causal explanation.
