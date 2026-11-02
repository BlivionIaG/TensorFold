# Diagnosis: unproductive speculation and missed M5 dispatch opportunities

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

Native `hold` instead sets its query width to the largest requested proposal depth in the current physical pass. A one-proposal request gets only the pending row; a three-proposal request gets pending plus two masks. The same short request beside a nine-proposal request gets all nine query rows. Earlier queries attend these later mask rows, so shortening the block changes the attention computation and can make a proposal depend on its batch companions.

This is a **confirmed source-level discrepancy**, not a demonstrated cause of the acceptance regression. Official SGLang permits an explicitly configured alternative block size; what is unsupported is treating a dynamically shortened noncausal block as prefix-preserving pruning of the configured nine-row computation.

The first correction is to separate configured backbone query width from returned proposal depth, and adapt physical pass packing, clean/context funding and the reference accordingly. It must not simply increase scheduler grants.

BF16 activations are compared with a separately implemented FP32 host reference; explicit 8-bit mode changes weights, not activation dtype. However, that reference uses native target captures and reproduces the shortened query block. Agreement therefore did not expose this input-shape discrepancy.

The official audit found no indexing, raw-versus-normalized capture, prompt/verified temporal shift, RoPE-layout, matrix-orientation or Markov-input mismatch in the inspected paths. It did find a drafter RMSNorm rounding-order difference and a different sampled-proposal policy; their effect remains unproven. Those are separate from the query-block correction, and do not justify silently changing target mathematics.

## Work order from here

1. Correct the released-model query-block discrepancy first. Keep the configured query width independent of retained proposals; update packing and funding together, and prove proposal-prefix behavior before/after.
2. Compare independent official/native intermediate captures, then correct the target M5 dispatch gap without changing M4 behavior or weakening numerical checks. The initial dispatch-only prototype remains private and unqualified.
3. Attribute drafter cost to context projection, transformer, vocabulary head and Markov work; relate spending to incremental accepted tokens over copies.
4. Retain only changes that remove identified waste. Resume full qualification after a causal fix, rather than using repeated full matrices as diagnosis.
5. Continue publishing independently verified correctness fixes. Publication does not mean merge-ready.

## Source and publication provenance

- Last complete serving evidence: `7eb317eda495891a5b6d1525e566d746bd249937`; all token checks passed, but 21 of 32 speed comparisons failed. This report does not relabel that evidence to newer heads.
- Execution-path inspection: `8fb1c35c417ffc1011ad62220ad6b23a4141accb` plus local generic context-pricing/adapter fixes. The cited target routing and hold code are unchanged by those local generic fixes.
- #1391 was fast-forward pushed to `08196a4b48badceef244a96353b1ad5932168de5`. Its PR description distinguishes production checks on parent `ed731bfa` from the formatting successor's own CPU checks and pending exact-head CI.
- #1392 was fast-forward pushed to `554e83b3c502c8f5a87c5f7327db8d0212dec52f`. Production-code checks belong to parent `74438fcc`; the formatting successor's own CPU suite passed. Its PR description preserves that distinction.
- Additional allocation and generic context-pricing fixes have passing before/after regressions. Their remaining publication checks are separate from the architectural diagnosis.
- No new full serving matrix is being used as a substitute for the missing causal explanation.
