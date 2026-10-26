# LFM2.5-24B-A2B, native Metal

This recipe covers [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit)
at revision `667408d494bde191c2cb9dd4f4549827816b9275`. The numbers below are measured on this 4-bit export only.

```bash
hf download LiquidAI/LFM2.5-24B-A2B-MLX-4bit \
  --revision 667408d494bde191c2cb9dd4f4549827816b9275 --local-dir "$HOME/models/lfm2.5-24b-a2b-4bit"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm2.5-24b-a2b-4bit" --port 8080 --parallel 16
```

A Mac with less memory may not hold 16 streams at the checkpoint's 131,072-token window; the server then names how
many fit. `--context 40960` holds 16 on a 48 GB M4 Pro.

## Shared rounds

Every stream's tokens go through the lane rounds. A round of several streams is one forward over all their rows.
In it each layer's short conv (30 layers) and each attention layer's qk, attention and merge (10 layers) run once
over every row (`lfm2_segments.metal`), from a table of each stream's cache addresses, row offsets and key counts
(`segments.zig`). Each row's arithmetic is the per-stream kernel's, so a reply equals its solo run. Prompt calls and
one-stream windows keep the per-stream launches. A greedy row takes the GPU's pick of its logits (`lfm2_argmax`);
sampled and masked rows draw on the host. The dense twin takes its MMA window only for matrices of 4,096 outputs or
more (the short conv's input and the head); narrower ones keep the scalar passes.

`zig build test-lfm2` runs the family's tests. `tf-kernel-contract --only lfm2` checks the shared-round kernels and
the pick, and `tf-lfm2-exact` compares windows and shared rounds with one-row decode on the real weights.

## Measured

Greedy, 256 tokens, 16 public prompts, a different one a stream, 3 reps in each of 3 alternated rounds on an M5 Ultra;
aggregate tok/s, the range of the rounds' medians. Before is main at 0d24ecb3 (each stream's own launches), after adds
the shared-round kernels on it. The M4 Pro is not yet measured on a quiet machine.

| Streams | before | after |
| --- | --- | --- |
| 1 | 258.5-259.2 | 266.5-267.7 |
| 4 | 483.5-486.5 | 565.2-566.1 |
| 8 | 551.5-551.9 | 732.2-735.6 |
| 16 | 660.7-661.9 | 922.5-923.5 |

Every reply was token for token equal before and after, and equal to its solo run and to `"draft": false`.
Cold prefill at 2k, 8k and 32k tokens and peak memory did not change.

## DSpark drafts

LiquidAI's DSpark drafter for this model is
[`LiquidAI/LFM2.5-24B-A2B-DSpark`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/tree/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55)
at revision `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55` (327.7M bf16 parameters, 655 MB). Its weights are under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/blob/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55/LICENSE);
TensorFold does not ship them, so read the license before use.

```bash
hf download LiquidAI/LFM2.5-24B-A2B-DSpark --revision e052e5b1dd103c38cdd4f8ddc068d7dea4586f55 \
  config.json model.safetensors LICENSE README.md --local-dir "$HOME/models/LFM2.5-24B-A2B-DSpark"
zig-out/native/bin/tensorfold-native serve "$MODEL_DIR" --port 8080 --parallel 16 \
  --drafter "$HOME/models/LFM2.5-24B-A2B-DSpark"
```

Without `--drafter` the engine serves as before. `"draft": false` on a request, or `--no-drafts`, turns the drafter
off; the lane core verifies every draft, so a drafted reply equals its `"draft": false` reply token for token.

The drafter has no embedding and no head of its own:

- Context: the target's rows after layers 2, 10, 18, 26 and 34 (the backend's taps, `lfm2_tap`), through `fc` and
  `hidden_norm`, give each of its five layers' context keys and values (k_norm, then RoPE at the row's position).
- Block: the pending token and mask tokens run five Qwen3 layers with non-causal attention over the context and the
  block, then the target's head. A block is as long as the deepest chain its pass holds, at most the nine rows the
  drafter was trained on.
- Markov head: row i's logits plus `W2 W1[previous token]` (rank 256); draft i is their argmax.

Its bf16 linears are quantized at load to 8-bit MLX affine codes in groups of 64 and run on the family's kernels
and `core/tile_qmm`; the q and k rows are reordered so that the target's half-split RoPE kernel gives the
checkpoint's interleaved RoPE. Its own kernels are `lfm2_tap`, `lfm2_attn_block` and `lfm2_attn_block_merge`
(non-causal block attention, the target's attention kernels with a template flag), `lfm2_markov_scan` and
`lfm2_markov_pick` (a Markov step) and `lfm2_draft_ids` (a hold's drafts into the next verify's rows on the GPU, so
the host does not wait for a hold that fits one pass). Each kept prompt state keeps the drafter's context too, so a
drafting stream resumes prompts. Shared rounds take the lane core's shared-round guard: they draft only while
drafted rounds measure faster than plain ones at that stream count.

`tf-kernel-contract --only lfm2` checks these kernels. `tf-lfm2-dspark MODEL DRAFTER TOKENS [ROWS]` runs the drafter
against an fp32 host reference of the released checkpoint (the same taps and the target's embedding and head),
prints each draft against the reference's pick with the margin between the reference's two best tokens, and times
the drafter's passes; a near tie can pick the other token.

### Verify schedule

The drafter also carries a confidence head: a linear map of a draft's final block row and `W1[previous token]`
(2,304 inputs) to one logit, through a sigmoid (`lfm2_dspark_confidence`). It does not change the drafts.
`--draft-verify` picks how they are verified:

- `scheduled` (the default): a chain's chance at draft j is the product of its first j confidences. Each round the
  lane core's window allocator takes the likeliest drafts over all streams while the expected tokens a ms rise, so
  each stream gets its own verify length, and the next hold drafts as deep as the stream's grant. A hold still
  running on the GPU is priced by its lane's last confidences, so the host does not wait for it.
- `depth`: the lane core's depth rule, from each stream's measured acceptance by depth.
- `fixed`: every round verifies the whole block, as LiquidAI's published benchmarks did; a shared round still fits
  32 rows.

With `scheduled` a stream with a copy ahead verifies the copy and asks no hold, and a drafting stream prices its copy
windows by its own copies. The lane core also weighs the hold pass itself: the drafter times holds of 1 to 16
blocks at load and corrects those prices toward each settled hold's GPU time. A lone stream holds drafts only while
its drafted rounds measure faster than plain ones (`zig/src/core/lanes/lone_guard.zig`); shared rounds keep the
shared-round guard.

`--verify-costs FILE` replaces the costs timed at load with a table from `tf-lfm2-verify-costs MODEL FILE [HISTORY
[IDS...]]` (lines `streams rows_each total_rows ms`), measured on the serving Mac. With token id files each stream
reads its history and verify rows from its own part of the ids, so its rows route its own text's experts. Both flags
are refused unless an `lfm2_moe` checkpoint is served with `--drafter`. `tf-lfm2-dspark` also prints the confidence
head against its fp32 reference and times shared holds of 2 to 16 streams.

The loader reads the checkpoint one matrix at a time and keeps only the 8-bit codes; a stream's drafter lane is one
Metal buffer (its layers' keys and values, then its staged rows), and its bytes join the streams the load limit
admits. At load the drafter logs its weights, scratch and a stream's bytes. `tf-lfm2-dspark` also counts host
allocations over twenty served drafter rounds, which must be none.

A hold runs on one concurrent encoder with a barrier only where a step reads an earlier step's writes: q, k and v,
and gate and up, share one sums pass, and the blocks of a shared hold attend side by side. A projection takes the
scalar twin up to 8 rows, the window kernel past them where the GPU has no tensor units, and the tensor units from
9 rows (the head from 5) where it has them. When one lane's stage of 128 verified rows fills, its absorb pass takes
every lane's staged rows, so the lanes' stages fill together.
