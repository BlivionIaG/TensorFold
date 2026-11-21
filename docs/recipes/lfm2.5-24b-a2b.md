# LFM2.5-24B-A2B, native Metal and CUDA

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

The drafter runs the released bf16 matrices as they are (`--drafter-bits 0`, the default). `--drafter-bits 8` instead
quantizes them at load to 8-bit MLX affine codes in groups of 64, about 5% faster on one stream and 361 MiB of weights
against 625 MiB; the server logs which mode runs, and 4 bits is refused. Both run on the family's kernels
(bf16 on `lfm2_bf16_rows`, `lfm2_bf16_mma` and `lfm2_markov_scan_bf16`, 8-bit on `core/tile_qmm`); the q and k rows are reordered so that the target's half-split RoPE kernel gives the
checkpoint's interleaved RoPE. Its own kernels are `lfm2_tap`, `lfm2_attn_block` and `lfm2_attn_block_merge`
(non-causal block attention, the target's attention kernels with a template flag), `lfm2_markov_scan` and
`lfm2_markov_pick` (a Markov step) and `lfm2_draft_ids` (a hold's drafts into the next verify's rows on the GPU, so
the host does not wait for a hold that fits one pass). Each kept prompt state keeps the drafter's context too, so a
drafting stream resumes prompts. The lane core prices drafts across the admitted stream counts.

`tf-kernel-contract --only lfm2` checks these kernels. `tf-lfm2-dspark MODEL DRAFTER TOKENS [ROWS]` runs the drafter
against an fp32 host reference of the released checkpoint (the same taps and the target's embedding and head),
prints each draft against the reference's pick with the margin between the reference's two best tokens, and times
the drafter's passes; a near tie can pick the other token.

### Verify schedule

The drafter also carries a confidence head: a linear map of a draft's final block row and `W1[previous token]`
(2,304 inputs) to one logit, through a sigmoid (`lfm2_dspark_confidence`). It does not change the drafts.
The verify rows follow it: a chain's chance at draft j is the product of its first j confidences. Each round the
lane core's window allocator takes the likeliest drafts over all streams while the expected tokens a ms rise, so
each stream gets its own verify length, and the next hold drafts as deep as the stream's grant. A hold still
running on the GPU is priced by its lane's last confidences, so the host does not wait for it. A checkpoint
without the head verifies by the core's depth rule, from each stream's measured acceptance by depth.

A stream with a copy ahead verifies the copy and asks no hold, and a drafting stream prices its copy
windows by its own copies. The lane core also weighs the hold pass itself: the drafter times holds of 1 to 16
blocks at every admissible depth at load and corrects those prices toward each settled hold's GPU time.
Prices interpolate neighboring measured depths, preserving the cost changes at kernel transitions.

`tf-lfm2-dspark` also prints the confidence
head against its fp32 reference and times shared holds of 2 to 16 streams.

The loader reads the checkpoint one matrix at a time and keeps the released bf16 matrices by default, or the
8-bit codes with `--drafter-bits 8`; a stream's drafter lane is one
Metal buffer (its layers' keys and values, then its staged rows), and its bytes join the streams the load limit
admits. At load the drafter logs its weights, scratch and a stream's bytes. `tf-lfm2-dspark` also counts host
allocations over twenty served drafter rounds, which must be none.

A hold runs on one concurrent encoder with a barrier only where a step reads an earlier step's writes: q, k and v,
and gate and up, share one sums pass, and the blocks of a shared hold attend side by side. An 8-bit projection takes
the scalar twin up to 8 rows, the window kernel past them where the GPU has no tensor units, and the tensor units
from 9 rows (the head from 5) where it has them. Held blocks use one bf16 16-row group, then the existing
64x64 tiles past 16 rows, avoiding a second weight pass for shared holds of 17 to 32 rows. Context ingestion
keeps the original row kernel through 32 rows, preserving the persistent context projections.
When one lane's stage of 128 verified rows fills, its absorb pass takes
every lane's staged rows, so the lanes' stages fill together.

Shared verification past the 16-row qualified window uses measured increments
from the shared forward path through 32 rows, anchored at the 16-row window
price. This calibrates the batch cost without widening a stream's qualified
window. The wide-batch receipt records the throughput gains, the accepted
four-stream tradeoff, startup/footprint cost and hardware coverage limits.

## CUDA

`lfm2_moe` also serves the 4-bit export above (same revision, `667408d494bde191c2cb9dd4f4549827816b9275`) on one
NVIDIA GPU, through the lane host. Checked on a GB10 (sm_121).

```bash
zig build native -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm2.5-24b-a2b-4bit" --port 8080 --parallel 8
```

The default window is 32,768 tokens (`--context`), and `--parallel` holds up to 128 streams if the memory budget fits
them. Streams lease 512-position pages of keys and values from one pool as they need them
(`TENSORFOLD_LFM2_KV_PAGES=0` gives each stream fixed pages instead). The prompt cache keeps a chat's state at its
history marks, the conv states and the K/V rows before the mark, in pages allocated once inside its budget
(`--prompt-cache-gib`). The family reads MLX affine 4-bit weights in groups of
64, with the embedding at 4, 6 or 8 bits. The 6- and 8-bit exports are refused by name and stay on Metal. Drafts
are the lane core's copy drafts from the prompt's own text; there is no draft head yet.

Most launches are shared CUDA operators: the affine linears and experts, the norms, `qk_rows`, the MoE combine, the
pick and the draws. The family adds three kernels: the router (fp32 weights, sigmoid plus bias, top-4) in
`lfm2_moe.cu`, and the 3-tap short conv and its state commit in `liquid_conv.cuh`. It also adds two instances of shared
templates at head dim 64: `pattn_kernel<64, 8, 4, 8>` for prompt rows, and `tf_paged_attn_chunk_segs_64` with
`tf_paged_attn_merge_segs_64` (`paged_attn_segs.cuh`) for decode rows. All of them are under the kernel contract
(see [the contract](../kernel-contract.md#coverage)).

Receipt (GB10, nvcc 13.0, `--parallel 8`, greedy):

- drafted == plain and concurrent == solo on 8 of 8 prompts, and the token hashes are the same after a restart;
- a float64 forward of the same weights on the CPU (`tools/lfm2_moe/ref64.py`) picks the engine's greedy token on
  421 of 435 teacher-forced prompt rows and 123 of 128 reply rows; `tools/lfm2_moe/REF64.md` lists every miss with
  both tokens and margins;
- decode, total tok/s: 87 at 1 stream, 194 at 4, 303 at 8;
- cold TTFT: 0.32 s at 2.8k prompt tokens, 1.32 s at 11.2k.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint). Use, changes and redistribution are allowed; commercial use is licensed only to
an organisation under US$10 million in annual revenue, and a non-profit may use it for research or other
non-commercial purposes regardless. Read the licence before serving it commercially.
